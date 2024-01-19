from ufl import (grad,sin,cos,as_matrix,SpatialCoordinate,FacetNormal,Measure,as_tensor,indices,VectorElement,MixedElement,TrialFunction,TestFunction,split)
from dolfinx.fem import (Expression,TensorFunctionSpace,assemble_scalar,form,Function,FunctionSpace,VectorFunctionSpace)
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from petsc4py import PETSc

import os
# # print(os.environ)
os.environ['SCIPY_USE_PROPACK'] = "1"
from scipy.sparse.linalg import svds,inv
import sparseqr
from scipy.sparse import csr_matrix,hstack

from dolfinx.plot import create_vtk_mesh
import pyvista

from FROOT_BAT.material import *
from FROOT_BAT.utils import get_vtx_to_dofs,get_pts_and_cells

#TODO: find cross-sectional properties about the zero point
#TODO: allow user to specify a point to find xs props about
#TODO: provide a method to translate between different xs values

class CrossSection:
    def __init__(self, msh, material ,celltags=None):
        #analysis domain
        self.msh = msh
        self.ct = celltags
        #dictionary of material properties 
        self.material = material
        '''
        dictionary defined in the following manner:
        {MATERIAL:{
            TYPE: 'ISOTROPIC', 'ORTHOTROPIC', etc
            MECH_PROPS: {}
            DENSITY: float } }
        ....
        '''
        
        #geometric dimension
        self.d = 3
        #number of materials
        self.num_mat = len(self.material)
        #tuple of material names and ids
        self.mat_ids = list(zip(list(self.material.keys()),list(range(self.num_mat))))
        #indices
        self.i,self.j,self.k,self.l=indices(4)
        self.p,self.q,self.r,self.s=indices(4)
        self.a,self.B = indices(2)
        
        #need to think a bit furth about mutlimaterial xs's here
        #TODO: rn, hard coded to just use the first material density
        self.rho = self.material[self.mat_ids[0][0]]['DENSITY']

        #integration measures (subdomain data accounts for different materials)
        if celltags==None:
            self.dx = Measure("dx",domain=self.msh)
        else:
            self.dx = Measure("dx",domain=self.msh,subdomain_data=self.ct)
        self.ds = Measure("ds",domain=self.msh)
        #spatial coordinate and facet normals
        self.x = SpatialCoordinate(self.msh)
        self.n = FacetNormal(self.msh)
        
        #compute area
        self.A = assemble_scalar(form(1.0*self.dx))
        #compute average y and z locations 
        self.yavg = assemble_scalar(form(self.x[0]*self.dx))/self.A
        self.zavg = assemble_scalar(form(self.x[1]*self.dx))/self.A

    def getXSStiffnessMatrix(self):
        
        # Construct Displacement Coefficient mixed function space
        self.Ve = VectorElement("CG",self.msh.ufl_cell(),1,dim=3)
        self.V = FunctionSpace(self.msh, MixedElement(4*[self.Ve]))
        
        #displacement and test functions
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        #displacement coefficient trial functions
        self.ubar,self.uhat,self.utilde,self.ubreve=split(self.u)

        #displacement coefficient test functions
        self.vbar,self.vhat,self.vtilde,self.vbreve=split(self.v)
        
        #construct residual by looping through materials
        print('Constructing Residual...')
        self.Residual = 0
        for mat_id in self.mat_ids:
            self.Residual += self.constructMatResidual(mat_id)

        print('extracting nullspace....')
        self.getModes()
        print('decoupling modes...')
        self.decoupleModes()
        print('computing stiffness matrix')
        self.computeXSStiffnessMat()
        # PETSc.garbage_cleanup()
   
    def applyRotation(self,C,alpha,beta,gamma):
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        p,q,r,s=self.p,self.q,self.r,self.s
        #rotation about X-axis
        Rx = as_matrix([[1,         0,         0],
                        [0,cos(alpha),-sin(alpha)],
                        [0,sin(alpha),cos(alpha)]])
        #rotation about Y-axis
        Ry = as_matrix([[cos(beta), 0,sin(beta)],
                        [0,         1,        0],
                        [-sin(beta),0,cos(beta)]])
        #rotation about Z-axis
        Rz = as_matrix([[cos(gamma),-sin(gamma),0],
                        [sin(gamma),cos(gamma), 0],
                        [0,         0,          1]])
        
        #3D rotation matrix
        R = Rz*Ry*Rx

        Cprime = as_tensor(R[p,i]*R[q,j]*C[i,j,k,l]*R.T[k,r]*R.T[l,s],(p,q,r,s))

        return Cprime
    
    def constructMatOrientation(self,orientation):
        #orientation is a 
        self.Q = VectorFunctionSpace(self.msh,("DG",0),dim=3)
        self.theta = Function(self.Q)

        self.theta.interpolate(orientation)

    def constructMatResidual(self,mat_id):
        #geometric dimension
        d = self.d
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
        #trial and test functions
        ubar,uhat,utilde,ubreve=self.ubar,self.uhat,self.utilde,self.ubreve
        vbar,vhat,vtilde,vbreve=self.vbar,self.vhat,self.vtilde,self.vbreve
        #restricted integration domain
        # dx = self.dx(mat_id[1])
        dx = self.dx      
        
        C_mat = getMatConstitutive(self.msh,self.material[mat_id[0]])

        #if an orthotropic material is used, the constructMatOrientation method
        #must be called prior to applying rotations
        if self.material[mat_id[0]]['TYPE'] == 'ORTHOTROPIC':
            C = self.applyRotation(C_mat,self.theta[0],self.theta[1],self.theta[2])
        elif self.material[mat_id[0]]['TYPE'] == 'ISOTROPIC':
            self.C = C_mat
            C = self.C

        #sub-tensors of stiffness tensor
        Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
        Ci1kB = as_tensor([[[C[i, 0, k, l] for l in [1,2]]
                    for k in range(d)] for i in range(d)])
        Ciak1 = as_tensor([[[C[i, j, k, 0] for k in range(d)]
                    for j in [1,2]] for i in range(d)])
        CiakB = as_tensor([[[[C[i, j, k, l] for l in [1,2]]
                    for k in range(d)] for j in [1,2]] 
                    for i in range(d)])
        
        #partial derivatives of displacement:
        ubar_B = grad(ubar)
        uhat_B = grad(uhat)
        utilde_B = grad(utilde)
        ubreve_B = grad(ubreve)

        #partial derivatives of shape fxn:
        vbar_a = grad(vbar)
        vhat_a = grad(vhat)
        vtilde_a = grad(vtilde)
        vbreve_a = grad(vbreve)

        # equation 1,2,3
        L1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
            + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
            - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
            - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \

        # # equation 4,5,6
        L2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
            + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
            - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
            - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \

        # equation 7,8,9
        L3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
            - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
            - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\

        #equation 10,11,12
        L4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\
        
        return L1+L2+L3+L4

    def getModes(self):
        print('assembling stiffesss matrix....')   
        self.A_mat = assemble_matrix(form(self.Residual))
        self.A_mat.assemble()

        import time
        #====numpy svd approach=====#
        #this is clearly one of the worst options at large scales
        t0 = time.time()
        # m1,n1=self.A_mat.getSize()
        # Anp = self.A_mat.getValues(range(m1),range(n1))

        # Usvd,sv,Vsvd = np.linalg.svd(Anp)
        # self.sols = Vsvd[-12:,:].T
        t1= time.time()
        #===========

        #========numpy qr approach ======#
        #faster, but still bad
        # m,n1=self.A_mat.getSize()
        # Anp = self.A_mat.getValues(range(m),range(n1))
        # q,r = np.linalg.qr(Anp,mode='complete')
        # # q1,r1 = np.linalg.qr(Anp.T,mode='complete')
        t2 = time.time()
        #=============

        #========numpy qr approach ======#
        #gram schmidt variant, still dense and slow and bad
        # m,n1=self.A_mat.getSize()
        # Anp = self.A_mat.getValues(range(m),range(n1))
        # q1,r1 = qr_gs_modsr(Anp.T)
        t3 = time.time()
        #=============

        #========sparseqr approach ======#
        #very very fast, but memory intensive
        m,n1=self.A_mat.getSize()
        Anp = self.A_mat.getValues(range(m),range(n1))
        print('computing QR factorization')
        print(self.A_mat.size)
        Acsr = csr_matrix(self.A_mat.getValuesCSR()[::-1], shape=self.A_mat.size)
        q1coo,r1coo,E,rank = sparseqr.qr(Acsr.transpose(),economy=True)
        X0 = q1coo.tocsc()[:,-12:]

        t4 = time.time()
        #=============

        #==========
        #scipy sparse svd, slow, but seems to work okay
        #still

        # Acsr = csr_matrix(self.A_mat.getValuesCSR()[::-1], shape=self.A_mat.size)
        # _,_,self.sparse_sols= svds(Acsr,k=12,which='SM',solver='propack', return_singular_vectors="vh")
        t5 = time.time()

        # self.A_mat.

        # #========LU+ suitesparse qr approach ======#
        # # m,n1=self.A_mat.getSize()
        # # Anp = self.A_mat.getValues(range(m),range(n1))
        # # q1,r1 = np.linalg.qr(Anp.T,mode='complete')
        # print(self.A_mat.size)
        # Acsr = csr_matrix(self.A_mat.getValuesCSR()[::-1], shape=self.A_mat.size)
        # from scipy.sparse.linalg import splu
        # from scipy.sparse import csc_matrix
        # B = splu(Acsr.transpose())
        # U = B.U
        # Pc = csc_matrix((np.ones(B.perm_c.shape[0]), (np.arange(B.perm_c.shape[0]), B.perm_c)))
        # q_lu_coo,r_lu_coo,E,rank = sparseqr.qr(U.transpose(),economy=True)
        # nullspace = Pc.dot(q_lu_coo.transpose().tocsc()[:,-12:])
        t6 = time.time()
        #=============
        #========scipy lu + sparse svd ======#
        # from scipy.sparse import csc_matrix
        # Acsc = Acsr.tocsc()
        # from scipy.sparse.linalg import splu
        # lu = splu(Acsc)
        # U = lu.U
        # _,_,self.sparse_sols= svds(U,k=12,which='SM',solver='propack', return_singular_vectors="vh")
        t7=time.time()
        
        #========scipy inverse iteration ======#
        # from scipy.sparse.linalg import splu
        # #'NATURAL' specifies partial pivoting
        # lu = splu(Acsr.tocsc(),permc_spec='NATURAL')
        # U = lu.U.tocsr()
        # L = lu.L.tocsr()
        
        # Xprev = np.ones((U.shape[0],12))
        # X = Xprev
        # # Xprev= X0/np.linalg.norm(X0,axis=0)
        # from scipy.sparse.linalg import spsolve_triangular,spsolve
        # from scipy.sparse.linalg import norm
        # from scipy.sparse import identity
        
        # # I = identity(U.shape[0])

        # tol = 1e-13
        # max_iter = 100

        # for _ in range(max_iter):
        #     # W = spsolve_triangular(U.T,Xprev/np.linalg.norm(Xprev,axis=0))
        #     # Y = spsolve_triangular(U,W,lower=False)
        #     # Ay = Acsr.dot(Y)
        #     # X,_ = np.linalg.qr(Y)
            
        #     # AX=Acsr.dot(X)
        #     # res = np.linalg.norm(X-Xprev)
        #     # res2 = np.linalg.norm(AX)
        #     # res3 = np.linalg.norm(AX-Xprev)

        #     # Xprev = X
        #     #==========
        #     # Y = spsolve_triangular(L,Xprev/np.linalg.norm(Xprev,axis=0))
        #     W = spsolve_triangular(L,Xprev/np.linalg.norm(Xprev,axis=0))
        #     # Y,_ = np.linalg.qr(Y)
        #     Y = spsolve_triangular(U,W,lower=False)

        #     # residual = np.linalg.norm(Acsr.dot(X/np.linalg.norm(X)))
        #     # X = X/np.linalg.norm(X,axis=0)
        #     AY = Acsr.dot(Y)
        #     X, _ = np.linalg.qr(X)
        #     AX = Acsr.dot(X)
        #     # Ax = Ax/np.linalg.norm(Ax,axis=0)
        #     res = np.linalg.norm(X-Xprev)
        #     res1 = np.linalg.norm(AY)
        #     res2 = np.linalg.norm(AX)
        #     res3 = np.linalg.norm(AX-Xprev)
        #     res4 = np.linalg.norm(AY-Xprev)
        #     res5 = np.linalg.norm(Y-Xprev)

        #     Xprev = X
        #     #=======

        #     # X = orthogonalize(X)
        #     # X = orthogonalize(X.T).T
        #     # residual = np.linalg.norm(Acsr.dot(X))
        #     # X = X/np.linalg.norm(X,axis=0)
        #     # residual = np.linalg.norm(Acsr.dot(X))
        #     # Xcsr = csr_matrix(X)
        #     # XXT = Xcsr.dot(Xcsr.T)
        #     # residual = np.linalg.norm((I-XXT).dot(Y))

        #     if res < tol:
        #         break
        self.sparse_sols = q1coo.tocsc()[:,-12:]
        self.sols = self.sparse_sols.A
        # Pr = csc_matrix((np.ones(U.shape[0]), (lu.perm_r, np.arange(U.shape[0]))))
        # Pc = csc_matrix((np.ones(U.shape[0]), (np.arange(U.shape[0]), lu.perm_c)))
        # X2 = Pr.T.dot(X)
        # X3 = Pr.dot(X)
        # X0 = q1coo.tocsc()[:,-12:]
        # X0new, _ = np.linalg.qr(X0.A)
        # self.sols = X
        # self.sparse_sols = csr_matrix(self.sols)
        t8=time.time()

        
        if True:
            print('numpy svd time:')
            print(t1-t0)
            print('numpy qr time:')
            print(t2-t1)
            print('numpy gs_modsr qr time:')
            print(t3-t2)
            print('sparse qr time:')
            print(t4-t3)
            print('scipy svds time:')
            print(t5-t4)
            print(t6-t5)
            print('scipy lu+svds time:')
            print(t7-t6)
            print('scipy inverse iteration time:')
            print(t8-t7)


        # A_mat_csr = self.A_mat.getValuesCSR()
        # print(A_mat_csr[0].shape)
        # print(A_mat_csr[1].shape)
        # print(A_mat_csr[2].shape)

        #convert to dense (want to avoid)
        # print('making nullspace from null column vectors....')
        t7=time.time()
        # self.sparse_sols = hstack([q1coo.getcol(i) for i in range(-12,0)])
        cols = np.arange(-12,0)
        # self.sparse_sols = q1coo.tocsc()[:,cols]
        t8=time.time()
        # self.sols = self.sparse_sols.toarray()
        t9=time.time()
        print("extraction of sparse cols:")
        print(t8-t7)
        print("saving as dense:")
        print(t9-t8)
        # self.sols = q1coo[:,-12:]

        # self.sols = nullspace.toarray()
        # self.sols = q_lu_coo.toarray()[:,-12:]
        # print(q1coo.shape)
        # self.sols_sparse = q1coo.tocsc()[:,-12:]

        #======slepc iterative approach (conv issues)=======#
        # import slepc4py,sys
        # slepc4py.init(sys.argv)
        # Prob = slepc4py.SLEPc.SVD().create()
        # Prob.setOperators(self.A_mat)
        # Prob.setType("trlanczos")
        # Prob.setDimensions(12,PETSc.DECIDE,PETSc.DECIDE)
        # Prob.setWhichSingularTriplets(Prob.Which.SMALLEST)
        # Prob.setTolerances(1e-8, 1000)
        # Prob.setImplicitTranspose(True)
        # Prob.solve()
        #===========

        # A_mat_csr = self.A_mat.getValuesCSR()
        print()
        
    def decoupleModes(self):
        x = self.x
        dx = self.dx
        C = self.C
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B

        #get maps of vertices to displacement coefficients DOFs 
        self.ubar_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(0))
        self.uhat_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(1))
        self.utilde_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(2))
        self.ubreve_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(3))
        
        ubar_vtx_to_dof = self.ubar_vtx_to_dof
        uhat_vtx_to_dof = self.uhat_vtx_to_dof

        #GET UBAR AND UHAT RELATED MODES FROM SVD
        ubar_modes = self.sols[ubar_vtx_to_dof,:]
        uhat_modes = self.sols[uhat_vtx_to_dof,:]

        #CONSTRUCT FUNCTION FOR UBAR AND UHAT SOLUTIONS GIVEN EACH MODE
        UBAR = self.V.sub(0).collapse()[0]
        UHAT = self.V.sub(1).collapse()[0]
        ubar_mode = Function(UBAR)
        uhat_mode = Function(UHAT)

        #area and center of xs
        A = self.A
        yavg = self.yavg
        zavg = self.zavg

        #INITIALIZE DECOUPLING MATRIX (12X12)
        mat = np.zeros((12,12))
        
        #LOOP THROUGH MAT'S COLUMN (EACH MODE IS A COLUMN OF MAT):
        for mode in range(mat.shape[1]):
            #construct function from mode
            ubar_mode.vector.array = ubar_modes[:,:,mode].flatten()
            uhat_mode.vector.array = uhat_modes[:,:,mode].flatten()

            #FIRST THREE ROWS : AVERAGE UBAR_i VALUE FOR THAT MODE
            mat[0,mode]=assemble_scalar(form(ubar_mode[0]*dx))/A
            mat[1,mode]=assemble_scalar(form(ubar_mode[1]*dx))/A
            mat[2,mode]=assemble_scalar(form(ubar_mode[2]*dx))/A
            
            #SECOND THREE ROWS : AVERAGE ROTATION (COMPUTED USING UBAR x Xi, WHERE X1=0, X2,XY=Y,Z)
            # mat[3,mode]=assemble_scalar(form(((ubar_mode[2]*(x[0]-yavg)-ubar_mode[1]*(x[1]-zavg))*dx)))
            # mat[4,mode]=assemble_scalar(form(((ubar_mode[0]*(x[1]-zavg))*dx)))
            # mat[5,mode]=assemble_scalar(form(((-ubar_mode[0]*(x[0]-yavg))*dx)))

            mat[3,mode]=assemble_scalar(form(((ubar_mode[2]*(x[0])-ubar_mode[1]*(x[1]))*dx)))
            mat[4,mode]=assemble_scalar(form(((ubar_mode[0]*(x[1]))*dx)))
            mat[5,mode]=assemble_scalar(form(((-ubar_mode[0]*(x[0]))*dx)))
            # CONSTRUCT STRESSES FOR LAST SIX ROWS

            #compute strains at x1=0
            gradubar=grad(ubar_mode)
            eps = as_tensor([[uhat_mode[0],uhat_mode[1],uhat_mode[2]],
                            [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
                            [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
            
            # construct strain and stress tensors based on u_sol
            sigma = as_tensor(C[i,j,k,l]*eps[k,l],(i,j))

            #relevant components of stress tensor
            sigma11 = sigma[0,0]
            sigma12 = sigma[0,1]
            sigma13 = sigma[0,2]

            #integrate stresses over cross-section at "root" of beam and construct xs load vector
            P1 = assemble_scalar(form(sigma11*dx))
            V2 = assemble_scalar(form(sigma12*dx))
            V3 = assemble_scalar(form(sigma13*dx))
            # T1 = assemble_scalar(form(((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
            # M2 = assemble_scalar(form((x[1]-zavg)*sigma11*dx))
            # M3 = assemble_scalar(form(-(x[0]-yavg)*sigma11*dx))
            T1 = assemble_scalar(form(((x[0])*sigma13 - (x[1])*sigma12)*dx))
            M2 = assemble_scalar(form((x[1])*sigma11*dx))
            M3 = assemble_scalar(form(-(x[0])*sigma11*dx))
            #THIRD THREE ROWS: AVERAGE FORCE (COMPUTED WITH UBAR AND UHAT)
            mat[6,mode]=P1
            mat[7,mode]=V2
            mat[8,mode]=V3   

            #FOURTH THREE ROWS: AVERAGE MOMENTS (COMPUTED WITH UBAR AND UHAT)
            mat[9,mode]=T1
            mat[10,mode]=M2
            mat[11,mode]=M3
        
        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        ubar_mode.vector.destroy()  #need to add to prevent PETSc memory leak 
        uhat_mode.vector.destroy()  #need to add to prevent PETSc memory leak 

        self.sols_decoup = (self.sparse_sols.dot(inv(csr_matrix(mat)))).toarray()
        # self.sols_decoup = self.sols@np.linalg.inv(mat)
        print()

    def computeXSStiffnessMat(self):
        x = self.x
        dx = self.dx
        yavg = self.yavg
        zavg = self.zavg
        C = self.C
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
        #vtx to dof maps
        ubar_vtx_to_dof = self.ubar_vtx_to_dof
        uhat_vtx_to_dof = self.uhat_vtx_to_dof

        #construct coefficient fields over 2D mesh
        U2d = FunctionSpace(self.msh,self.Ve)
        ubar_field = Function(U2d)
        uhat_field = Function(U2d)
        # utilde_field = Function(U2d)
        # ubreve_field = Function(U2d)

        #==================================================#
        #======== LOOP FOR BUILDING LOAD MATRICES =========#
        #==================================================#

        Cstack = np.vstack((np.zeros((6,6)),np.eye(6)))

        from itertools import combinations
        idx_ops = [0,1,2,3,4,5]
        comb_list = list(combinations(idx_ops,2))
        Ccomb = np.zeros((6,len(comb_list)))
        for idx,ind in enumerate(comb_list):
            np.put(Ccomb[:,idx],ind,1)
        Ccomb = np.vstack((np.zeros_like(Ccomb),Ccomb))

        Ctotal = np.hstack((Cstack,Ccomb))

        K1 = np.zeros((6,6))
        K2 = np.zeros((6,6))

        #START LOOP HERE and loop through unit vectors for K1 and diagonal entries of K2
        # then combinations of ci=1 where there is more than one nonzero entry
        for idx,c in enumerate(Ctotal.T):
            # use right singular vectors (nullspace) to find corresponding
            # coefficient solutions given arbitrary coefficient vector c
            c = np.reshape(c,(-1,1))
            u_coeff=self.sols_decoup@c

            #populate displacement coeff fxns based on coeffcient values
            self.coeff_to_field(ubar_field,u_coeff,self.ubar_vtx_to_dof)
            self.coeff_to_field(uhat_field,u_coeff,self.uhat_vtx_to_dof)

            # #use previous dof maps to populate arrays of the individual dofs that depend on the solution coefficients
            # ubar_coeff = u_coeff.flatten()[ubar_vtx_to_dof]
            # uhat_coeff = u_coeff.flatten()[uhat_vtx_to_dof]

            # #populate functions with coefficient function values
            # ubar_field.vector.array = ubar_coeff.flatten()
            # uhat_field.vector.array = uhat_coeff.flatten()

            #map displacement to stress
            sigma = self.map_disp_to_stress(ubar_field,uhat_field)
            
            # #compute strains at x1=0
            # gradubar=grad(ubar_field)
            # eps = as_tensor([[uhat_field[0],uhat_field[1],uhat_field[2]],
            #                 [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
            #                 [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
            
            # # construct strain and stress tensors based on u_sol
            # sigma = as_tensor(C[i,j,k,l]*eps[k,l],(i,j))

            #relevant components of stress tensor
            sigma11 = sigma[0,0]
            sigma12 = sigma[0,1]
            sigma13 = sigma[0,2]

            #integrate stresses over cross-section at "root" of beam and construct xs load vector
            P1 = assemble_scalar(form(sigma11*dx))
            V2 = assemble_scalar(form(sigma12*dx))
            V3 = assemble_scalar(form(sigma13*dx))
            # T1 = assemble_scalar(form(((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
            # M2 = assemble_scalar(form((x[1]-zavg)*sigma11*dx))
            # M3 = assemble_scalar(form(-(x[0]-yavg)*sigma11*dx))
            T1 = assemble_scalar(form(((x[0])*sigma13 - (x[1])*sigma12)*dx))
            M2 = assemble_scalar(form((x[1])*sigma11*dx))
            M3 = assemble_scalar(form(-(x[0])*sigma11*dx))


            #assemble loads into load vector
            P = np.array([P1,V2,V3,T1,M2,M3])

            eps = self.get_eps_from_disp_fxns(ubar_field,uhat_field)

            #compute complementary energy given coefficient vector
            Uc = assemble_scalar(form(sigma[i,j]*eps[i,j]*dx))
            
            if idx<=5:
                K1[:,idx]= P
                K2[idx,idx] = Uc
            else:
                idx1 = comb_list[idx-6][0]
                idx2 = comb_list[idx-6][1]
                Kxx = 0.5 * ( Uc - K2[idx1,idx1] - K2[idx2,idx2])
                K2[idx1,idx2] = Kxx
                K2[idx2,idx1] = Kxx



        #see https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        ubar_field.vector.destroy()     #need to add to prevent PETSc memory leak   
        uhat_field.vector.destroy()     #need to add to prevent PETSc memory leak 

        #compute Flexibility matrix
        self.K1_inv = np.linalg.inv(K1)

        self.S = self.K1_inv.T@K2@self.K1_inv
        self.K = np.linalg.inv(self.S)
    
    def getXSMassMatrix(self):
        return
    
    def coeff_to_field(self,fxn,coeff,vtx_to_dof):
        #uses vtx_to_dof map to populate field with correct solution coefficients
        fxn.vector.array = coeff.flatten()[vtx_to_dof].flatten()
        fxn.vector.destroy

    def map_disp_to_stress(self,ubar,uhat):
        i,j,k,l=self.i,self.j,self.k,self.l

        #compute strains at x1=0
        eps = self.get_eps_from_disp_fxns(ubar,uhat)        
        
        # construct strain and stress tensors based on u_sol
        sigma = as_tensor(self.C[i,j,k,l]*eps[k,l],(i,j))
        return sigma
    
    def get_eps_from_disp_fxns(self,ubar,uhat):
        gradubar=grad(ubar)
        return as_tensor([[uhat[0],uhat[1],uhat[2]],
                        [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
                        [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
    
    def recover_stress_xs(self,loads):
        '''
        loads: 6x1 vector of the loads over the xs
        '''
        # print('loads:')
        # print(loads)
        # print('K1 inverse:')
        # print(self.K1_inv)
        # print('sols_decoup:')
        # print(self.sols_decoup)
        disp_coeff = self.sols_decoup[:,6:]@self.K1_inv@loads
        # print(disp_coeff)

        U2d = FunctionSpace(self.msh,self.Ve)
        ubar = Function(U2d)
        uhat = Function(U2d)
        utilde = Function(U2d)
        ubreve = Function(U2d)

        #populate displacement coeff fxns based on coeffcient values
        self.coeff_to_field(ubar,disp_coeff,self.ubar_vtx_to_dof)
        self.coeff_to_field(uhat,disp_coeff,self.uhat_vtx_to_dof)
        self.coeff_to_field(utilde,disp_coeff,self.utilde_vtx_to_dof)
        self.coeff_to_field(ubreve,disp_coeff,self.ubreve_vtx_to_dof)

        # u_total = ubar

    
        sigma = self.map_disp_to_stress(ubar,uhat)

        S = TensorFunctionSpace(self.msh,('CG',1),shape=(3,3))
        sigma_sol = Function(S)
        sigma_sol.interpolate(Expression(sigma,S.element.interpolation_points()))
        points_on_proc,cells=get_pts_and_cells(self.msh,self.msh.geometry.x)

        sigma_sol_eval = sigma_sol.eval(points_on_proc,cells)
        print("Sigma Sol:")
        print(sigma_sol_eval)
        L = 0
        utotal_exp = ubar+uhat*L+utilde*(L**2)+ubreve*(L**3)
        utotal = Function(U2d)
        utotal.interpolate(Expression(utotal_exp,U2d.element.interpolation_points()))
        # points_on_proc,cells=get_pts_and_cells(self.msh,self.msh.geometry.x)

        # sigma_sol_eval = utotal.eval(points_on_proc,cells)        
        #TODO: now plot over xs
        warp_factor = 1

        #plot Axial mesh
        tdim = self.msh.topology.dim
        topology, cell_types, geom = create_vtk_mesh(self.msh,tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        plotter = pyvista.Plotter()
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
        RBG = np.array([[ 0,  0,  1],
                             [ 1,  0,  0],
                             [ 0,  1, 0]])
        print(geom.shape[0])
        # grid.point_data['u'] = (utotal.x.array.reshape((geom.shape[0],3)).T).T
        # warped = grid.warp_by_vector("u",factor=warp_factor)
        # actor_1 = plotter.add_mesh(warped, show_edges=True)
        grid.point_data['sigma11'] = sigma_sol_eval[:,0]
        warped = grid.warp_by_scalar("sigma11",factor=0.000001)
        actor_2 = plotter.add_mesh(warped,show_edges=True)

        # actor_1 = plotter.add_mesh(grid, style='points',color='k',point_size=12)
        # grid.point_data["u"]= self.o.x.array.reshape((geom.shape[0],3))
        # glyphs = grid.glyph(orient="u",factor=.25)
        # actor_2 = plotter.add_mesh(glyphs,color='b')


        plotter.view_xy()
        plotter.show_axes()

        # if not pyvista.OFF_SCREEN:
        plotter.show()

#from https://colab.research.google.com/github/QuantEcon/lecture-python.notebooks/blob/master/qr_decomp.ipynb
# import numpy.linalg as lin
# def qr_gs_modsr(A, type=complex):
    
#     A = np.array(A, dtype=type)
    
#     (m,n) = np.shape(A) # Get matrix A's shape m - # of rows, m - # of columns
   
#     # Q - an orthogonal matrix of m-column vectors
#     # R - an upper triangular matrix (the Gaussian elimination of A to the row-echelon form)
    
#     # Initialization: [ Q - multivector Q = A of shape (m x n) ]
#     #                 [ R - multivector of shape (n x n)       ]

#     Q = np.array(A, dtype=type)      # Q - matrix A
#     R = np.zeros((n, n), dtype=type) # R - matrix of 0's    

#     # **** Objective: ****

#     # For each column vector r[k] in R:
#        # Compute r[k,i] element in R, k-th column q[k] in Q;

#     for k in range(n):
#         # For a span of the previous column vectors q[0..k] in Q, 
#         # compute the R[i,k] element in R as the inner product of vectors q[i] and q[k],
#         # compute k-th column vector q[k] as the product of scalar R[i,k] and i-th vector q[i],
#         # subtracting it from the k-th column vector q[k] in Q
#         for i in range(k):

#             # **** Compute k-th column q[k] of Q and k-th row r[k] of R **** 
#             R[i,k] = np.transpose(Q[:,i]).dot(Q[:,k])
#             Q[:,k] = Q[:,k] - R[i,k] * Q[:,i]
            
#         # Compute the r[k,k] pseudo-diagonal element in R 
#         # as the Euclidean norm of the k-th vector q[k] in Q,

#         # Normalize the k-th vector q[k] in Q, dividing it by the norm r[k,k]
#         R[k,k] = lin.norm(Q[:,k]); Q[:,k] = Q[:,k] / R[k,k]
    
#     return -Q, -R  # Return the resultant negative matrices Q and R

class CrossSectionAnalytical:
    def __init__(self,params):
        #process params based on shape

        if params['shape']=='rectangle':
            self.h = params['h']
            self.w = params['w']
            self.E = params['E']
            self.nu = params['nu']

        elif params['shape'] =='box':
            self.h = params['h']
            self.w = params['w']
            self.t_h = params['t_h']
            self.t_w = params['t_w']
            self.E = params['E']
            self.nu = params['nu']

        else:
            print("busy doing nothing...")
        
        
    def compute_stiffness(self):
        A = (self.h*self.w)-((self.h-2*self.t_h)*(self.w-self.t_w))
        G = self.E / (2*(1+self.nu))
        if self.h>=self.w:
            J = (self.h * self.w ** 3) * (2 / 9) * (1 / (1 + (self.w / self.h) ** 2))
        else:
            J = (self.w * self.h ** 3) * (2 / 9) * (1 / (1 + (self.h / self.w) ** 2))

        kappa = 5/6
        
        EA = self.E*A
        kGA1=kappa*G*A
        kGA2=kappa*G*A
        GJ = G*J
        EI1 = self.E*(self.w*self.h**3 /12 - ((self.w-2*self.t_w)*(self.h-2*self.t_h)**3) /12)
        EI2 = self.E*(self.h*self.w**3 /12 - ((self.h-2*self.t_h)*(self.w-2*self.t_w)**3) /12)
        
        self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2,]))

def orthogonalize(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    ```
    """
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= np.linalg.norm(V[i])
    return V.T