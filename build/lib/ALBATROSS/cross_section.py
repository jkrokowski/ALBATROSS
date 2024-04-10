from ufl import (dot,as_vector,variable,cross,diff,grad,sin,cos,as_matrix,SpatialCoordinate,FacetNormal,Measure,as_tensor,indices,VectorElement,MixedElement,TrialFunction,TestFunction,split,TensorElement)
# from ufl import Constant as uflConstant
from dolfinx.fem import (Constant,Expression,TensorFunctionSpace,assemble_scalar,form,Function,FunctionSpace,VectorFunctionSpace)
from dolfinx.fem.petsc import assemble_matrix
# from dolfinx.fem import Constant as femConstant
from dolfinx.mesh import meshtags
import numpy as np
from petsc4py import PETSc

default_scalar_type = PETSc.ScalarType    

from scipy.sparse.linalg import inv
import sparseqr
from scipy.sparse import csr_matrix,csc_matrix

from dolfinx.plot import create_vtk_mesh
import pyvista

from ALBATROSS.material import getMatConstitutive,Material,getMatConstitutiveIsotropic
from ALBATROSS.utils import plot_xdmf_mesh,get_vtx_to_dofs,get_pts_and_cells,sparseify

#TODO: allow user to specify a point to find xs props about
#TODO: provide a method to translate between different xs values?

class CrossSection:
    def __init__(self, msh, materials ,celltags=None):
        #analysis domain
        self.msh = msh
        '''
        TODO: example for four elements (include type/size checks)
        assert(len(celltags['mat_id'])==len(msh))
        assert(len(celltags['orientation'])==len(msh))
        units for orientation?
        celltags = {'mat_id': [0, 0, 1,0], 'orientation': {0, 90, 90, 0}}
        '''
        self.ct = celltags
        #list of material objects 
        self.materials = materials
        
        #geometric dimension
        self.d = 3
        self.tdim = 2

        #number of materials
        self.num_mat = len(self.materials)
        #tuple of material names and ids
        # self.mat_ids = list(zip(list(self.material.keys()),list(range(self.num_mat))))
        # mat_names = [self.materials[i].name for i in range(self.num_mat)]
          
        
        # print("mat ids:")
        # print(self.mat_ids)

        #indices
        self.i,self.j,self.k,self.l=indices(4)
        # self.p,self.q,self.r,self.s=indices(4)
        self.a,self.B = indices(2)
        
        #integration measures (subdomain data accounts for different materials)
        if self.ct!=None:
            #check that the number and values of celltags match those specified in the material objects
            mesh_ct = np.unique(self.ct.values)
            mat_ct = np.unique([self.materials[_i].id for _i in range(self.num_mat)] )
            assert(np.logical_and.reduce(mesh_ct==mat_ct))
            
            #material property functions
            self.Q = FunctionSpace(self.msh,('DG',0))
            # self.C = TensorFunctionSpace(self.msh,('DG',0),shape=(3,3,3,3))
            self.E = Function(self.Q)
            self.nu = Function(self.Q)
            for material in self.materials:
                if material.type == "ISOTROPIC":
                    cells = self.ct.find(material.id)
                    self.E.x.array[cells] = np.full_like(cells,material.E,dtype=default_scalar_type)
                    self.nu.x.array[cells] = np.full_like(cells,material.nu,dtype=default_scalar_type)
                elif material.type == "ORTHOROPIC":
                    print("forthcoming")
                else:
                    print("unsupported material type")

            self.C = getMatConstitutiveIsotropic(self.msh,self.E,self.nu)
            
            #construct measure for subdomains using celltag info
            # self.dx = Measure("dx",domain=self.msh,subdomain_data=self.ct)
            self.dx = Measure("dx",domain=self.msh,subdomain_data=self.ct)

        elif self.ct == None:
            self.dx = Measure("dx",domain=self.msh)
        #     for material in self.materials:
        #         print(self.ct.find(material.id))
        #         material_facets=meshtags(self.msh,self.tdim,self.ct.find(material.id),material.id)
        #         material.dx = Measure("dx",domain=self.msh,subdomain_data=material_facets)
            self.C = getMatConstitutiveIsotropic(self.msh,self.materials[0].E,self.materials[0].nu)
        self.ds = Measure("ds",domain=self.msh)
        
        #spatial coordinate and facet normals
        self.x = SpatialCoordinate(self.msh)
        self.n = FacetNormal(self.msh)
        
        #compute area over the cross-section
        self.A = assemble_scalar(form(1.0*self.dx))
        if self.num_mat >1:
            for material in self.materials:
                material.A = assemble_scalar(form(1.0*self.dx(material.id)))
        #TODO: compute density weight areas and areas of each subdomain?
        
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

        #partial derivatives of displacement:
        self.ubar_B = grad(self.ubar)
        self.uhat_B = grad(self.uhat)
        self.utilde_B = grad(self.utilde)
        self.ubreve_B = grad(self.ubreve)

        #partial derivatives of shape fxn:
        self.vbar_a = grad(self.vbar)
        self.vhat_a = grad(self.vhat)
        self.vtilde_a = grad(self.vtilde)
        self.vbreve_a = grad(self.vbreve)
        
        #construct material constitutive tensor field
        # self.constructConstitutiveField()

        #assemble matrix
        print('Constructing Residual...')
        self.constructResidual()
        # self.Residual += self.constructMatResidual(self.materials[0])

        print('Computing non-trivial solutions....')
        self.getModes()
        print('Orthogonalizing w.r.t. elastic modes...')
        self.decoupleModes()
        print('Computing Beam Constitutive Matrix....')
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
        #orientation is a list of angles
        self.Q = VectorFunctionSpace(self.msh,("DG",0),dim=3)
        self.theta = Function(self.Q)

        self.theta.interpolate(orientation)

    def constructResidual(self):
        #geometric dimension
        d = self.d
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
        #trial and test functions
        ubar,uhat,utilde,ubreve=self.ubar,self.uhat,self.utilde,self.ubreve
        vbar,vhat,vtilde,vbreve=self.vbar,self.vhat,self.vtilde,self.vbreve
        #partial derivatives of trial and test functions
        ubar_B,uhat_B,utilde_B,ubreve_B=self.ubar_B,self.uhat_B,self.utilde_B,self.ubreve_B
        vbar_a,vhat_a,vtilde_a,vbreve_a=self.vbar_a,self.vhat_a,self.vtilde_a,self.vbreve_a

        C = self.C
        # #restricted integration domain
        # if self.ct == None:
        #     dx = self.dx
        # else:
        #     print("material id:")
        #     # print(material.id)
        #     # subdomain_indices = self.ct.find(material.id)
        #     # print(subdomain_indices)
        #     # subdomain_values  = np.full_like(subdomain_indices, material.id, dtype=np.int32)
        #     # print(subdomain_values)
        #     # subdomain = meshtags(self.msh, 2, subdomain_indices, subdomain_values)
        #     # dx = Measure('dx', domain=self.msh, subdomain_data=subdomain,subdomain_id=material.id)
        #     dx = self.dx
        #     # dx = self.dx
        dx = self.dx      
        

        #if an orthotropic material is used, the constructMatOrientation method
        #must be called prior to applying rotations
        # if material.type == 'ORTHOTROPIC':
        #     #TODO: need to think about how to store these tensors? 
        #     #We can't store potentially thousands of these, so we need to store the constitutive tensor (per material)
        #     # the rotation angles for each element associated with each cell as a DG0 fxn? 
        #     C = self.applyRotation(material.C,self.theta[0],self.theta[1],self.theta[2])
        # elif material.type == 'ISOTROPIC':
        #     C = material.C
            # C = getMatConstitutive(self.msh,material)

        #sub-tensors of stiffness tensor
        Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
        Ci1kB = as_tensor([[[C[i_, 0, k_, l_] for l_ in [1,2]]
                    for k_ in range(d)] for i_ in range(d)])
        Ciak1 = as_tensor([[[C[i_, j_, k_, 0] for k_ in range(d)]
                    for j_ in [1,2]] for i_ in range(d)])
        CiakB = as_tensor([[[[C[i_, j_, k_, l_] for l_ in [1,2]]
                    for k_ in range(d)] for j_ in [1,2]] 
                    for i_ in range(d)])
        
        # print(Ci1k1.ufl_shape)
        # print(Ci1kB.ufl_shape)
        # print(Ciak1.ufl_shape)
        # print(CiakB.ufl_shape)
        # print("--------------")
        # print(ubar.ufl_shape)
        # print(uhat.ufl_shape)
        # print(utilde.ufl_shape)
        # print(ubreve.ufl_shape)
        # print("--------------")
        # print(ubar_B.ufl_shape)
        # print(uhat_B.ufl_shape)
        # print(utilde_B.ufl_shape)
        # print(ubreve_B.ufl_shape)

        # #partial derivatives of displacement:
        # ubar_B = grad(ubar)
        # uhat_B = grad(uhat)
        # utilde_B = grad(utilde)
        # ubreve_B = grad(ubreve)

        # #partial derivatives of shape fxn:
        # vbar_a = grad(vbar)
        # vhat_a = grad(vhat)
        # vtilde_a = grad(vtilde)
        # vbreve_a = grad(vbreve)

        # material.simpleL1=form(2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx)
        
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
        
        self.Residual =  L1+L2+L3+L4
        # return L4

    def getModes(self):
        print('Assembling System Matrix....')   
        self.A_mat = assemble_matrix(form(self.Residual))
        self.A_mat.assemble()

        m,n1=self.A_mat.getSize()
        # Anp = self.A_mat.getValues(range(m),range(n1))
        print('Computing QR factorization')
        print(self.A_mat.size)
        Acsr = csr_matrix(self.A_mat.getValuesCSR()[::-1], shape=self.A_mat.size)
        import time
        t0= time.time()
        #perform QR factorization and store as struct in householder form
        QR= sparseqr.qr_factorize( Acsr.transpose() )
        #build matrix of unit vectors for selecting last 12 columns
        X = np.zeros((m,12))
        for i in range(12):
            X[m-1-i,11-i]=1
        #perform matrix multiplication implicitly to construct orthogonal nullspace basis
        self.sols = sparseqr.qmult(QR,X)
        # print(self.sols)
        self.sparse_sols = sparseify(self.sols,sparse_format='csc')
        print("QR factorization time = %f" % (time.time()-t0))

    def decoupleModes(self):
        # We need to handle this operation on a per material basis, so we can't just store C one time

        #this is a change of basis operation from the originally computed basis to one
        # using our knowledge of what the solution should look like
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
        
        mat_sparse = sparseify(mat,sparse_format='csc')

        self.sols_decoup = (self.sparse_sols.dot(inv(mat_sparse))).toarray()
        # self.sols_decoup = self.sols@np.linalg.inv(mat)

    def computeXSStiffnessMat(self):       
        #unpacking values
        x = self.x
        dx = self.dx
        yavg = self.yavg
        zavg = self.zavg
        C = self.C
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B

        #initialize matrices
        K1 = np.zeros((6,6))
        K2 = np.zeros((6,6))

        #should we use a mixed element tensor function space for each of the warping functions?
        Ne = TensorElement('CG',self.msh.ufl_cell(),1,shape=(3,6))
        # N_space = TensorFunctionSpace(domain,('CG',1),shape=(12,6))
        N_space = FunctionSpace(self.msh,MixedElement(4*[Ne]))
        N = Function(N_space)

        N_bar, N_hat, N_tilde, N_breve = N.split() 
        N_bar_vtx_to_dofs = get_vtx_to_dofs(self.msh,N_space.sub(0))
        N_hat_vtx_to_dofs = get_vtx_to_dofs(self.msh,N_space.sub(1))
        N_tilde_vtx_to_dofs = get_vtx_to_dofs(self.msh,N_space.sub(2))
        N_breve_vtx_to_dofs = get_vtx_to_dofs(self.msh,N_space.sub(3))

        elastic_sols = self.sols_decoup[:,6:]
        N_bar_vals = elastic_sols[self.ubar_vtx_to_dof.flatten(),:]
        N_hat_vals = elastic_sols[self.uhat_vtx_to_dof.flatten(),:]
        N_tilde_vals = elastic_sols[self.utilde_vtx_to_dof.flatten(),:]
        N_breve_vals = elastic_sols[self.ubreve_vtx_to_dof.flatten(),:]

        N_bar.vector.array[N_bar_vtx_to_dofs.flatten()] = N_bar_vals.flatten()
        N_hat.vector.array[N_hat_vtx_to_dofs.flatten()] = N_hat_vals.flatten()
        N_tilde.vector.array[N_tilde_vtx_to_dofs.flatten()] = N_tilde_vals.flatten()
        N_breve.vector.array[N_breve_vtx_to_dofs.flatten()] = N_breve_vals.flatten()
        
        c7 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c8 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c9 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c10 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c11 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c12 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c = as_tensor([c7,c8,c9,c10,c11,c12])
        # c_var = variable(c)

        # x1 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))

        ubar_c = dot(N_bar,c)
        uhat_c = dot(N_hat,c)
        utilde_c = dot(N_tilde,c)
        ubreve_c = dot(N_breve,c)

        # u_c = ubar_c + uhat_c * x1 + utilde_c * x1**2 + ubreve_c * x1**3

        gradubar_c=grad(ubar_c)
        eps_c = as_tensor([[uhat_c[0],uhat_c[1],uhat_c[2]],
                            [gradubar_c[0,0],gradubar_c[1,0],gradubar_c[2,0]],
                            [gradubar_c[0,1],gradubar_c[1,1],gradubar_c[2,1]]])

        sigma_c = as_tensor(C[i,j,k,l]*eps_c[k,l],(i,j))

        sigma11_c = sigma_c[0,0]
        sigma12_c = sigma_c[0,1]
        sigma13_c = sigma_c[0,2]

        P1 = sigma11_c*dx
        V2 = sigma12_c*dx
        V3 = sigma13_c*dx
        T1 = -((x[0])*sigma13_c - (x[1])*sigma12_c)*dx
        M2 = -(x[1])*sigma11_c*dx
        M3 = (x[0])*sigma11_c*dx

        Uc = sigma_c[i,j]*eps_c[i,j]*dx

        P = [P1,V2,V3,T1,M2,M3]
        for idx1 in range(6):
            for idx2 in range(6):
                #build values of K1 matrix using derivatives of load vector w.r.t. elastic solution coefficients
                K1_xx_form = diff(P[idx1],c[idx2])
                K1[idx1,idx2] = assemble_scalar(form(K1_xx_form))
                
                #build values of K2 matrix using derivatives of internal energy form w.r.t. elastic solution coefficients
                K2_xx_form = diff(diff(Uc,c[idx1]),c[idx2])
                K2[idx1,idx2] = 0.5*assemble_scalar(form(K2_xx_form))

        #compute Flexibility matrix
        self.K1 = K1
        self.K2 = K2
        self.K1_inv = np.linalg.inv(K1)
        self.S = self.K1_inv.T@K2@self.K1_inv
        #invert to find stiffness matrix
        self.K = np.linalg.inv(self.S)
        

    def getXSMassMatrix(self):
        return
    
    def coeff_to_field(self,fxn,coeff,vtx_to_dof):
        #uses vtx_to_dof map to populate field with correct solution coefficients
        fxn.vector.array = coeff.flatten()[vtx_to_dof].flatten()
        fxn.vector.destroy

    def construct_warping_fxns(self,u_c,N):
        #utility to populate data from decoupled modes to warping displacement functions
        Uc = u_c.function_space #fxn space associated with fxn

        #loop through subspaces and map values from nullspace to specified fxn values
        vtx_to_dofs = []
        for i in range(Uc.num_sub_spaces):
            vtx_to_dofs.append(get_vtx_to_dofs(self.msh,Uc.sub(0)))
            vtx_to_dofs_flat = vtx_to_dofs[i].flatten()
            u_c.vector.array[vtx_to_dofs_flat] = N.flatten()[vtx_to_dofs_flat]

        # ubar_c,uhat_c,utilde_c,ubreve_c = split(u_c)
        # ubar_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(0))
        # uhat_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(1))
        # utilde_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(2))
        # ubreve_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(3))

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
    
    def recover_stress_xs(self,loads,plot_stress=True):
        '''
        loads: 6x1 vector of the loads over the xs
        '''

        disp_coeff = self.sols_decoup[:,6:]@self.K1_inv@loads

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

        sigma = self.map_disp_to_stress(ubar,uhat)

        S = TensorFunctionSpace(self.msh,('CG',1),shape=(3,3))
        sigma_sol = Function(S)
        sigma_sol.interpolate(Expression(sigma,S.element.interpolation_points()))
        points_on_proc,cells=get_pts_and_cells(self.msh,self.msh.geometry.x)

        sigma_sol_eval = sigma_sol.eval(points_on_proc,cells)
        # print("Sigma Sol:")
        # print(sigma_sol_eval)
        # L = 0
        # utotal_exp = ubar+uhat*L+utilde*(L**2)+ubreve*(L**3)
        # utotal = Function(U2d)
        # utotal.interpolate(Expression(utotal_exp,U2d.element.interpolation_points()))
        # points_on_proc,cells=get_pts_and_cells(self.msh,self.msh.geometry.x)

        # sigma_sol_eval = utotal.eval(points_on_proc,cells)        
        if plot_stress==True:
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
            # print(geom.shape[0])
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
    def plot_mesh(self):
        plot_xdmf_mesh(self.msh)
        # #plots mesh o
        # pyvista.global_theme.background = [255, 255, 255, 255]
        # pyvista.global_theme.font.color = 'black'
        # tdim = self.msh.topology.dim
        # topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
        # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
        # plotter = pyvista.Plotter()
        # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
        # plotter.view_isometric()
        # plotter.show_axes()
        # if not pyvista.OFF_SCREEN:
        #     plotter.show()

    def _old_compute_xs_stiffness():
        print()
                #OLD METHOD BETWEEN HERE:
        
        # x = self.x
        # dx = self.dx
        # yavg = self.yavg
        # zavg = self.zavg
        # C = self.C
        # #indices
        # i,j,k,l=self.i,self.j,self.k,self.l
        # a,B = self.a,self.B
        # #vtx to dof maps
        # ubar_vtx_to_dof = self.ubar_vtx_to_dof
        # uhat_vtx_to_dof = self.uhat_vtx_to_dof

        # #construct coefficient fields over 2D mesh
        # U2d = FunctionSpace(self.msh,self.Ve)
        # ubar_field = Function(U2d)
        # uhat_field = Function(U2d)
        # # utilde_field = Function(U2d)
        # # ubreve_field = Function(U2d)

        # #==================================================#
        # #======== LOOP FOR BUILDING LOAD MATRICES =========#
        # #==================================================#

        # Cstack = np.vstack((np.zeros((6,6)),np.eye(6)))

        # from itertools import combinations
        # idx_ops = [0,1,2,3,4,5]
        # comb_list = list(combinations(idx_ops,2))
        # Ccomb = np.zeros((6,len(comb_list)))
        # for idx,ind in enumerate(comb_list):
        #     np.put(Ccomb[:,idx],ind,1)
        # Ccomb = np.vstack((np.zeros_like(Ccomb),Ccomb))

        # Ctotal = np.hstack((Cstack,Ccomb))

        # K1 = np.zeros((6,6))
        # K2 = np.zeros((6,6))

        # #START LOOP HERE and loop through unit vectors for K1 and diagonal entries of K2
        # # then combinations of ci=1 where there is more than one nonzero entry
        # for idx,c in enumerate(Ctotal.T):
        #     # use right singular vectors (nullspace) to find corresponding
        #     # coefficient solutions given arbitrary coefficient vector c
        #     c = np.reshape(c,(-1,1))
        #     u_coeff=self.sols_decoup@c

        #     #populate displacement coeff fxns based on coeffcient values
        #     self.coeff_to_field(ubar_field,u_coeff,self.ubar_vtx_to_dof)
        #     self.coeff_to_field(uhat_field,u_coeff,self.uhat_vtx_to_dof)

        #     # #use previous dof maps to populate arrays of the individual dofs that depend on the solution coefficients
        #     # ubar_coeff = u_coeff.flatten()[ubar_vtx_to_dof]
        #     # uhat_coeff = u_coeff.flatten()[uhat_vtx_to_dof]

        #     # #populate functions with coefficient function values
        #     # ubar_field.vector.array = ubar_coeff.flatten()
        #     # uhat_field.vector.array = uhat_coeff.flatten()

        #     #map displacement to stress
        #     sigma = self.map_disp_to_stress(ubar_field,uhat_field)
            
        #     # #compute strains at x1=0
        #     # gradubar=grad(ubar_field)
        #     # eps = as_tensor([[uhat_field[0],uhat_field[1],uhat_field[2]],
        #     #                 [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
        #     #                 [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
            
        #     # # construct strain and stress tensors based on u_sol
        #     # sigma = as_tensor(C[i,j,k,l]*eps[k,l],(i,j))

        #     #relevant components of stress tensor
        #     sigma11 = sigma[0,0]
        #     sigma12 = sigma[0,1]
        #     sigma13 = sigma[0,2]

        #     #integrate stresses over cross-section at "root" of beam and construct xs load vector
        #     P1 = assemble_scalar(form(sigma11*dx))
        #     V2 = assemble_scalar(form(sigma12*dx))
        #     V3 = assemble_scalar(form(sigma13*dx))
        #     # T1 = assemble_scalar(form(((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
        #     # M2 = assemble_scalar(form((x[1]-zavg)*sigma11*dx))
        #     # M3 = assemble_scalar(form(-(x[0]-yavg)*sigma11*dx))
        #     T1 = assemble_scalar(form(((x[0])*sigma13 - (x[1])*sigma12)*dx))
        #     M2 = assemble_scalar(form((x[1])*sigma11*dx))
        #     M3 = assemble_scalar(form(-(x[0])*sigma11*dx))


        #     #assemble loads into load vector
        #     P = np.array([P1,V2,V3,T1,M2,M3])

        #     eps = self.get_eps_from_disp_fxns(ubar_field,uhat_field)

        #     #compute complementary energy given coefficient vector
        #     Uc = assemble_scalar(form(sigma[i,j]*eps[i,j]*dx))
            
        #     if idx<=5:
        #         K1[:,idx]= P
        #         K2[idx,idx] = Uc
        #     else:
        #         idx1 = comb_list[idx-6][0]
        #         idx2 = comb_list[idx-6][1]
        #         Kxx = 0.5 * ( Uc - K2[idx1,idx1] - K2[idx2,idx2])
        #         K2[idx1,idx2] = Kxx
        #         K2[idx2,idx1] = Kxx

        # #see https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        # ubar_field.vector.destroy()     #need to add to prevent PETSc memory leak   
        # uhat_field.vector.destroy()     #need to add to prevent PETSc memory leak 

        # # K1_sparse = sparseify(K1,sparse_format='csr')

        # #compute Flexibility matrix
        # self.K1 = K1
        # self.K1_inv = np.linalg.inv(K1)
        # self.S = self.K1_inv.T@K2@self.K1_inv
        # #invert to find stiffness matrix
        # self.K = np.linalg.inv(self.S)


        # AND HERE ^^^^


        #####################################
        ##### alternative computation #######
        #####################################
        if False:
            cell = self.msh.ufl_cell() #get cell type

            c = variable(uflConstant(cell,shape=(6,))) #create vector variable for elastic solution coefficients
            # z = variable(uflConstant(self.msh, PETSc.ScalarType(0.0))) #stand-in variable for axial direction (will not be needed as we take derivative at z=0)
            z = variable(uflConstant(cell))

            #construct mixed tensor function space and functions for specifying warping displacements 
            Uc_e = TensorElement("CG",self.msh.ufl_cell(),1,shape=(3,6))
            Uc = FunctionSpace(self.msh,MixedElement(4*[Uc_e]))
            u_c = Function(Uc)

            #identify individual warping displacment functions
            ubar_c,uhat_c,utilde_c,ubreve_c = split(u_c)
                
            #populate warping displacement functions with information from orthogonalized solutions
            self.construct_warping_fxns(u_c,self.sols_decoup)

            # #construct displacment as a fxn of solution coefficients
            # u = dot(ubar_c,c) + dot(uhat_c,c)*z # + utilde_c*c*pow(z,2)+ ubreve_c*c*pow(z,3)
            ubar = dot(ubar_c,c)
            uhat = dot(uhat_c,c)

            eps_xy = grad(ubar)
            eps_z = uhat
            eps = as_tensor([[eps_z[0],eps_z[1],eps_z[2]],
                        [eps_xy[0,0],eps_xy[1,0],eps_xy[2,0]],
                        [eps_xy[0,1],eps_xy[1,1],eps_xy[2,1]]])

            # #construct strain as a fxn of solution coefficients
            # eps_xy = grad(u)
            # eps_z = diff(u,z)
            # eps = as_tensor([[eps_z[0],eps_z[1],eps_z[2]],
            #             [eps_xy[0,0],eps_xy[1,0],eps_xy[2,0]],
            #             [eps_xy[0,1],eps_xy[1,1],eps_xy[2,1]]])
            # print(eps_xy.ufl_shape)
            # print(eps_z.ufl_shape)
            # print(eps.ufl_shape)

            #get relevant stresses through xs as fxn of solution coefficients
            sigma = as_tensor(self.C[0,j,k,l]*eps[k,l],(j))
            print(sigma.ufl_shape)

            #construct potential energy functional
            x_vec = as_vector([x[0],x[1],0]) # vector for cross-product
            # F = [sigma[i]*dx for i in range(sigma.ufl_shape[0])]
            F = sigma

            M = cross(x_vec,sigma)
            print(M.ufl_shape)
            
            V = FunctionSpace(self.msh,("CG",1))
            v = TestFunction(V)
            L = as_vector([F[0],F[1],F[2],M[0],M[1],M[2]])
            print(L.ufl_shape)
            L_int = sum([L[i]*dx for i in range(L.ufl_shape[0])])
            # print(L_int.ufl_shape)
            #construct jacobian matrix of potential energy functional w.r.t. solution coefficients
            self.K1_alt = diff(L,c)
            print(self.K1_alt.ufl_shape)
            
            # print(self.K1_alt.ufl_shape)
            # self.K1_alt_form = form(self.K1_alt)
            # self.K1_alt_assembled = assemble_matrix(self.K1_alt)
            # self.K1_alt_assembled = assemble_matrix(self.K1_alt_form)

            # #construct sum of Hessian matrices of potential energy functional w.r.t. each soltion coefficient
            # self.K2_alt = sum([diff(self.K1_alt,c)[:,:,i] for i in range(c.ufl_shape)])
            # self.K2_alt_form = form(self.K2_alt)
            # self.K2_alt_assembled = assemble_matrix(self.K2_alt_form)


class CrossSectionAnalytical:
    def __init__(self,params):
        #process params based on shape
        self.shape = params['shape']
        self.E = params['E']
        self.nu = params['nu']

        if self.shape=='rectangle':
            self.h = params['h']
            self.w = params['w']

        elif self.shape =='box':
            self.h = params['h']
            self.w = params['w']
            self.t_h = params['t_h']
            self.t_w = params['t_w']

        elif self.shape == 'circle':
            self.r = params['r']

        elif self.shape == 'hollow circle':
            self.r = params['r']
            self.t = params['t']

        elif self.shape == 'ellipse':
            self.r_x = params['radius_x']
            self.r_y = params['radius_y']

        elif self.shape == 'hollow ellipse':
            self.r_x = params['radius_x']
            self.r_y = params['radius_y']
            self.r_x = params['t_x']
            self.r_y = params['t_y']

        elif self.shape == 'I':
            self.h = params['h']
            self.w = params['w']
            self.t_h = params['t_h'] #flange thickness
            self.t_w = params['t_w'] #web thickness

        else:
            print("busy doing nothing...")
        
    def compute_stiffness(self):

        #### RECTANGULAR XS ####

        ########
        ########
        ########
        ########
        if self.shape == 'rectangle':
            A = (self.h*self.w)
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
            EI1 = self.E*(self.w*self.h**3 /12 )
            EI2 = self.E*(self.h*self.w**3 /12 )
            
            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))
        
        
        #### BOX XS ####
        ########
        #      #
        #      #
        #      #
        ########

        elif self.shape =='box':
            #this torsional model assumes a wall thickness that is less than 10% of the width or height
            A = (self.h*self.w)-((self.h-2*self.t_h)*(self.w-2*self.t_w))
            G = self.E / (2*(1+self.nu))
            
            #from wikipedia page on Timoshenko theory: https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory
            m = ( (self.w*self.t_h) / ((self.h*self.t_w)) )
            n = ( self.w/self.h )
            kappa = ( (10*(1+self.nu)*(1+3*m)**2) /
                      ((12 + 72*m + 150*m**2 + 90*m**3)
                       +self.nu*(11+66*m + 135*m**2 + 90*m**3)
                       +10*n**2*((3+self.nu)*m + 3*m**2)) )
            
            EA = self.E*A
            kGA1=kappa*G*A
            kGA2=kappa*G*A
            EI1 = self.E*(self.w*self.h**3 /12 - ((self.w-2*self.t_w)*(self.h-2*self.t_h)**3) /12)
            EI2 = self.E*(self.h*self.w**3 /12 - ((self.h-2*self.t_h)*(self.w-2*self.t_w)**3) /12)
            
            J = (2*self.t_w*self.t_h*(self.w-self.t_w)**2 * (self.h-self.t_h)**2) / (self.h*self.t_h + self.w*self.t_w - self.t_w**2 - self.t_h**2)
            GJ = G*J

            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))

        #### CIRCULAR XS ######
           #######
         ###########
        #############
        #############
         ###########
           #######
        elif self.shape == 'circle':
            A = np.pi*self.r**2
            G = self.E / (2*(1+self.nu))

            #from wikipedia page on Timoshenko theory: https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory
            kappa = (6*(1+self.nu)) / (7+6*self.nu)
            
            EA = self.E*A
            kGA1=kappa*G*A
            kGA2=kappa*G*A
            I = ((np.pi/4)*(self.r**4))
            EI1 = self.E*I
            EI2 = self.E*I

            J = 2*I
            GJ = G*J

            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))

        #### HOLLOW CIRCULAR XS ######
           #######
         ##       ##
        ##         ##
        ##         ##
         ##       ##
           #######        

        elif self.shape == 'hollow circle':
            A = np.pi*(self.r**2-(self.r-self.t)**2)
            G = self.E / (2*(1+self.nu))

            #from wikipedia page on Timoshenko theory
            m = self.r-self.t / self.r
            kappa = ( (6*(1+self.nu) *(1+m**2)**2 )
                        / ((7+6*self.nu) * (1+m**2)**2 + (20+12*self.nu)*m**2))
            
            EA = self.E*A
            kGA1=kappa*G*A
            kGA2=kappa*G*A
            I=(np.pi/4)*(self.r**4-(self.r-self.t)**4)
            EI1 = self.E*I
            EI2 = self.E*I
            
            J = 2*I
            GJ = G*J

            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))
        
        #### ELLIPSE XS ####
            #############
         ###################
        #####################
        #####################
         ###################
            #############        

        elif self.shape == 'ellipse':
            A = np.pi*self.r_x* self.r_y
            G = self.E / (2*(1+self.nu))

            #from wikipedia page on Timoshenko theory: https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory
            kappa1 = ( (12*(1+self.nu)*self.r_y**2*(3*self.r_y**2 + self.r_x**2))
                    / ( ( (40+ 37*self.nu)*self.r_y**4) + ((16 + 10*self.nu)*self.r_y**2 * self.r_x**2) + self.nu*self.r_x**4) ) 
            kappa2 = ( (12*(1+self.nu)*self.r_x**2*(3*self.r_x**2 + self.r_y**2))
                        / ( ( (40+ 37*self.nu)*self.r_x**4) + ((16 + 10*self.nu)*self.r_x**2 * self.r_y**2) + self.nu*self.r_y**4) ) 
            
            EA = self.E*A
            kGA1=kappa1*G*A
            kGA2=kappa2*G*A
            EI1 = self.E*(np.pi/4)*(self.r_x * self.r_y**3 )
            EI2 = self.E*(np.pi/4)*(self.r_x**3 * self.r_y )
            
            #from https://roymech.org/Useful_Tables/Torsion.html
            J = (np.pi * self.r_x**3 * self.r_y**3) / (self.r_x**2 + self.r_y**2)
            GJ = G*J

            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))

        #### HOLLOW ELLIPSE XS ####
            #############
         ###################
        #####################
        #####################
         ###################
            #############  
        elif self.shape == 'hollow ellipse':
            # inner and outer ellipse are assumed to be similar
            A = np.pi*((self.r_x*self.r_y) - ((self.r_x-self.t_x)* (self.r_y-self.t_y)) )
            G = self.E / (2*(1+self.nu))

            #TODO: fix these kappas
            #from wikipedia page on Timoshenko theory: https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory
            kappa1 = ( (12*(1+self.nu)*self.r_y**2*(3*self.r_y**2 + self.r_x**2))
                    / ( ( (40+ 37*self.nu)*self.r_y**4) + ((16 + 10*self.nu)*self.r_y**2 * self.r_x**2) + self.nu*self.r_x**4) ) 
            kappa2 = ( (12*(1+self.nu)*self.r_x**2*(3*self.r_x**2 + self.r_y**2))
                        / ( ( (40+ 37*self.nu)*self.r_x**4) + ((16 + 10*self.nu)*self.r_x**2 * self.r_y**2) + self.nu*self.r_y**4) ) 
            
            EA = self.E*A
            kGA1=kappa1*G*A
            kGA2=kappa2*G*A
            EI1 = self.E*(np.pi/4)*( (self.r_x * self.r_y**3 ) - ((self.r_x-self.t_x) * (self.r_y-self.t_y)**3 ) )
            EI2 = self.E*(np.pi/4)*( (self.r_x**3 * self.r_y ) - ((self.r_x-self.t_x)**3 * (self.r_y-self.t_y) ) )
            
            #from Roark's Table 10.1
            q = (self.r_x-self.t_w)/self.r_x
            J = ( (np.pi * self.r_x**3 * self.r_y**3) / (self.r_x**2 + self.r_y**2) ) * (1- q )
            GJ = G*J

            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))

        #### I XS #####
        #################
                #
                #
                #
                #
                #
                #
                #
        #################
            
        elif self.shape == 'I':
            A = self.w*self.h - (self.h - 2*self.t_h)*(self.w-self.t_w)
            G = self.E / (2*(1+self.nu))
            
            #from cowper (wikipedia list)
            m = 2*self.w*self.t_h / self.h*self.t_w
            n = self.w/self.h
            kappa1 = ( (10*(1+self.nu)*(1+3*m)**2) / 
                        ((12+72*m+150*m**2+90*m**3) 
                         + self.nu*(11+66*m+135*m**2+90*m**3) 
                         + 30*n**2*(m+m**2) 
                         + 5*self.nu*n*2*(8*m+9*m**2)) )
            #TODO: find the best shear correction factor about the y 
            kappa2 = 5/6 #lacking an accurate answer, we just use the rectangular one

            EA = self.E*A
            kGA1=kappa1*G*A
            kGA2=kappa1*G*A

            #from roark's table 10.2
            J = 1/3 * (2*self.t_h**3*self.w + self.t_w**3*self.h)
            GJ = G*J
            EI1 = self.E*((self.w*self.h**3 /12 ) - (( (self.w-self.t_w) * (self.h - 2*self.t_h)**3 ) /12 ))
            EI2 = self.E*((self.h-2*self.t_h)*self.t_w**3 /12 ) + 2*(self.t_h*self.w**3 / 12)
            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))            
        
        else:
            print('busy doing nothing')
        