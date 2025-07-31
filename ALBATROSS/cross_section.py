from ufl import (Argument,derivative,dot,ds,cross,Identity,sqrt,inner,tr,variable,
                 diff,grad,sin,cos,as_matrix,SpatialCoordinate,FacetNormal,
                 Measure,as_tensor,indices,as_vector,sym,
                 TrialFunction,TestFunction,split)
from basix.ufl import element,mixed_element
from dolfinx.fem import (Constant,Expression,assemble_scalar,form,Function,
                         functionspace,assemble_vector,petsc)
from dolfinx import fem
import numpy as np
from petsc4py import PETSc
from dolfinx.mesh import locate_entities_boundary,meshtags
from dolfinx import geometry # import compute_collisions_trees
from scipy.sparse.linalg import inv,lsqr,spsolve
# import sparseqr
from scipy.sparse import csr_matrix
import ufl 
import pyvista
from dolfinx import plot
from scifem import create_real_functionspace
from dolfinx.cpp.la.petsc import get_local_vectors

from ALBATROSS.material import getMatConstitutiveIsotropic
from ALBATROSS.utils import plot_xdmf_mesh,get_vtx_to_dofs,sparseify
from ALBATROSS.nonmatching_utils import (Region,Separation,Collision,
                                         get_bbtrees,get_collision_celltags,
                                         get_overlap_boundary_facets,
                                         compute_union_polygon,
                                         mesh_from_polygon,
                                         pts_to_dofs,get_petsc_system,
                                         celltags_to_dofs,
                                         get_interpolation_matrix,
                                         get_points_from_cells)
from ALBATROSS.petsc_utils import convert_petsc_to_numpy,AT_C_B
default_scalar_type = PETSc.ScalarType    

#TODO: allow user to specify a point to find xs props about
#TODO: provide a method to translate between different xs values?
#TODO: update sensitivities plotting for higher order basis functions

class CrossSection:
    def __init__(self, msh, materials ,celltags=None,verbose=False,degree=1):
        #analysis domain
        self.msh = msh
        self.verbose = verbose
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

        #Finite element shape function degree: (1=linear,2=quadratic,etc)
        self.degree = degree

        #number of materials
        self.num_mat = len(self.materials)
        
        #indices
        self.i,self.j,self.k,self.l=indices(4)
        self.a,self.B = indices(2)
        
        #DG0 space, used for material properties, etc
        self.Q = functionspace(self.msh,('DG',0))
        #construct DG spaces for modulus of elasticity and poisson ratio (assuming all materials are ISOTROPIC)
        self.E = Function(self.Q)
        self.nu = Function(self.Q)

        #integration measures (subdomain data accounts for different materials)
        if self.ct is not None:
            #check that the number and values of celltags match those specified in the material objects
            mesh_ct = np.unique(self.ct.values)
            mat_ct = np.unique([self.materials[_i].id for _i in range(self.num_mat)] )
            assert(np.logical_and.reduce(mesh_ct==mat_ct))
            
            #TODO: JJK 3/7/25, need to handle case of orthotropic materials

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
            self.dx = Measure("dx",domain=self.msh,subdomain_data=self.ct)

        elif self.ct is None:
            self.dx = Measure("dx",domain=self.msh)
            self.E.x.array[:] = np.full_like(self.E.x.array,self.materials[0].E,dtype=default_scalar_type)
            self.nu.x.array[:] = np.full_like(self.nu.x.array,self.materials[0].nu,dtype=default_scalar_type)
            self.C = getMatConstitutiveIsotropic(self.msh,self.E,self.nu)
        self.ds = Measure("ds",domain=self.msh)
        
        #spatial coordinate and facet normals
        self.x = SpatialCoordinate(self.msh)
        self.VX = functionspace(self.msh,("CG",self.degree,(self.tdim,)))
        self.n = FacetNormal(self.msh)
        
        #compute cross-sectional area and linear density (used for body forces)
        self.A = assemble_scalar(form(1.0*self.dx))
        self.linear_density = 0
        if self.ct is not None:
            for material in self.materials:
                material.A = assemble_scalar(form(1.0*self.dx(material.id)))
                self.linear_density += material.A*material.density
        else:
            self.materials[0].A = assemble_scalar(form(1.0*self.dx))
            self.linear_density += self.materials[0].A*self.materials[0].density
        #TODO: compute density weighted areas and areas of each subdomain?
        
        #compute average y and z locations 
        self.yavg = assemble_scalar(form(self.x[0]*self.dx))/self.A
        self.zavg = assemble_scalar(form(self.x[1]*self.dx))/self.A

        #vectorfunctionspace for initializing displacement functions
        self.recovery_V = functionspace(self.msh,('CG',self.degree,(self.d,)))

        #initialize warping displacement fxn space
        self._set_up_fxnspace_and_fxns()
        
    def get_xs_stiffness_matrix(self):
               
        #construct material constitutive tensor field
        # self.constructConstitutiveField()

        #assemble matrix
        if self.verbose:
            print('Constructing Cross-Section System...')
        self._construct_xs_form()

        #construct the LHS blocks with the Lagrange Multipliers,
        #   assemble LHS
        #   set up RHS forms
        if self.verbose:
            print('Constructing Constraints....')   
        self._construct_KKT_forms()
        
        #set up KSP solver
        if self.verbose:
            print('Computing warping functions....')
        self._set_up_solver()
        self._solve_system()
        
        if self.verbose:
            print('Computing Beam Constitutive Matrix....')
        self._compute_xs_stiffness_matrix()

        print("DONE computing Beam Constitutive Matrix") 

    def _set_up_fxnspace_and_fxns(self):
        # Construct Displacement Coefficient mixed function space
        self.Ve = element("CG",self.msh.topology.cell_name(),self.degree,shape=(self.d,))
        self.V = functionspace(self.msh, mixed_element(4*[self.Ve]))
        
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

        #construct lagrange multpliers
        self.num_constraints=30
        self.LM = create_real_functionspace(self.msh, value_shape=(self.num_constraints,))

        #lagrange multipliers
        self.lmbda = TrialFunction(self.LM)
        self.dlmbda = TestFunction(self.LM)

        #get maps from block vectors ---> warping function & lagrange multiplier vectors
        self.maps = [(self.V.dofmap.index_map, self.V.dofmap.index_map_bs), (self.LM.dofmap.index_map, self.LM.dofmap.index_map_bs)]
        

    def _apply_rotation(self,C,alpha,beta,gamma):
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
        #rotation about Z-axisself.d
        Rz = as_matrix([[cos(gamma),-sin(gamma),0],
                        [sin(gamma),cos(gamma), 0],
                        [0,         0,          1]])
        
        #3D rotation matrix
        R = Rz*Ry*Rx

        Cprime = as_tensor(R[p,i]*R[q,j]*C[i,j,k,l]*R.T[k,r]*R.T[l,s],(p,q,r,s))

        return Cprime
    
    def _construct_mat_orientation(self,orientation):
        #orientation is a list of angles
        self.Q = functionspace(self.msh,("DG",0,(self.d,)))
        self.theta = Function(self.Q)

        self.theta.interpolate(orientation)

    def _construct_xs_form(self,u=None,dx=None,return_form=False):
        if u is None:
            u=self.u
        #geometric dimension
        d = self.d
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
        #trial and test functions
        ubar,uhat,utilde,ubreve=split(u)
        vbar,vhat,vtilde,vbreve=self.vbar,self.vhat,self.vtilde,self.vbreve
        #partial derivatives of trial and test functions
        # ubar_B,uhat_B,utilde_B,ubreve_B=self.ubar_B,self.uhat_B,self.utilde_B,self.ubreve_B
        #partial derivatives of displacement:
        ubar_B = grad(ubar)
        uhat_B = grad(uhat)
        utilde_B = grad(utilde)
        ubreve_B = grad(ubreve)
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
        if dx is None:
            dx = self.dx
        else:
            dx = dx(1)      
        
        #if an orthotropic material is used, the constructMatOrientation method
        #must be called prior to applying rotations
        # if material.type == 'ORTHOTROPIC':
        #     #TODO: need to think about how to store these tensors? 
        #     #We can't store potentially thousands of these, so we need to store the constitutive tensor (per material)
        #     # the rotation angles for each element associated with each cell as a DG0 fxn? 
        #     C = self._apply_rotation(material.C,self.theta[0],self.theta[1],self.theta[2])
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
        
        # n = self.n
        # ds = self.ds
        #traction free boundary conditions
        # Tbar = Ciak1[i,a,k]*uhat[k]*n[a]*vbar[i]*ds \
        #         + CiakB[i,a,k,B]*ubar_B[k,B]*n[a]*vbar[i]*ds 
        # That = 2*Ciak1[i,a,k]*utilde[k]*n[a]*vhat[i]*ds \
        #         + CiakB[i,a,k,B]*uhat_B[k,B]*n[a]*vhat[i]*ds
        # Ttilde = 3*Ciak1[i,a,k]*ubreve[k]*n[a]*vtilde[i]*ds \
        #         + CiakB[i,a,k,B]*utilde_B[k,B]*n[a]*vtilde[i]*ds 
        # Tbreve = CiakB[i,a,k,B]*ubreve_B[k,B]*n[a]*vbreve[i]*ds 

        # Tbar = []
        # for i in range(3):
        #     Tbar+=Ciak1[i,a,k]*uhat[k]*n[a]*ds \
        #         + CiakB[i,a,k,B]*ubar_B[k,B]*n[a]*ds 
        # That=[]
        # for i in range(3):
        #     That += 2*Ciak1[i,a,k]*utilde[k]*n[a]*ds \
        #         + CiakB[i,a,k,B]*uhat_B[k,B]*n[a]*ds
        # Ttilde=[]
        # for i in range(3):
        #     Ttilde += 3*Ciak1[i,a,k]*ubreve[k]*n[a]*ds \
        #         + CiakB[i,a,k,B]*utilde_B[k,B]*n[a]*ds 
        # Tbreve=[]
        # for i in range(3):
        #     Tbreve += CiakB[i,a,k,B]*ubreve_B[k,B]*n[a]*ds 

        # equation 1,2,3
        eq1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
            + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
            - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
            - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \
            # + Tbar
        
        # # equation 4,5,6
        eq2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
            + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
            - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
            - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \
            # + That

        # equation 7,8,9
        eq3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
            - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
            - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\
            # + Ttilde

        #equation 10,11,12
        eq4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\
            # + Tbreve
        
        #construct residual
        a00 = eq1+eq2+eq3+eq4

        if return_form is False:
            self.a00 = a00
        else:
            return a00

    def _construct_constraint_form(self,lmbda,u):
        
        form = inner(lmbda, self.constraints(u)) * self.dx
        
        return form
    
    def _construct_KKT_forms(self,u=None,lmbda=None):
        if u is None:
            u = self.u
        if lmbda is None:
            lmbda = self.lmbda
        #main system block
        a00 = self.a00
        
        #construct constraint forms
        a01 = self._construct_constraint_form(lmbda,self.v)
        a10 = self._construct_constraint_form(self.dlmbda,u)

        a = [[a00, a01], [a10, None]]

        #construct RHS form vector with no body force (e.g. unchanged for each mode)
        f0 = fem.Constant(self.msh, default_scalar_type([0.0]*12)) 
        L0 = inner(self.v, f0) * self.dx

        #assemble the RHS for mode i:
        f1_list = []
        for i in range(6):
            f1_np = np.zeros((self.num_constraints,))
            f1_np[i]= 1.0
            f1_list.append(f1_np)
        L1_list = [inner(fem.Constant(self.msh, default_scalar_type(f1)), self.dlmbda) * self.dx for f1 in f1_list]

        #since we have different RHS's, return the list of L1's i
        L = [L0,L1_list]

        self.a_form = a
        self.L_form = L

        # return a,L
    
    def _set_up_solver(self):

        #assemble matrix and vector
        # pRk/puk is the system stiffness matrix 
        self.pRkpuk_form = fem.form(self.a_form)
        self.pRkpuk = fem.petsc.assemble_matrix_block(self.pRkpuk_form)
        self.pRkpuk.assemble()

        # set up the solver with the LHS
        ksp = PETSc.KSP().create(self.msh.comm)
        ksp.setOperators(self.pRkpuk)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        
        self.solver = ksp

    def _solve_system(self):
        self.solution_vectors= []
        self.warping_functions = []
        self.lmbdas = []
        self.residuals = []
        for idx_k,L1 in enumerate(self.L_form[1]):
            #construct RHS form blocks
            L0 = self.L_form[0]
            L = [L0, L1]
            L_compiled = fem.form(L)

            bcs = []
            b = fem.petsc.assemble_vector_block(L_compiled, self.pRkpuk_form, bcs=bcs)
            xh = fem.petsc.create_vector_block(L_compiled)

            #solve the linear systesm
            self.solver.solve(b, xh)
            xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            #populate the warping function and the lagrange multiplier vectors
            uh = fem.Function(self.V, name="u"+str(idx_k))
            lmbdah = fem.Function(self.LM,name="lmbda"+str(idx_k))

            x_local = get_local_vectors(xh, self.maps)
            uh.x.array[: len(x_local[0])] = x_local[0]
            lmbdah.x.array[: len(x_local[1])] = x_local[1]

            uh.x.scatter_forward()
            lmbdah.x.scatter_forward()

            self.solution_vectors.append(xh.copy())
            self.warping_functions.append(uh.copy())
            self.lmbdas.append(lmbdah.copy())

            # TODO TODO TODO: need to clean up the residual assembly to allow for proper sensitivity computation
            # #TODO: currently, need to do this because we are using a ufl.TestFunction() in the residual construction
            # #       This can be re-written so that uh is used to construct the form, so that we don't have to repeatedly
            # #       re-assemble a00,a10 or a01, just L0 and L1
            # a00_form = self._construct_xs_form(uh,return_form=True)
            # a01_form = inner(lmbdah,self._construct_constraint_form(self.v))*self.dx
            # a10_form = inner(self.dlmbda, self._construct_constraint_form(uh)) * self.dx

            # #main system residual
            # residual00 = a00_form + a01_form - L0 
            # #lagrange multiplier system residual
            # residual10 = a10_form - L1

            # self.residuals.append((residual00,residual10))

            print(f'lagrange multipliers for mode{idx_k}:{x_local[1]}')


    def _compute_xs_stiffness_matrix(self):             
        #unpacking values
        x = self.x
        dx = self.dx
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B

        #construct fenicsx variables pertaining to elastic solution modes
        c7 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c8 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c9 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c10 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c11 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c12 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c = as_tensor([c7,c8,c9,c10,c11,c12])

        #construct general warping displacement functions in terms of the 
        #   elastic solution modes and elastic solution mode coefficients
        u_c = dot(c,as_tensor(self.warping_functions))

        #these elastic solution modes are related by the general expression 
        # for the displacement as:
        # u_c = ubar_c + uhat_c * x1 + utilde_c * x1**2 + ubreve_c * x1**3
        # where x1 is the beam axis direction
                
        # expressions for the stress and strain at x1=0 (beam root)
        eps_c = self.warping2strain(u_c,0) 
        sigma_c = self.warping2stress(u_c,0)

        # get loads over cross-section
        P = self.stress2loads(sigma_c)
        P_form = [Pi*dx for Pi in P]
        
        # internal energy
        Uc = 0.5*sigma_c[i,j]*eps_c[i,j]*dx

        # differentiation of the constructed form 
        self.K1_form = [[diff(P_form[idx1],c[idx2]) for idx2 in range(6)] 
                        for idx1 in range(6)]
        self.K2_form = [[diff(diff(Uc,c[idx1]),c[idx2]) for idx2 in range(6)]
                        for idx1 in range(6)]
        
        #assemble the K1 and K2 matrices
        self.K1 = np.array([[assemble_scalar(form(self.K1_form[idx1][idx2]))
                     for idx2 in range(6)] 
                        for idx1 in range(6)])
        self.K2 = np.array([[assemble_scalar(form(self.K2_form[idx1][idx2]))
                     for idx2 in range(6)] 
                        for idx1 in range(6)])
        
        #store K1^-1 for recovery and sensitivity computation
        self.K1inv = np.linalg.inv(self.K1)

        #stor K2^-1 for sensitivity computation
        self.K2inv = np.linalg.inv(self.K2)
        
        #compute Flexibility matrix
        self.S = self.K1inv.T@self.K2@self.K1inv

        #compute Beam Stiffness Matrix
        self.K =  self.K1.T@self.K2inv@self.K1

    
    def rigid_constraints(self,u):
        '''constraints on averages generalized stresses (forces + moments)'''
        ubar,uhat,_,_ = split(u)
        x1,x2 = self.x[0],self.x[1]

        ubar_r = cross(as_vector([0,x1,x2]),ubar)
        # gradubar = grad(ubar)


        # disp_grad =  as_tensor([[uhat[0], gradubar[0,0], gradubar[0,1]],
        #                     [uhat[1], gradubar[1,0], gradubar[1,1]],
        #                     [uhat[2], gradubar[2,0], gradubar[2,1]],
        #                 ])
        # disp_grad =  as_tensor([[0, gradubar[0,0], gradubar[0,1]],
        #                     [0, gradubar[1,0], gradubar[1,1]],
        #                     [0, gradubar[2,0], gradubar[2,1]],
        #                 ])
        # w = disp_grad + disp_grad.T
        
        U = [ ubar[0],      # translation x
            ubar[1],        # translation y
            ubar[2],        # translation z
            ubar_r[0],      # rotation about x
            ubar_r[1],      # rotation about y #NOTE: THIS IS a RIGID rotation, 
            ubar_r[2]      # rotation about z
            # gradubar[0,0],      # rotation about y (infinitesimal)
            # gradubar[0,1],      # rotation about z  (infinitesimal)
            # w[0,1],
            # w[0,2],
            # w[1,2]
            ]

        return U

    def stress_constraints(self,u,order):
        '''Constraints for average forces/moments'''
        sigma = self.warping2stress(u,order)
        P = self.stress2loads(sigma)

        return P

    def constraints(self,u):
        '''return cross-sectional constraints'''
        P = self.stress_constraints(u,0)
        Px1= self.stress_constraints(u,1)
        Px1_2= self.stress_constraints(u,2)
        Px1_3= self.stress_constraints(u,3)
        U = self.rigid_constraints(u)

        return as_vector(P+Px1+Px1_2+Px1_3+U)

    @staticmethod
    def warping2strain(u,order):
        '''construct strain from the warping functions of a certain polynomial order '''
        try:
            u_list = split(u)
        except:
            u_list = []
            for idx in range(4):
                u_list.append(as_tensor([u[3*idx],u[3*idx+1],u[3*idx+2]]))
        
        gradu = grad(u_list[order])

        if order < 3:
            eps = sym(as_tensor([
                    [(order+1)*u_list[order+1][0], gradu[0,0], gradu[0,1]],
                    [(order+1)*u_list[order+1][1], gradu[1,0], gradu[1,1]],
                    [(order+1)*u_list[order+1][2], gradu[2,0], gradu[2,1]],
                ]))
            
        else:
            eps = sym(as_tensor([
                    [0, gradu[0,0], gradu[0,1]],
                    [0, gradu[1,0], gradu[1,1]],
                    [0, gradu[2,0], gradu[2,1]],
                ]))

        return eps


    def warping2stress(self,u,order):
        i,j,k,l=self.i,self.j,self.k,self.l

        eps = self.warping2strain(u,order)
        sigma = as_tensor(self.C[i,j,k,l]*eps[k,l],(i,j))

        return sigma


    def stress2loads(self,sigma):
        x1,x2 = self.x[0],self.x[1]
        sigma11 = sigma[0,0]
        sigma12 = sigma[1,0]
        sigma13 = sigma[2,0]

        m = cross(as_vector([0,x1,x2]),
                    as_vector([sigma11,sigma12,sigma13]))
        
        P = [   sigma11,    # extension
                sigma12,    # shear 1
                sigma13,    # shear 2
                m[0],       # torsion
                m[1],       # bending 1
                m[2]        # bending 2
                    ]
        
        return P

    # def _assemble_system_matrix(self,residual = None):
        # if residual is None:
        #     self.system_mat = petsc.assemble_matrix(form(self.Residual))
        #     self.system_mat.assemble()
        # else:
        #     system_mat = petsc.assemble_matrix(form(residual))
        #     system_mat.assemble()
        #     return system_mat


    # def _get_modes(self):
        
    #     m,n1=self.system_mat.getSize()
    #     if self.verbose:
    #         print('Computing QR factorization')
    #     Acsr = csr_matrix(self.system_mat.getValuesCSR()[::-1], shape=self.system_mat.size)
        
    #     #perform QR factorization and store as struct in householder form
    #     QR= sparseqr.qr_factorize( Acsr.transpose() )

    #     #build matrix of unit vectors for selecting last 12 columns
    #     X = np.zeros((m,12))
    #     for i in range(12):
    #         X[m-1-i,11-i]=1

    #     #perform matrix multiplication implicitly to construct orthogonal nullspace basis
    #     self.sols = sparseqr.qmult(QR,X)
    #     # Q,_ = np.linalg.qr(Acsr.transpose().toarray())
    #     # self.sols  = Q[:,-12:]
    #     self.sparse_sols = sparseify(self.sols,sparse_format='csc')
    #     # self.sols = self.sparse_sols.toarray()

    # def _decouple_modes(self,basis_matrix_only=False):
    #     #this is a change of basis operation from the standard R^12 basis to
    #     #   the basis defined by the 6 rigid body modes and the 6 elastic modes
    #     #
    #     #the change of basis matrix can be easily computed by simply evaluating
    #     #   the functions defining the rigid+elastic basis at all the dofs
    #     x = self.x
    #     dx = self.dx
    #     C = self.C
    #     #indices
    #     i,j,k,l=self.i,self.j,self.k,self.l
    #     a,B = self.a,self.B

    #     # get collapsed subspace and maps from subspaces to parent space 
    #     UBAR,self.ubar_vtx_to_dof = self.V.sub(0).collapse()
    #     UHAT,self.uhat_vtx_to_dof = self.V.sub(1).collapse()
    #     _,self.utilde_vtx_to_dof = self.V.sub(2).collapse()
    #     _,self.ubreve_vtx_to_dof = self.V.sub(3).collapse()

    #     #GET UBAR AND UHAT RELATED MODES
    #     ubar_modes = self.sols[self.ubar_vtx_to_dof,:]
    #     uhat_modes = self.sols[self.uhat_vtx_to_dof,:]

    #     #CONSTRUCT FUNCTION FOR UBAR AND UHAT SOLUTIONS GIVEN EACH MODE
    #     ubar_mode = Function(UBAR)
    #     uhat_mode = Function(UHAT)

    #     #INITIALIZE DECOUPLING MATRIX (12X12)
    #     self.mat = np.zeros((6,12))

    #     #HERES THE NEW APPROACH:
    #     #what we want is the set of warping functions Nbar and Nhat
    #     # the other warping functions have no effect on the beam stiffness matrix or sensitivities
    #     # so we'll first extract ubar and uhat
    #     # then we'll use the gram-schmidt process to factor out the rigid body modes from ubar
    #     # rigid body translation and displacement only affect ubar, no other warping function
    #     # so... we can orthogonalize ubar and explicitly construct a reduced basis transformation matrix M_e
    #     # that only considers the elastic modes, which we can decouple with a 6x6 matrix in the same manner as below
                
    #     #LOOP THROUGH MAT'S COLUMN (EACH MODE IS A COLUMN OF MAT):
    #     for mode in range(self.mat.shape[1]):
    #         #construct function from mode
    #         ubar_mode.vector.array = ubar_modes[:,mode]
    #         uhat_mode.vector.array = uhat_modes[:,mode]
          
    #         #get stress from warping functions
    #         sigma = self.warping2stress(ubar_mode,uhat_mode)

    #         #relevant components of stress tensor
    #         sigma11 = sigma[0,0]
    #         sigma12 = sigma[1,0]
    #         sigma13 = sigma[2,0]

    #         #integrate stresses over cross-section at "root" of beam and construct xs load vector
    #         P1 = assemble_scalar(form(sigma11*dx))
    #         V2 = assemble_scalar(form(sigma12*dx))
    #         V3 = assemble_scalar(form(sigma13*dx))
            
    #         T1 = assemble_scalar(form( (((x[0])*(sigma13)) - ((x[1])*(sigma12)))*dx))
    #         M2 = assemble_scalar(form((x[1])*(sigma11)*dx))          
    #         M3 = assemble_scalar(form(-(x[0])*(sigma11)*dx))  
            
    #         # AVERAGE FORCE (COMPUTED WITH UBAR AND UHAT)
    #         self.mat[0,mode]=P1
    #         self.mat[1,mode]=V2
    #         self.mat[2,mode]=V3   

    #         #AVERAGE MOMENTS (COMPUTED WITH UBAR AND UHAT)
    #         self.mat[3,mode]=T1
    #         self.mat[4,mode]=M2
    #         self.mat[5,mode]=M3

    #     if basis_matrix_only is False:
    #         mat_sparse = sparseify(self.mat,sparse_format='csc')

    #         # self.sols_decoup = (self.sparse_sols.dot(inv(mat_sparse))).toarray()
    #         # self.sols_decoup = self.sols@np.linalg.inv(mat)
    #         # self.sols_decoup = self.sols@self.mat.T
    #         # ubar_uhat_dofs = np.concatenate([self.ubar_vtx_to_dof,self.uhat_vtx_to_dof])
    #         # sparse_sols = sparseify(self.sols[ubar_uhat_dofs,:])
    #         # # # self.sols_decoup = self.sols[ubar_uhat_dofs,:]@self.mat.T
    #         # self.sols_decoup = sparse_sols.dot(mat_sparse.T).toarray()

    #         ubar_uhat_dofs = np.concatenate([self.ubar_vtx_to_dof,self.uhat_vtx_to_dof])
    #         # # self.sols_decoup = self.sols[ubar_uhat_dofs,:]@self.mat.T
    #         # self.sols_decoup = (self.sparse_sols.dot(mat_sparse.T).toarray())[ubar_uhat_dofs,:]
    #         # self.sols_decoup = (self.sparse_sols.dot(mat_sparse.T).toarray())

    #         #USING PSEUDOINVERSE
    #         mat_pinv = sparseify(np.linalg.pinv(mat_sparse.toarray()))

    #         # print(f"condition number of basis transformation:{np.linalg.cond(self.mat)}")
    #         self.sols_decoup = self.sparse_sols.dot(mat_pinv).toarray()
    #         # self.sols_decoup=mat@self.sols
    #         print()


    # def _build_elastic_solution_modes(self):
    #     #Initialize a tensor element and mixed tensor function space 
    #     # for the elastic solution modes
    #     Ne = element('CG',self.msh.topology.cell_name(),self.degree,shape=(3,6))
    #     self.N_space = functionspace(self.msh,mixed_element(2*[Ne]))
    #     self.N = Function(self.N_space)
        
    #     #extract portions of elastic solution mode function related to each warping fxn
    #     self.N_bar, self.N_hat = self.N.split() 

    #     #get map of function dofs 
    #     N_bar_vtx_to_dofs = self.N_space.sub(0).collapse()[1]
    #     N_hat_vtx_to_dofs = self.N_space.sub(1).collapse()[1]
    #     # N_tilde_vtx_to_dofs = self.N_space.sub(2).collapse()[1]
    #     # N_breve_vtx_to_dofs = self.N_space.sub().collapse()[1]

    #     #get separate elastic solution mode values
    #     N_bar_vals = sparseify(self.sols_decoup[self.ubar_vtx_to_dof,:]).toarray().flatten()
    #     N_hat_vals = sparseify(self.sols_decoup[self.uhat_vtx_to_dof,:]).toarray().flatten()
        
    #     #populate elastic solution modes to elastic solution mode function
    #     self.N_bar.vector.array[N_bar_vtx_to_dofs] = N_bar_vals
    #     self.N_hat.vector.array[N_hat_vtx_to_dofs] = N_hat_vals

    def _get_stiffness_contribution(self,dx=None):             
        #unpacking values
        x = self.x
        if dx is None:
            dx = self.dx
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
   
        #elastic solution mode function related to each warping fxn
        N_bar = self.N_bar
        N_hat = self.N_hat

        #construct fenicsx variables pertaining to elastic solution modes
        c7 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c8 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c9 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c10 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c11 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c12 = variable(Constant(self.msh,PETSc.ScalarType((1.0))))
        c = as_tensor([c7,c8,c9,c10,c11,c12])

        #construct general warping displacement functions in terms of the 
        #   elastic solution modes and elastic solution mode coefficients
        ubar_c = dot(N_bar,c)
        uhat_c = dot(N_hat,c)
        # utilde_c = dot(N_tilde,c)
        # ubreve_c = dot(N_breve,c)

        #these elastic solution modes are related by the general expression 
        # for the displacement as:
        # u_c = ubar_c + uhat_c * x1 + utilde_c * x1**2 + ubreve_c * x1**3
        # where x1 is the beam axis direction
                
        # expressions for the stress and strain in terms of the polynomial 
        # from expansion above:
        eps_c = self.warping2strain(ubar_c,uhat_c)
        sigma_c = self.warping2stress(ubar_c,uhat_c)

        #only stresses with a 1x component are of concern:
        sigma11_c = sigma_c[0,0]
        sigma12_c = sigma_c[0,1]
        sigma13_c = sigma_c[0,2]

        #construct expression for the load applied to a cross-section in 
        # terms of stress and strain expressions defined based on  the 
        # polynomial expansion:
        P1 = sigma11_c*dx
        V2 = sigma12_c*dx
        V3 = sigma13_c*dx
        T1 = ((x[0])*sigma13_c - (x[1])*sigma12_c)*dx
        M2 = (x[1])*sigma11_c*dx
        M3 = -(x[0])*sigma11_c*dx

        #store loads in a list instead of a ufl vector as we cannot take 
        # variable derivatives of non-scalar forms
        P = [P1,V2,V3,T1,M2,M3]
        
        # construct expression for the internal energy of the beam based on
        # the polynomial expansion:
        Uc = 0.5*sigma_c[i,j]*eps_c[i,j]*dx

        # differentiation of the constructed form 
        K1_form = [[diff(P[idx1],c[idx2]) for idx2 in range(6)] 
                        for idx1 in range(6)]
        K2_form = [[diff(diff(Uc,c[idx1]),c[idx2]) for idx2 in range(6)]
                        for idx1 in range(6)]
        
        #assemble the K1 and K2 matrices
        K1 = np.array([[assemble_scalar(form(K1_form[idx1][idx2]))
                     for idx2 in range(6)] 
                        for idx1 in range(6)])
        K2 = np.array([[assemble_scalar(form(K2_form[idx1][idx2]))
                     for idx2 in range(6)] 
                        for idx1 in range(6)])
        
        K1 = sparseify(K1).toarray()
        K2 = sparseify(K2).toarray()
        
        #store K1^-1 for recovery and sensitivity computation
        K1inv = np.linalg.inv(K1)
        K1inv = sparseify(K1inv).toarray()

        #store K2^-1 for sensitivity computation
        K2inv = np.linalg.inv(K2)
        K2inv = sparseify(K2inv).toarray()
        
        #compute Flexibility matrix
        S = K1inv.T@K2@K1inv

        #compute stiffness matrix
        K = K1@sparseify(K2inv).toarray()@K1.T

        return K

    #TODO: NEED TO UPDATE WITH ADJOINT SENSITIVITY CODE (REQUIRES FIXES TO RESIDUAL ASSEMBLY)
    def compute_xs_stiffness_matrix_sensitivities(self):
        #TODO: combine EB and TS sensitivities...
        args = self.K1_form[0][0].arguments()
        n = max(a.number() for a in args) if args else -1
        dX = Argument(self.VX,n+1)
        # n = max(a.number() for a in args) if args else -1
        # du2 = Argument(self.VX,n+1)
        # du = Argument(self.VX,0) #there are no arguments in any of these forms?
        self.dK1dx_form = [[derivative(self.K1_form[idx1][idx2],self.x,dX)
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
        self.dK2dx_form = [[derivative(self.K2_form[idx1][idx2],self.x,dX)
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
        # self.dK1dx = np.array([[petsc.assemble_vector(form(self.dK1dx_form[idx1][idx2]))
        #                 for idx1 in range(6)] 
        #                     for idx2 in range(6)])
        # self.dK2dx = np.array([[petsc.assemble_vector(form(self.dK2dx_form[idx1][idx2]))
        #         for idx1 in range(6)] 
        #             for idx2 in range(6)])
        
        self.dK1dx = [[petsc.assemble_vector(form(self.dK1dx_form[idx1][idx2]))
                        for idx2 in range(6)] 
                            for idx1 in range(6)]
        self.dK2dx = [[petsc.assemble_vector(form(self.dK2dx_form[idx1][idx2]))
                for idx2 in range(6)] 
                    for idx1 in range(6)]

        #TODO: np arrays are likely contributing to numerical inaccuracies
        # different idea: flatten across the 36 stiffness matrix entries and use sparse matrices in petsc or scipy
        #make dK2dx matrix:
        import scipy.sparse as sp

        def return_sparse_mat(l_of_l_of_vec):
            flat_list = [vec for row in l_of_l_of_vec for vec in row]
            sparse_list = []
            for vec in flat_list:
                sparse_list.append(sparseify(vec.array))
            sparse_mat = sp.vstack(sparse_list)
            return sparse_mat


        self.dK1dx_sparse = return_sparse_mat(self.dK1dx)
        self.dK2dx_sparse = return_sparse_mat(self.dK2dx)
        
        self.dK1dx = self.dK1dx_sparse.toarray().reshape((6,6,self.dK1dx_sparse.shape[1]))
        self.dK2dx = self.dK2dx_sparse.toarray().reshape((6,6,self.dK2dx_sparse.shape[1]))
        # K1inv_sparse = sparseify(self.K1inv.flatten())
        # K1_sparse = sparseify(self.K1inv.flatten())
        # K2_sparse = sparseify(self.K2.flatten())

        # #try to zero out the near zero entries
        self.K1inv = sparseify(self.K1inv).toarray()
        self.K1 = sparseify(self.K1).toarray()
        self.K2 = sparseify(self.K2).toarray()

        #boundary dofs ([:,:,self.boundary_dofs])
        self.boundary_nodes = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
        
        #TODO: can simplify this
        #compact einsums:
        term1 = np.einsum("ijm,ik,kl->jlm", self.dK1dx, self.K2inv, self.K1)
        term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1.T,self.K2inv,self.dK2dx,self.K2inv,self.K1)
        term3 = np.einsum("ij,jk,lkm->ilm", self.K1.T, self.K2inv, self.dK1dx)

        #full sensitivities
        self.dKdx = term1 + term2 + term3 
               
        #get map from vtx to dofs to restrict to boundary (this only works for CG1)
        self.boundary_dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.VX.value_size)
        indices_to=[]
        for i in range(self.VX.num_sub_spaces):
            _,map_to = self.VX.sub(i).collapse()
            indices_to.extend(map_to)
        self.boundary_dof_to_vertex_map = self.boundary_dof_to_vertex_map[np.argsort(indices_to)]

        #find all the indices where the boundary_node is in the boundary_dof_to_vertex_map and save those indices as a list
        
        boundary_indices = []
        for i in self.boundary_nodes:
            boundary_indices.extend(list(np.where(self.boundary_dof_to_vertex_map==i)[0]))
        
        self.dKdx_boundary = self.dKdx[:,:,boundary_indices]

         
    # def warping2strain(self,ubar,uhat):
    #     gradubar=grad(ubar)

    #     #derivatives of displacement
    #     #this is known from our displacement expression
    #     dubxdx = uhat[0]
    #     dubxdy = uhat[1]
    #     dubxdz = uhat[2]
    #     dubydx = gradubar[0,0]
    #     dubydy = gradubar[1,0]
    #     dubydz = gradubar[2,0]
    #     dubzdx = gradubar[0,1]
    #     dubzdy = gradubar[1,1]
    #     dubzdz = gradubar[2,1]

    #     #form ufl displacement for grad(u_i)
    #     gradu = as_tensor([[dubxdx,dubxdy,dubxdz],
    #                     [dubydx,dubydy,dubydz],
    #                     [dubzdx,dubzdy,dubzdz]])
        
    #     #ensure that strains are symmetric
    #     # eps = 0.5 * (gradu + gradu.T)
    #     eps = gradu

    #     return eps 

    # def warping2stress(self,ubar,uhat):
    #     i,j,k,l=self.i,self.j,self.k,self.l
    #     eps = self.warping2strain(ubar,uhat)

    #     stress = as_tensor(self.C[i,j,k,l]*eps[k,l],(i,j))
        
    #     return stress 
    
    # def warping2loads(self,ubar,uhat):

    def recover_stress(self,reactions):
        c = self.K1inv@reactions

        c_const=Constant(self.msh,PETSc.ScalarType(c))
        ubar = dot(self.N_bar,c_const)
        uhat = dot(self.N_hat,c_const)
        # utilde = dot(self.N_tilde,c_const)
        # ubreve = dot(self.N_breve,c_const)

        stress = self.warping2stress(ubar,uhat)
        return stress

    def get_von_mises_stress(self,stress):
        #deviatoric stress
        s = stress - 1. / 3 * tr(stress) * Identity(stress.ufl_shape[0])
        von_Mises = sqrt(3. / 2 * inner(s, s))
        V_von_mises = functionspace(self.msh, ("DG", 0))
        stress_expr = Expression(von_Mises, V_von_mises.element.interpolation_points())
        stresses = Function(V_von_mises)
        stresses.interpolate(stress_expr)
        
        return stresses

    def getXSMassMatrix(self):
        #compute xs mass properties:
        self.M = np.zeros((6,6))
  
    def plot_mesh(self):
        plot_xdmf_mesh(self.msh)

    def plot_warping_fxns(self,fxn_order=0):
        elastic_sols = np.zeros((self.warping_functions[0].sub(0).collapse().x.array.shape[0],6))
        for i in range(6):
            elastic_sols[:,i]=self.warping_functions[i].sub(fxn_order).collapse().x.array

        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()
        mode = ['Axial','Shear 1', 'Shear 2', 'Torsion', 'Bending 1', 'Bending 2']
        plotter = pyvista.Plotter(shape=(2,3))
        grids = []
        warped = []
        for i in range(6):
            row = int(i/3)
            col = i%3
            name = f'mode_{i}'
            plotter.subplot(row,col)
            #plot mesh
            tdim = self.msh.topology.dim
            
            V0,V0_to_V = self.V.sub(0).collapse()
            topology, cell_types, geom = plot.vtk_mesh(V0)
            grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))
            
            cnp = np.zeros((6,1))
            cnp[i,:] = 1

            warping_sol = elastic_sols@cnp
            
            solution_mode = warping_sol.reshape((geom.shape[0], 3))[:,[1,2,0]]
            grids[i][name]= solution_mode/np.max(np.linalg.norm(solution_mode,axis=1))
            print(f"maximum magnitude for mode: {np.max(np.linalg.norm(solution_mode,axis=1))}")
            warped.append(grids[i].warp_by_vector(name,factor=.1))


            plotter.add_mesh(warped[i],show_edges=True,opacity=.9,scalar_bar_args={'title': 'Norm of Disp. Magnitude'},)
            plotter.add_mesh(grids[i],show_edges=True,opacity=0.75,color='gray',style="wireframe",show_scalar_bar=False)
            plotter.add_text(mode[i])
            plotter.view_isometric()
            plotter.show_bounds(location='outer',
                                show_zlabels=False,
                                n_xlabels=2,
                                n_ylabels=2,
                                n_zlabels=2)
        if not pyvista.OFF_SCREEN:
            plotter.show()

    def plot_warping_strain(self,component=(0,0)):
        '''
        Plot the warping strain from the warping displacement functions
        component: tuple of the strain entry to plot
                    for example: (0,0) = xx, 
                                 (1,2)=yz
        '''
        
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()

        mode = ['Axial','Shear 1', 'Shear 2', 'Torsion', 'Bending 1', 'Bending 2']
        plotter = pyvista.Plotter(shape=(2,3))
        grids = []
        warped = []
        for i in range(6):
            row = int(i/3)
            col = i%3
            name = f'mode_{i}'
            plotter.subplot(row,col)
            #plot mesh
            tdim = self.msh.topology.dim
            
            c_np = np.zeros((6,))
            c_np[i] = 1

            c = Constant(self.msh,PETSc.ScalarType(c_np))

            ubar = dot(self.N_bar,c)
            uhat = dot(self.N_hat,c)
            eps_ufl = self.warping2strain(ubar,uhat)
            #Vstrain is a scalar functionspace for only one strain component 
            Vstrain = functionspace(self.msh,("DG",0)) 
            # Vstrain = functionspace(self.msh,("DG",0,(self.d,self.d))) 
            strain_component=fem.Expression(eps_ufl[component], Vstrain.element.interpolation_points())
            # strain_to_plot = fem.Function(Vstrain.sub(0).collapse()[0])
            strain_to_plot = fem.Function(Vstrain)
            strain_to_plot.interpolate(strain_component)
            
            V0,V0_to_V = self.V.sub(0).collapse()
            topology, cell_types, geom = plot.vtk_mesh(V0)
            grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

            grids[i][name]= strain_to_plot.vector.array#/np.max(np.linalg.norm(solution_mode,axis=1))

            plotter.add_mesh(grids[i],show_edges=True,opacity=.75,scalar_bar_args={'title': f'warping mode {i}'})
            plotter.add_text(mode[i])
            plotter.view_xy()
            plotter.show_bounds()
        if not pyvista.OFF_SCREEN:
            plotter.show()


    def plot_sensitivities(self):
        plotter = pyvista.Plotter(shape=(2,3))
        grids = []
        warped = []
        for i in range(6):
            row = int(i/3)
            col = i%3
            plotter.subplot(row,col)
            #plot mesh
            tdim = self.msh.topology.dim
            topology, cell_types, geom = plot.vtk_mesh(self.msh, tdim)
            grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

            sensitivity_to_plot = np.zeros((geom.shape[0],2))

            sensitivity_to_plot[self.boundary_nodes,:] = self.dKdx[i,i,:].reshape(-1,2)[self.boundary_nodes,:]

            sensitivity = np.concatenate([sensitivity_to_plot,np.zeros_like(sensitivity_to_plot)],axis=1)

            sensitivity = np.concatenate([sensitivity_to_plot,np.zeros((sensitivity_to_plot.shape[0],1))],axis=1)

            grids[i].point_data["sensitivity"] = sensitivity
            norm = np.linalg.norm(sensitivity,2)
            # print(norm)
            warped.append(grids[i].warp_by_vector("sensitivity",factor=1/(norm)))
            plotter.add_mesh(grids[i],show_edges=True,opacity=0.5,scalar_bar_args={'title': f'Sensitivity_{i}'})
            plotter.add_mesh(warped[i],show_edges=True,opacity=1,scalar_bar_args={'title': f'Sensitivity_{i}'})
            plotter.add_text(f'dK/dx({i},{i})')
            plotter.view_xy()
            plotter.add_axes()
        plotter.subplot(0,0)
        plotter.show_bounds()
        if not pyvista.OFF_SCREEN:
            plotter.show()
    
    #========== EB ARCHIVE =========#
    # def get_xs_stiffness_matrix_EB(self):
        
    #     #construct material constitutive tensor field
    #     # self.constructConstitutiveField()

    #     if self.verbose:
    #         print('Constructing Cross-Section System...')
    #     self._construct_residual()

    #     if self.verbose:
    #         print('Assembling System Matrix....')   
    #     self._assemble_system_matrix()

    #     if self.verbose:
    #         print('Computing non-trivial solutions....')
    #     self._get_modes()

    #     if self.verbose:
    #         print('Orthogonalizing w.r.t. elastic modes...')
    #     self._decouple_modes()
    #     self._build_elastic_solution_modes_EB()
        
    #     if self.verbose:
    #         print('Computing Beam Constitutive Matrix....')
    #     self._compute_xs_stiffness_matrix_EB()

    #     print("DONE computing Beam Constitutive Matrix")  
    
    # def _build_elastic_solution_modes_EB(self):
    #     #Initialize a tensor element and mixed tensor function space 
    #     # for the elastic solution modes
    #     Ne = element('CG',self.msh.topology.cell_name(),self.degree,shape=(3,4))
    #     self.N_space = functionspace(self.msh,mixed_element(4*[Ne]))
    #     self.N = Function(self.N_space)
        
    #     #extract portions of elastic solution mode function related to each warping fxn
    #     self.N_bar, self.N_hat, self.N_tilde, self.N_breve = self.N.split() 

    #     #unpack elastic solution modes
    #     elastic_sols = np.concatenate([self.sols_decoup[:,6:7],self.sols_decoup[:,9:]],axis=1)

    #     #get map of function dofs 
    #     N_bar_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(0))
    #     N_hat_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(1))
    #     N_tilde_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(2))
    #     N_breve_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(3))

    #     #get separate elastic solution mode values
    #     N_bar_vals = elastic_sols[self.ubar_vtx_to_dof.flatten(),:]
    #     N_hat_vals = elastic_sols[self.uhat_vtx_to_dof.flatten(),:]
    #     N_tilde_vals = elastic_sols[self.utilde_vtx_to_dof.flatten(),:]
    #     N_breve_vals = elastic_sols[self.ubreve_vtx_to_dof.flatten(),:]

    #     #populate elastic solution modes to elastic solution mode function
    #     self.N_bar.vector.array[N_bar_vtx_to_dofs.flatten()] = N_bar_vals.flatten()
    #     self.N_hat.vector.array[N_hat_vtx_to_dofs.flatten()] = N_hat_vals.flatten()
    #     self.N_tilde.vector.array[N_tilde_vtx_to_dofs.flatten()] = N_tilde_vals.flatten()
    #     self.N_breve.vector.array[N_breve_vtx_to_dofs.flatten()] = N_breve_vals.flatten()

    # def _compute_xs_stiffness_matrix_EB(self):             
    #     #unpacking values
    #     x = self.x
    #     dx = self.dx
    #     #indices
    #     i,j,k,l=self.i,self.j,self.k,self.l
    #     a,B = self.a,self.B
   
    #     #elastic solution mode function related to each warping fxn
    #     N_bar = self.N_bar
    #     N_hat = self.N_hat
    #     N_tilde = self.N_tilde
    #     N_breve = self.N_breve 

    #     #construct fenicsx variables pertaining to elastic solution modes
    #     c7 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     # c8 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     # c9 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     c10 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     c11 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     c12 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
    #     c = as_tensor([c7,c10,c11,c12])

    #     #construct general warping displacement functions in terms of the 
    #     #   elastic solution modes and elastic solution mode coefficients
    #     ubar_c = dot(N_bar,c)
    #     uhat_c = dot(N_hat,c)
    #     utilde_c = dot(N_tilde,c)
    #     ubreve_c = dot(N_breve,c)

    #     #these elastic solution modes are related by the general expression 
    #     # for the displacement as:
    #     # u_c = ubar_c + uhat_c * x1 + utilde_c * x1**2 + ubreve_c * x1**3
    #     # wereh x1 is the beam axis direction
                
    #     # expressions for the stress and strain in terms of the polynomial 
    #     # from expansion above:
    #     eps_c = self.warping2strain(ubar_c,uhat_c,utilde_c,ubreve_c)
    #     sigma_c = self.warping2stress(ubar_c,uhat_c,utilde_c,ubreve_c)

    #     #only stresses with a 1x component are of concern:
    #     sigma11_c = sigma_c[0,0]
    #     sigma12_c = sigma_c[0,1]
    #     sigma13_c = sigma_c[0,2]

    #     #construct expression for the load applied to a cross-section in 
    #     # terms of stress and strain expressions defined based on  the 
    #     # polynomial expansion:
    #     P1 = sigma11_c*dx
    #     # V2 = sigma12_c*dx
    #     # V3 = sigma13_c*dx
    #     T1 = -((x[0])*sigma13_c - (x[1])*sigma12_c)*dx
    #     M2 = -(x[1])*sigma11_c*dx
    #     M3 = (x[0])*sigma11_c*dx

    #     #store loads in a list instead of a ufl vector as we cannot take 
    #     # variable derivatives of non-scalar forms
    #     P = [P1,T1,M2,M3]
        
    #     # construct expression for the internal energy of the beam based on
    #     # the polynomial expansion:
    #     Uc = 0.5*sigma_c[i,j]*eps_c[i,j]*dx

    #     #now we begin the differentiation, form construction, and form assembly
    #     # to get K1 & K2 as well as dK1dx & dK2dx (used for shape optimization)
    #     self.K1_form = [[diff(P[idx1],c[idx2]) for idx1 in range(4)] 
    #                     for idx2 in range(4)]
    #     self.K2_form = [[diff(diff(Uc,c[idx1]),c[idx2]) for idx1 in range(4)]
    #                     for idx2 in range(4)]
        
    #     self.K1 = np.array([[assemble_scalar(form(self.K1_form[idx1][idx2]))
    #                  for idx1 in range(4)] 
    #                     for idx2 in range(4)])
    #     self.K2 = np.array([[assemble_scalar(form(self.K2_form[idx1][idx2]))
    #                  for idx1 in range(4)] 
    #                     for idx2 in range(4)])
        
    #     #store K1^-1 for recovery and sensitivity computation
    #     self.K1inv = np.linalg.inv(self.K1)
        
    #     #compute Flexibility matrix
    #     self.S = self.K1inv.T@self.K2@self.K1inv
        
    #     #invert Flexibility matrix to find beam constitutive matrix
    #     self.K = np.linalg.inv(self.S)
    
    # def compute_xs_stiffness_matrix_sensitivities_EB(self):
    #     args = self.K1_form[0][0].arguments()
    #     n = max(a.number() for a in args) if args else -1
    #     du = Argument(self.VX,n+1)
    #     # du = Argument(self.VX,0) #there are no arguments in any of these forms?

    #     m = 4
    #     self.dK1dx_form = [[derivative(self.K1_form[idx1][idx2],self.x,du)
    #                         for idx1 in range(m)] 
    #                             for idx2 in range(m)]
    #     self.dK2dx_form = [[derivative(self.K2_form[idx1][idx2],self.x,du)
    #                         for idx1 in range(m)] 
    #                             for idx2 in range(m)]
    #     self.dK1dx = np.array([[petsc.assemble_vector(form(self.dK1dx_form[idx1][idx2]))
    #                     for idx1 in range(m)] 
    #                         for idx2 in range(m)])     
    #     self.dK2dx = np.array([[petsc.assemble_vector(form(self.dK2dx_form[idx1][idx2]))
    #             for idx1 in range(m)] 
    #                 for idx2 in range(m)])
        
    #     #boundary dofs ([:,:,self.boundary_dofs])
    #     self.boundary_dofs = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
        
    #     #use chain rule for derivative of flexibility matrix dSdx:
    #     #first term of dSdx
    #     self.dK1invT = -np.einsum('ijk,ij->ijk',
    #                          self.K1inv.T @ self.dK1dx.transpose(1,0,2),
    #                            self.K1inv.T @ self.K2 @ self.K1inv ) 
    #     #second term of dSdx
    #     self.dK2 = np.einsum('ijk,ij->ijk',
    #                     self.K1inv.T@self.dK2dx,
    #                     self.K1inv)
        
    #     #third term of dSdx
    #     self.dK1inv = -np.einsum('ijk,ij->ijk',
    #                         self.K1inv.T @ self.K2 @ self.K1inv @ self.dK1dx,
    #                           self.K1inv)

    #     #add terms to get dSdx
    #     self.dSdx = self.dK1invT + self.dK2 + self.dK1inv

    #     #compute derivative of stiffness matrix (dKdx) from derivative of flexibility matrix (dSdx)
    #     self.dKdx = - np.einsum('ijk,ij->ijk',
    #                             self.K @ self.dSdx,
    #                             self.K)
    
    def _orthonormalize_rbm(self,fxn,verbose=False):
        V = fxn.function_space
        x = self.x
        dx  = self.dx

        #Rigid Body Modes expression (3D)
        rbms = [
            fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((1.0,0.0,0.0))),V.element.interpolation_points()),
            fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((0.0,1.0,0.0))),V.element.interpolation_points()),
            fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((0.0,0.0,1.0))),V.element.interpolation_points()),
            fem.Expression(ufl.as_vector([0,-x[1],x[0]]),V.element.interpolation_points())#,
            # fem.Expression(ufl.as_vector([x[1],0,0]),V.element.interpolation_points()),
            # fem.Expression(ufl.as_vector([-x[0],0,0]),V.element.interpolation_points())
        ]

        # List of functions to orthogonalise
        vx = fem.Function(V)
        vy = fem.Function(V)
        vz = fem.Function(V)
        vrx = fem.Function(V)
        # vry = fem.Function(V)
        # vrz = fem.Function(V)
        vx.interpolate(rbms[0])
        vy.interpolate(rbms[1])
        vz.interpolate(rbms[2])
        vrx.interpolate(rbms[3])
        # vry.interpolate(rbms[4])
        # vrz.interpolate(rbms[5])

        # v = list((vx,vy,vz,vrx,vry,vrz))
        v = list((vx,vy,vz,vrx))

        # GS Projection
        def proj(u, v):
            res = fem.assemble_scalar(fem.form(inner(u, v)*dx))/fem.assemble_scalar(fem.form(inner(u, u)*dx)) * u.vector.array
            return res

        # GS orthogonalisation
        def ortho(v):
            xi = [None]*len(v)
            xi[0] = v[0]
            for j in range(1, len(xi)):
                xi[j] = fem.Function(V)
                xi[j].vector.array = v[j].vector.array - sum(proj(xi[i], v[j]) for i in range(j))
            return xi
        
        xi = ortho(v)

        # Orthonormalised vector basis
        e = [fem.Function(V) for i in range(len(v))]
        for i,xi_ in enumerate(xi):
            e[i].vector.array = xi_.vector.array/fem.assemble_scalar(fem.form(inner(xi_, xi_)*dx))**0.5
        
        new_fxn = Function(V)
        new_fxn.vector.array  = fxn.vector.array - sum(proj(e_, fxn) for e_ in e)

        if verbose is True:
            print("orthonormalisation test:")
            for i in range(len(xi)):
                for j in range(i+1):
                    print(f"inner(e[{i}], e[{j}])*dx {fem.assemble_scalar(fem.form(inner(e[i], e[j])*dx))}")

            print(f"u norm {fxn.vector.norm(2)}, u_star norm {new_fxn.vector.norm(2)}")
            print(f"orthogonalisation of u_star with rigid body modes test:")
            for j in range(len(v)):
                print(f"(rbms[{j}], u_star) = {fem.assemble_scalar(fem.form(inner(new_fxn, v[j])*dx))}")

        return new_fxn



class CoupledCrossSection:
    '''class containing methods for gluing multiple overlapping, nonmatching meshes to 
        compute combined beam cross-sectional properties'''
    def __init__(self,XSs,pen=1e2):
        
        #assign cross-sections objects to regions 
        self.XSs = XSs
        self.regions = {i:Region(XS.msh,fxn_space=XS.V) for i,XS in enumerate(XSs)} 
        self.meshes = {i:XS.msh for i,XS in enumerate(XSs)}
        self.num_meshes = len(self.meshes)
        
        #base penalty parameter
        self.pen = pen

        #adjust penalty based on average mesh size
        self._set_penalty_values()

        #compute collisions between all meshes
        self._find_overlap()

        # #construct mortar meshes
        # self._construct_mortar_meshes()

    def _set_penalty_values(self):
        h_avg_list = []
        for XS in self.XSs:
            h_expr = ufl.CellDiameter(XS.msh)
            h_avg = fem.assemble_scalar(fem.form(h_expr*XS.dx))
            print(f'average cell size: {h_avg}')
            h_avg_list.append(h_avg)
        self.nu_u = self.pen / np.average(h_avg_list)**2
        self.nu_t = 1
        return
    
    def get_xs_stiffness_matrix(self):
        self._construct_coupling()

        #construct each region's system
        self._construct_system_forms()

        self._organize_system_forms()

        self._get_system_sizes()

        self._get_system_matrices()

        #apply the penalty terms
        self._apply_coupling()

        #construct block system
        self._construct_block_system()

        #solve for the warping functions
        # self._solve_block_system()

        
        # #map elastic solutions to construct warping functions
        # self._compute_xs_stiffness_matrix(correction=correction)
        
        return
    
    def _organize_system_forms(self):
        #construct the forms, sizes and matrices
        system_forms = []
        for i in range(self.num_meshes):
            system_forms_row = []
            for j in range(self.num_meshes+1):
                #diagonal block
                if i==j:
                    system_forms_row.append(self.XSs[i].a00)
                #contstrain column
                elif j==self.num_meshes:
                    system_forms_row.append(self.XSs[i].a_form[0][1])
                #off-diagonal
                else:
                    system_forms_row.append(None)  
            system_forms.append(system_forms_row)
        #last constraint row:
        constraint_row = [self.XSs[i].a_form[1][0] for i in range(self.num_meshes)]
        constraint_row.append(None)
        system_forms.append(constraint_row)

        self.system_LHS_forms = system_forms
        self.system_RHS_forms = [xs.L_form[0] for xs in self.XSs]
        self.system_RHS_forms.append(None)
    
    def _get_system_sizes(self):
        
        system_sizes = []
        for xs in self.XSs:
            size = xs.V.dofmap.index_map.size_global * xs.V.dofmap.index_map_bs
            system_sizes.append(size)
        size_lm = self.XSs[0].LM.dofmap.index_map.size_global * self.XSs[0].LM.dofmap.index_map_bs
        system_sizes.append(size_lm)
        self.system_size_list= system_sizes
        self.system_sizes = [[(size_i,size_j) for size_j in system_sizes] for size_i in system_sizes]

    
    def _get_system_matrices(self):
        system_matrices = []
        for idx_i,system_forms_i in enumerate(self.system_LHS_forms):
            system_matrices_i = []
            for idx_j,system_form in enumerate(system_forms_i):
                if system_form is not None:
                    system_matrix = fem.petsc.assemble_matrix(fem.form(system_form))
                else:
                    system_matrix = PETSc.Mat().createAIJ(self.system_sizes[idx_i][idx_j])
                system_matrix.assemble()
                system_matrices_i.append(system_matrix)
            system_matrices.append(system_matrices_i)
        
        self.system_matrices = system_matrices


    def _find_overlap(self):
        '''
        Construct collision objects 
        '''
        bb_trees = get_bbtrees(list(self.meshes.values()))
        self.bb_trees = {i:bbtree for i,bbtree in zip(self.meshes.keys(),bb_trees)}

        #compute all collisions
        # TODO: some collision detection computational time can be saved by avoiding 
        # the collision detection on the inverse mesh combination with a non-overlapping section 
        # 
        # TODO: JJK 2/14/25 Need to add the computation of the collision area estimate at this stage,
        #           as well as add the measure corresponding to the collision celltags      
        collisions = {}
        separations = {}
        adjacency = np.zeros((self.num_meshes,self.num_meshes),dtype=int)
        for i in range(self.num_meshes):
            # collisions_i = {}
            # separations_i = {}
            for j in range(i,self.num_meshes):
                #TODO: need to ENHANCE with self-interesection capability
                if i==j:
                    continue

                #get collisions 
                bb_tree_collisions = geometry.compute_collisions_trees(self.bb_trees[i], self.bb_trees[j])
                
                if bb_tree_collisions.size != 0:
                    #update the adjaceny matrix
                    adjacency[i,j] = 1
                    
                    #nodal points of geometry
                    meshptsi = self.meshes[i].geometry.x
                    meshptsj = self.meshes[j].geometry.x

                    # get the bounding box trees for the overlap between mesh i and j
                    bbleaves_ij = geometry.compute_collisions_points(self.bb_trees[j],meshptsi)
                    bbleaves_ji = geometry.compute_collisions_points(self.bb_trees[i],meshptsj)

                    # Get the adjacency list of the cells in mesh i that collide with mesh points j
                    adj_list_ij = geometry.compute_colliding_cells(self.regions[j].msh, bbleaves_ij, meshptsi)
                    adj_list_ji = geometry.compute_colliding_cells(self.regions[i].msh, bbleaves_ji, meshptsj)

                    # find the points of mesh i that are contained within the bounds of mesh j
                    pts_i = [i for i in range(len(meshptsi)) if len(adj_list_ij.links(i)) > 0]
                    pts_j = [j for j in range(len(meshptsj)) if len(adj_list_ji.links(j)) > 0]
                    # #find the points that interpolate onto mesh i
                    # cells_j = np.unique(bbleaves_ij.array)
                    # pts_candidates = []
                    # for cell in cells_j:
                    #     pts_candidates.append(self.meshes[j].topology.connectivity(2,0).links(cell)) 
                    # pts_j = np.unique(pts_candidates)

                    #create vertex-to-cell connectivity has been created if it hasn't been done yet
                    # if self.regions[i].msh.topology.connectivity(0,2) is None:
                    self.regions[i].msh.topology.create_connectivity(0,2)
                    # if self.regions[j].msh.topology.connectivity(0,2) is None:
                    self.regions[j].msh.topology.create_connectivity(0,2)
                    
                    # #get the penalty dofs:
                    # penalty_dofs_i=fem.locate_dofs_topological(self.regions[i].fxn_space,0,pts_i)
                    # penalty_dofs_j=fem.locate_dofs_topological(self.regions[j].fxn_space,0,pts_j)
                    
                    #tag cells based on whether they are in the overlap (1),
                    #    on the boundary(2) or outside the overlap (0)
                    celltags_i,celltags_j = get_collision_celltags(self.meshes[i],
                                                                   self.meshes[j],
                                                                   bb_tree_collisions,
                                                                   partial=True,
                                                                   pts=(pts_i,pts_j))

                    #information about a collision of mesh i on mesh j
                    collision_ij = Collision(bb_tree_collisions,
                                             (i,j),
                                             (celltags_i,celltags_j),
                                             (pts_i,pts_j))#,
                                             #(penalty_dofs_i,penalty_dofs_j))

                    # pen_vec = self._build_penalty_vector(self.regions[i],collision_ij)
                    # collision_ij.add_pen_vec(pen_vec)
                    
                    #TODO: move this to a different location
                    #modify each region's material properties based on the effective material rule 
                    self._adjust_material(collision_ij)

                    collisions[(i,j)]=collision_ij

                # elif collisions_bbtree_ij.size == 0:
                #     separations_i[j]=Separation()
                    
            # #add all collisions to dictionary list
            # if collisions_i:
            #     collisions[i] = collisions_i
            # if separations_i:
            #     separations[i] = separations_i

        self.collisions = collisions
        self.separations = separations
        self.adjacency = adjacency
    
    
    def _construct_coupling(self):
        for collision in self.collisions:
            mshA = self.meshes[collision[0]]
            mshB = self.meshes[collision[1]]
            
            #constructing mortar mesh:
            tags_A,tags_B=self.collisions[collision].celltags

            bndry_facets_A = get_overlap_boundary_facets(mshA,tags_A)
            bndry_facets_B = get_overlap_boundary_facets(mshB,tags_B)

            facet_tags_A = meshtags(mshA,mshA.topology.dim-1,bndry_facets_A,np.ones_like(bndry_facets_A))
            facet_tags_B = meshtags(mshB,mshB.topology.dim-1,bndry_facets_B,np.ones_like(bndry_facets_B))

            poly_C = compute_union_polygon(mshA, facet_tags_A, mshB, facet_tags_B)
            self.collisions[collision].msh = mesh_from_polygon(poly_C)
            mesh_C = self.collisions[collision].msh
        
            #intialize functions on mortar mesh and add to collision
            Ve_C = element("CG",mesh_C.topology.cell_name(),1,shape=(3,))
            self.collisions[collision].fxn_space = fem.functionspace(mesh_C, mixed_element(4*[Ve_C]))
            VC = self.collisions[collision].fxn_space
            
            self.collisions[collision].u = TrialFunction(VC)
            self.collisions[collision].v = TestFunction(VC)
            self.collisions[collision].dx = Measure("dx",domain=mesh_C)
            uC = self.collisions[collision].u
            vC = self.collisions[collision].v
            dx_C = self.collisions[collision].dx

            #construct projection operators
            self.collisions[collision].PA = get_interpolation_matrix(VC,self.XSs[collision[0]].V,mixed=True)
            self.collisions[collision].PB = get_interpolation_matrix(VC,self.XSs[collision[1]].V,mixed=True)

            #construct displacement term (penalty weighted mass matrix)
            MC_form = self.nu_u * inner(uC, vC) * dx_C
            MC = fem.petsc.assemble_matrix(fem.form(MC_form))
            MC.assemble()
            self.collisions[collision].MC_form = MC_form
            self.collisions[collision].MC = MC

            #construct traction term (penalty weighted traction matrix)
            n = FacetNormal(mesh_C)
            n3 = as_tensor([0,n[0],n[1]])

            #DG0 space, used for material properties, etc
            Q = fem.functionspace(mesh_C,('DG',0))
            #construct DG spaces for modulus of elasticity and poisson ratio (assuming all materials are ISOTROPIC)
            E = fem.Function(Q)
            nu = fem.Function(Q)
            E.x.array[:] = np.full_like(E.x.array,self.XSs[0].materials[0].E,dtype=default_scalar_type)
            nu.x.array[:] = np.full_like(nu.x.array,self.XSs[0].materials[0].nu,dtype=default_scalar_type)
            C_C = getMatConstitutiveIsotropic(mesh_C,E,nu)
            i,j,k,l = indices(4)

            #trial function strain/stress:
            eps_C = self.XSs[0].warping2strain(uC,0)
            sigma_c =  as_tensor(C_C[i,j,k,l]*eps_C[k,l],(i,j))

            #test function strain/stress:
            eps_vC = self.XSs[0].warping2strain(vC,0)
            sigma_vc =  as_tensor(C_C[i,j,k,l]*eps_vC[k,l],(i,j))

            #traction stiffness matrix:
            S_C_form = self.nu_t * dot(dot(sigma_c,n3),dot(sigma_vc,n3))*ds
            S_C = fem.petsc.assemble_matrix(fem.form(S_C_form))
            S_C.assemble()
            self.collisions[collision].SC_form = S_C_form
            self.collisions[collision].S_C = S_C

            #construct displacement penalty terms
            PA = self.collisions[collision].PA
            PB = self.collisions[collision].PB
            S_AA = AT_C_B(PA, MC, PA)
            S_AB = AT_C_B(PA, MC, PB)
            S_BA = AT_C_B(PB, MC, PA)
            S_BB = AT_C_B(PB, MC, PB)

            #add traction term to the penalty terms:
            S_AA.axpy(1.0, AT_C_B(PA, S_C, PA) )
            S_AB.axpy(1.0, AT_C_B(PA, S_C, PB) )
            S_BA.axpy(1.0, AT_C_B(PB, S_C, PA) )
            S_BB.axpy(1.0, AT_C_B(PB, S_C, PB) )

            self.collisions[collision].Sij = [[S_AA,S_AB],
                                              [S_BA,S_BB]]
        
        return
    
    def _apply_coupling(self):
        for enum_idx, (msh_indices,collision) in enumerate(self.collisions.items()):
            for idx_i in msh_indices:
                for idx_j in msh_indices:
                    if idx_i == idx_j:
                        scale = 1.0
                    else:
                        scale = -1.0
                    #add coupling term to system matrices
                    self.system_matrices[idx_i][idx_j].axpy(scale,self.collisions[msh_indices].Sij[idx_i][idx_j])
    

    def _construct_block_system(self):
        #Set up the full block system
        self.system_mat = PETSc.Mat()
        self.system_mat.createNest(self.system_matrices)

        # set up the solver with the LHS
        self.solver = PETSc.KSP().create(self.meshes[0].comm)
        self.solver.setOperators(self.system_mat)
        self.solver.setType("preonly")
        pc = self.solver.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        


        return


    # def _initialize_mortar_mesh_fxns(self):
        
    #     return
    
    # def _get_projection_operators(self):
    #     return
    

    # def _construct_disp_term(self):
    #     return
    
    # def _construct_traction_term(self):
    #     return
    
    # def _assemble_coupled_system(self):
    #     return
    
    def _solve_coupled_system(self):
        #create functions for solution for each region
        for xs_num,xs in self.XSs:
            #populate the warping function and the lagrange multiplier vectors
            xs.uh = fem.Function(xs.V, name="u_"+str(xs_num))
            xs.lmbdah= fem.Function(xs.LM,name="lmbda_"+str(xs_num))

        #================== solve constrained system for each mode ==================#
        solutions = []
        functions = []
        lmbdas = []
        residuals = []
        L1_list = self.XSs[0].L_form[1] #identical global constraints
        for idx_l,L1 in enumerate(L1_list):
            b1 = fem.petsc.assemble_vector(fem.form(L1))
            self.system_RHS_forms[-1] = b1

            #TODO: Lucky us, no special BCS to apply rn, may change if there were any elastic foundations, etc
            b = PETSc.Vec().createNest(self.system_RHS_forms)

            xh = b.copy()

            #solve the linear systesm
            self.solver.solve(b, xh)
            # xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            #get the local vectors:
            x_local = []
            offset = 0
            for size in self.system_size_list:
                x_local.append(xh.array[offset:offset+size])
                offset += size

            #populate the warping function and the lagrange multiplier vectors
            for xs_num,xs in self.XSs:
                #populate the warping function and the lagrange multiplier vectors
                # xs.uh = fem.Function(xs.V, name="u_"+str(xs_num)+"_"+str(idx_l))
                # xs.lmbdah= fem.Function(xs.LM,name="lmbda_"+str(xs_num)+"_"+str(idx_l))

                xs.uh.x.array[: len(x_local[xs_num])] = x_local[xs_num]
                xs.lmbdah.x.array[: len(x_local[-1])] = x_local[-1]

            # uh.x.scatter_forward()
            # lmbdah.x.scatter_forward()

            solutions.append(xh.copy())

            #TODO: turn into a loop:
            functions.append([xs.uh.copy() for xs in self.XSs])
            lmbdas.append([xs.lmbdah.copy() for xs in self.XSs])

            # #TODO: currently, need to do this because we are using a ufl.TestFunction() in the residual construction
            # #       This can be re-written so that uh is used to construct the form, so that we don't have to repeatedly
            # #       re-assemble a00,a10 or a01, just L0 and L1
            # a00_form = squareXS._construct_xs_form(uh,return_form=True)
            # a01_form = inner(lmbdah,constraints(squareXS.v))*dx
            # a10_form = inner(dlmbda, constraints(uh)) * dx

            # #main system residual
            # residual00 = a00_form + a01_form - L0 
            # #lagrange multiplier system residual
            # residual10 = a10_form - L1

            # residuals.append((residual00,residual10))

            # print(f'lagrange multipliers for mode{k}:{x_local[2]}')


        #==================== compute overall stiffness matrix ======================#
        #TODO: loop time!
        #populate individual functions with warping fucntions
        for xs in self.XSs:
            xs.warping_functions = [function[0] for function in functions]
        # TXS_nm.XSs[1].warping_functions = [functionAB[1] for functionAB in functions]


        return
    

    # def _build_penalty_vector(self,region_i,collision_ij):
    #     '''
    #     Build the vector of penalty terms per dof
    #     This is a PETSc Vector that can be directly multiplied by the interpolation matrix
    #     '''
    #     #initialize empty PETSc vector
    #     pen_vec = PETSc.Vec().create()
    #     vec_size = region_i.fxn_space.dofmap.index_map.size_global #* region_i.fxn_space.num_sub_spaces
    #     pen_vec.setSizes(vec_size)
    #     pen_vec.setFromOptions()

    #     #compute areas of each element in the overlapping subdomain using a DG0 space
    #     DG0 = functionspace(region_i.msh,("DG",0))
    #     v = ufl.TestFunction(DG0)
    #     # dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_id = 1, subdomain_data=collision_ij.celltags)
    #     dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_data=collision_ij.celltags[0])
    #     cell_area_form = form(v*dx_overlap((1,2)))
    #     cell_areas = assemble_vector(cell_area_form)

    #     avg_cell_size = np.sum(cell_areas.array)/cell_areas.array.shape[0]

    #     #create connectivity between cells and vertices (if not already created)
    #     region_i.msh.topology.create_connectivity(0,2)
    #     pen_values = np.zeros((len(collision_ij.penalty_dofs[0]),),dtype=float)
    #     #for each pt, update the penalty value for that vertex
    #     for i,pt in enumerate(collision_ij.pts[0]):
    #         #get the cells connected to the penalty dof
    #         cells = region_i.msh.topology.connectivity(0,2).links(pt)

    #         dofs = fem.locate_dofs_topological(region_i.fxn_space,0,[pt])
            
    #         #add up area of all cells that are incident to the penalty dof
    #         #  adjust penalty proportionately to the supported area
    #         indices=np.where(np.isin(collision_ij.penalty_dofs[0],dofs))
    #         # pen_values[indices] = self.pen * np.sum(cell_areas.array[cells])
    #         pen_values[indices] = self.pen * avg_cell_size*np.ones_like(indices)

    #         #increase the penaly value by 1-2 orders of magnitude for the 
    #         #   out of plane warping displacement
    #         out_of_plane_dofs = [dof for sublist in 
    #                              [list(fem.locate_dofs_topological(region_i.fxn_space.sub(i).sub(0),0,[pt])) 
    #                               for i in range(region_i.fxn_space.num_sub_spaces)] for dof in sublist]
    #         out_of_plane_indices=np.where(np.isin(collision_ij.penalty_dofs[0],out_of_plane_dofs))
    #         # pen_values[out_of_plane_indices] *= self.pen#*self.pen

    #     #populate the PETSc vector with the values at the proper indices
    #     for idx,val in zip(collision_ij.penalty_dofs[0],pen_values):
    #         pen_vec.setValue(idx,val)
        
    #     return pen_vec

    def _adjust_material(self,collision_ij):
        '''
        For a collision, modify the material tensor in the overlapping 
        region based on the effective material policy'''
        #this can be done by using the celltags in the collision object and overwriting the material constitutive tensor:
        XSi = self.XSs[collision_ij.mesh_indices[0]]
        XSj = self.XSs[collision_ij.mesh_indices[1]]

        celltags_i = collision_ij.celltags[0]
        celltags_j = collision_ij.celltags[1]

        for (XS,celltags) in [(XSi,celltags_i),(XSj,celltags_j)]:
            # cells = celltags.find(1)
            cells = np.concatenate([celltags.find(1),celltags.find(2)])
            # XS.E.x.array[cells] *= 1/np.sqrt(2)
            # XS.nu.x.array[cells] *= 1/np.sqrt(2)
            XS.E.x.array[cells] *= 0.5
            # XS.nu.x.array[cells] *= 0.5
            
            XS.C = getMatConstitutiveIsotropic(XS.msh,XS.E,XS.nu)

    def _construct_system_forms(self):
        #construct the residual and assemble the system mat for each region
        for XS in self.XSs:
            XS._construct_xs_form()
            XS._construct_KKT_forms()

        #     #construct RHS form vectors with no body force (e.g. unchanged for each mode)
        #     f0_A= fem.Constant(mesh_A, default_scalar_type([0.0]*12)) 
        #     L0_A = inner(TXS_nm.XSs[0].v, f0_A) * TXS_nm.XSs[0].dx
        #     b0_A = fem.petsc.assemble_vector(fem.form(L0_A))

        # #construct RHS forms for the constraints:
        # f_constraints = []
        # for i in range(6):
        #     f1_np = np.zeros((self.system_sizes[-1][-1][0],))
        #     f1_np[i]= 1.0
        #     f_constraints.append(f1_np)
        # system_RHS_forms_constraint = [inner(fem.Constant(self.meshes[0], default_scalar_type(f1)), self.XSs[0].dlmbda) * self.XSs[0].dx for f1 in f_constraints]

    def _compute_xs_stiffness_matrix(self,correction=None):
        '''
        for each region, get the elastic solution modes and compute the stiffness
        store the accumulated matrices for recovery, etc
        '''
        self.K = np.zeros((6,6))
        self.K1 = np.zeros((6,6))
        self.K2 = np.zeros((6,6))
        # self.S = np.zeros((6,6))

        for i,region in zip(self.regions,self.regions.values()):
            self.XSs[i]._compute_xs_stiffness_matrix()
            self.K1 += self.XSs[i].K1
            self.K2 += self.XSs[i].K2

            # self.S += self.XSs[i].S
            # self.K += self.XSs[i].K
        
        self.K = self.K1.T @ np.linalg.inv(self.K2) @ self.K1
           

    def get_overlap_area(self):
        for idx,val in np.ndenumerate(self.adjacency):
            if val == 0:
                continue
            else:
                dx_overlap = Measure("dx", domain=self.regions[idx[0]].msh, subdomain_data=self.collisions[idx[0]][idx[1]].celltags[1])

                A_plus = fem.assemble_scalar(fem.form(1.0*dx_overlap((1,2))))
                A_minus = fem.assemble_scalar(fem.form(1.0*dx_overlap((1))))
                A_avg = 0.5*(A_plus+A_minus)

                self.collisions[idx[0]][idx[1]].add_overlap_areas(A_plus,A_minus,A_avg)
    
    def plot_meshes(self):
        plot_xdmf_mesh(list(self.meshes.values()),surface=True)

    def plot_warping_fxns(self,fxn_order=0):
        
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()
                
        mode = ['Axial','Shear 1', 'Shear 2', 'Torsion', 'Bending 1', 'Bending 2']
        plotter = pyvista.Plotter(shape=(2,3))
        grids = []
        warped = []
        for i in range(6):
            row = int(i/3)
            col = i%3
            name = f'mode_{i}'
            plotter.subplot(row,col)
            #plot mesh
            solution_modes=[]
            indiv_grids = []
            indiv_warped = []
            for j,xs in enumerate(self.XSs):
                tdim = xs.msh.topology.dim

                elastic_sols = np.zeros((xs.warping_functions[0].sub(0).collapse().x.array.shape[0],6))
                for idx in range(6):
                    elastic_sols[:,idx]=xs.warping_functions[idx].sub(fxn_order).collapse().x.array

                V0,V0_to_V = xs.V.sub(0).collapse()
                topology, cell_types, geom = plot.vtk_mesh(V0)
                indiv_grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))
                
                c = np.zeros((6,1))
                c[i,:] = 1

                warping_sol = elastic_sols@c

                solution_mode = warping_sol.reshape((geom.shape[0], 3))[:,[1,2,0]]
                solution_modes.append(solution_mode)
            
            for j,xs in enumerate(self.XSs):
                # print(np.max(np.linalg.norm(solution_mode,axis=1)))
                scaling_factor = np.max(np.linalg.norm(np.vstack(solution_modes),axis=1))
                print(scaling_factor)
                indiv_grids[j][name]= solution_modes[j]/scaling_factor
                
                indiv_warped.append(indiv_grids[j].warp_by_vector(name,factor=.1))

                plotter.add_mesh(indiv_warped[j],show_edges=True,opacity=.9,scalar_bar_args={'title': 'Norm of Disp. Magnitude'},)
                plotter.add_mesh(indiv_grids[j],show_edges=True,opacity=0.75,style='wireframe',show_scalar_bar=False)
            
            grids.append(indiv_grids)
            warped.append(indiv_warped)
            
            plotter.add_text(mode[i])
        
            plotter.view_isometric()
            plotter.show_bounds(location='outer',
                                show_zlabels=False,
                                n_xlabels=2,
                                n_ylabels=2,
                                n_zlabels=2)
        if not pyvista.OFF_SCREEN:
            plotter.show()

    def plot_warping_strains(self,component=(0,0)):
        
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()
                
        mode = ['Axial','Shear 1', 'Shear 2', 'Torsion', 'Bending 1', 'Bending 2']
        plotter = pyvista.Plotter(shape=(2,3))
        grids = []
        warped = []
        for i in range(6):
            row = int(i/3)
            col = i%3
            name = f'mode_{i}'
            plotter.subplot(row,col)
            #plot mesh
            strains_to_plot=[]
            indiv_grids = []
            indiv_warped = []
            for j,xs in enumerate(self.XSs):
                tdim = xs.msh.topology.dim

                c_np = np.zeros((6,))
                c_np[i] = 1

                c = Constant(xs.msh,PETSc.ScalarType(c_np))

                ubar = dot(xs.N_bar,c)
                uhat = dot(xs.N_hat,c)
                eps_ufl = xs.warping2strain(ubar,uhat)
                #Vstrain is a scalar functionspace for only one strain component 
                Vstrain = functionspace(xs.msh,("DG",0)) 
                # Vstrain = functionspace(self.msh,("DG",0,(self.d,self.d))) 
                strain_component=fem.Expression(eps_ufl[component], Vstrain.element.interpolation_points())
                # strain_to_plot = fem.Function(Vstrain.sub(0).collapse()[0])
                strain_to_plot = fem.Function(Vstrain)
                strain_to_plot.interpolate(strain_component)
                
                V0,V0_to_V = xs.V.sub(0).collapse()
                topology, cell_types, geom = plot.vtk_mesh(V0)
                indiv_grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

                indiv_grids[j][name]= strain_to_plot.vector.array

                strains_to_plot.append(strain_to_plot)
            
            for j,xs in enumerate(self.XSs):
                # # print(np.max(np.linalg.norm(solution_mode,axis=1)))
                # scaling_factor = np.max(np.linalg.norm(np.vstack(solution_modes),axis=1))
                # indiv_grids[j][name]= solution_modes[j]/scaling_factor
                
                # indiv_warped.append(indiv_grids[j].warp_by_vector(name,factor=.1))

                # plotter.add_mesh(indiv_warped[j],show_edges=True,opacity=.9)
                plotter.add_mesh(indiv_grids[j],show_edges=True,opacity=.75,scalar_bar_args={'title': f'warping mode {i}'})
            
            grids.append(indiv_grids)
            # warped.append(indiv_warped)
            
            plotter.add_text(mode[i])
        
            plotter.view_xy()
        plotter.show_bounds()
        if not pyvista.OFF_SCREEN:
            plotter.show()

        #TODO: I believe we can use the block matrix interface here and this will significantly simplify the assembly of this system

        # #compile system matrices for each individual region into a list 
        # #   accessible by the coupled problem class
        # # system_mats = []
        # offset = 0
        # for i,region in zip(self.regions,self.regions.values()):
        #     region.system_mat = self.XSs[i].system_mat
        #     #store offset values for the computed 
        #     region.offset_start = offset
        #     offset += region.system_mat.getSize()[0]
        #     region.offset_end = offset
        # #     system_mats.append(region.system_mat)
        # # self.system_mats = system_mats


    # def _construct_coupled_system_matrix(self):
    #     '''
    #     Given collisions and regions, 
    #     set up the coupled system matrix with the penalty terms
    #     '''

    #     # for each collision, compute the interpolation matrices and add the penalty terms to the corresponding dofs
    #     for idx,val in np.ndenumerate(self.adjacency):
    #         if val == 1:
    #             #get the interpolation matrix
    #             self.collisions[idx[0]][idx[1]].inter_mat = get_interpolation_matrix(self.regions[idx[1]].fxn_space,
    #                                                                                  self.regions[idx[0]].fxn_space,
    #                                                                                  mixed=True)
    #             #copy the interpolation matrix :
    #             self.collisions[idx[0]][idx[1]].pen_mat = self.collisions[idx[0]][idx[1]].inter_mat.duplicate()

    #         elif val == 0 and idx[0] != idx[1]:
    #             self.separations[idx[0]][idx[1]].mat.createAIJ([self.regions[idx[1]].system_mat.getSize()[0],
    #                                                             self.regions[idx[0]].system_mat.getSize()[1]])
    #             self.separations[idx[0]][idx[1]].mat.assemble()

    #     #populate an array of the same size as the adjacency matrix of the petsc matrices
    #     #  using a *nearly* incomprehensible list "comprehension" 
    #     # this adds the unadultered system to the diagonals, the interpolation matrices where there is a collision
    #     # and the assembled empty matrices where there is a "separation"
    #     A_list = [ [self.regions[i].system_mat if i==j
    #                 else self.separations[i][j].mat if self.adjacency[i][j] == 0 and i!=j
    #                 else self.collisions[i][j].pen_mat 
    #                     for i in range(self.num_meshes)]
    #                  for j in range(self.num_meshes) ]
               
    #     # add the penalty to the relevant block of A_list
    #     for idx,val in np.ndenumerate(self.adjacency):
    #         # if idx[0]==idx[1]:
    #         if val==1:
    #             pen_term = PETSc.Mat().createAIJ(A_list[idx[0]][idx[0]].getSize())
    #             pen_term.assemble()
    #             pen_term.setDiagonal(self.collisions[idx[0]][idx[1]].pen_vec)
    #             pen_term.assemble()
                
    #             #add penalty term to diagonal block
    #             A_list[idx[0]][idx[0]].axpy(1.0,pen_term)
    #             # if idx[0]<idx[1]:
    #             #     A_list[idx[0]][idx[0]].axpy(1.0,pen_term)
    #             # elif idx[0]>idx[1]:
    #             #     A_list[idx[0]][idx[0]].axpy(-1.0,pen_term)

    #             #TODO: need to come up with a better way of populating the 
    #             #   nested list than simply filling with the interpolation matrix, then overwriting it...

    #             #add penalty term to off diagonal block (overwriting the )     
    #             A_list[idx[0]][idx[1]] = pen_term.matMult(self.collisions[idx[1]][idx[0]].inter_mat)
    #             A_list[idx[0]][idx[1]].assemble()
    #             A_list[idx[0]][idx[1]].scale(-1.0)
    #             # if idx[0]<idx[1]:
    #             #     A_list[idx[0]][idx[1]].scale(-1.0)
    #             # elif idx[0]>idx[1]:
    #             #     A_list[idx[0]][idx[1]].scale(1.0)

    #             # # TODO: this correction modifies the warping function discovery, which
    #             # #       does NOT modify the discovered stiffness matrix properly
    #             # #apply the correction for the overlap
    #             # dx_correction = ufl.Measure("dx", 
    #             #                             domain=self.XSs[idx[0]].msh,
    #             #                             subdomain_data= self.collisions[idx[0]][idx[1]].celltags[0])
    #             # res = self.XSs[idx[0]]._construct_residual(dx=dx_correction,
    #             #                                            return_residual=True)
    #             # correction = self.XSs[idx[0]]._assemble_system_matrix(residual=res)
    #             # # correction.view()
    #             # A_list[idx[0]][idx[0]].axpy(-0.5,correction)


    #     A = PETSc.Mat()
    #     A.createNest(A_list)
        
    #     A.assemble()

    #     self.system_mat = A


    # def _get_modes(self):
    #     m,n1=self.system_mat.getSize()
    #     print('Computing QR factorization')
    #     A_aij = self.system_mat.convert('aij')
    #     Acsr = csr_matrix(A_aij.getValuesCSR()[::-1], shape=self.system_mat.size)
        
    #     #perform QR factorization and store as struct in householder form
    #     QR= sparseqr.qr_factorize( Acsr.transpose() )

    #     #build matrix of unit vectors for selecting last 12 columns
    #     X = np.zeros((m,12))
    #     for i in range(12):
    #         X[m-1-i,11-i]=1

    #     #perform matrix multiplication implicitly to construct orthogonal nullspace basis
    #     self.sols = sparseqr.qmult(QR,X)
    #     self.sparse_sols = sparseify(self.sols,sparse_format='csc')


    # def _decouple_modes(self):
    #     ''' 
    #     for each region, decouple the modes corresponding to that region
    #     '''
    #     #intialize empty basis transformation matrix
    #     self.basis_trans_matrix = np.zeros((6,12))

    #     #compute contribution to basis transformation matrix for each region
    #     for i,region in zip(self.regions,self.regions.values()):
    #         self.XSs[i].sols = self.sols[region.offset_start:region.offset_end,:]
    #         self.XSs[i]._decouple_modes(basis_matrix_only=True)
    #         print(f"Condition number for sub mesh {i}: {np.linalg.cond(self.XSs[i].mat)}")
    #         self.basis_trans_matrix += self.XSs[i].mat
    #     #perform the basis transformation (use the sparse matrix to prevent numerical inaccuracies during inversion)
    #     # self.sols_decoup = self.sols@np.linalg.inv(self.basis_trans_matrix)
        
    #     print(f"Condition number for overall system: {np.linalg.cond(self.basis_trans_matrix)}")
    #     self.basis_trans_matrix_sparse = sparseify(self.basis_trans_matrix)#,sparse_format='csc')
        
    #     self.basis_trans_matrix_pinv = sparseify(np.linalg.pinv(self.basis_trans_matrix_sparse.toarray()))

    #     self.sols_decoup = (self.sparse_sols.dot(self.basis_trans_matrix_pinv)).toarray()

    #     #get the decoupled basis
    #     for i,region in zip(self.regions,self.regions.values()):
    #         # ubar_uhat_dofs = np.concatenate([self.XSs[i].ubar_vtx_to_dof,self.XSs[i].uhat_vtx_to_dof])
    #         # self.XSs[i].sols_decoup = self.sols_decoup[region.offset_start:region.offset_end,:][ubar_uhat_dofs,:]
    #         self.XSs[i].sols_decoup = self.sols_decoup[region.offset_start:region.offset_end,:]


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
            EI2 = self.E*(((self.h-2*self.t_h)*self.t_w**3 /12 ) + 2*(self.t_h*self.w**3 / 12))
            self.K =  np.diag(np.array([EA,kGA1,kGA2,GJ,EI1,EI2]))            
        
        else:
            print('busy doing nothing')
        