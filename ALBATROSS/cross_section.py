from ufl import (Argument,derivative,dot,ds,cross,Identity,sqrt,inner,tr,variable,
                 diff,grad,sin,cos,as_matrix,SpatialCoordinate,FacetNormal,
                 Measure,as_tensor,indices,as_vector,sym,
                 TrialFunction,TestFunction,split)
from basix.ufl import element,mixed_element
from dolfinx.fem import (Constant,Expression,assemble_scalar,form,Function,
                         functionspace,assemble_vector,petsc)
from dolfinx import fem,plot,geometry,mesh
import numpy as np
from petsc4py import PETSc
from scipy.sparse.linalg import inv,lsqr,spsolve
# import sparseqr
from scipy.sparse import csr_matrix
import ufl 
import pyvista
from scifem import create_real_functionspace
from dolfinx.cpp.la.petsc import get_local_vectors

from ALBATROSS.material import getMatConstitutiveIsotropic
from ALBATROSS.utils import plot_xdmf_mesh,get_vtx_to_dofs,sparseify,order_boundary_nodes
from ALBATROSS.nonmatching_utils import (Region,Separation,Collision,MortarMesh,
                                         get_bbtrees,get_collision_celltags,
                                         get_overlap_boundary_facets,
                                         compute_union_polygon,
                                         mesh_from_polygon,
                                         pts_to_dofs,get_petsc_system,
                                         celltags_to_dofs,
                                         get_interpolation_matrix,
                                         get_points_from_cells)
from ALBATROSS.petsc_utils import convert_petsc_to_numpy,AT_C_B,sparse_mat_from_loflofvec
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

        #number of warping functions
        self.n_w = 6
        
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
        self.V_x = functionspace(self.msh,("CG",self.degree,(self.tdim,)))
        #TODO: argument number is hard coded, may need to adjust based on form 
        self.dX = Argument(self.V_x,2) #direction for spatial derivative, 
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
        
        #compute centroid coordinates 
        self.yavg = assemble_scalar(form(self.x[0]*self.dx))/self.A
        self.zavg = assemble_scalar(form(self.x[1]*self.dx))/self.A

        #vectorfunctionspace for displacement functions
        self.V_u = functionspace(self.msh,('CG',self.degree,(self.d,)))

        #function space for stress output:
        self.V_sigma = functionspace(self.msh,('DG',0,(self.d,self.d)))
        self.V_vm = functionspace(self.msh, ("DG", 0))

        #label nodes and provide dofs to xy mapping:
        self.all_nodes = mesh.locate_entities(self.msh,0,lambda x: np.ones_like(x[0]))
        self.boundary_nodes = mesh.locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
        self.interior_nodes = self.all_nodes[~np.isin(self.all_nodes, self.boundary_nodes)]
        self.dofs_x_boundary = fem.locate_dofs_topological(self.V_x.sub(0),0,self.boundary_nodes)
        self.dofs_y_boundary = fem.locate_dofs_topological(self.V_x.sub(1),0,self.boundary_nodes)
        self.dofs_x_interior = fem.locate_dofs_topological(self.V_x.sub(0),0,self.interior_nodes)
        self.dofs_y_interior = fem.locate_dofs_topological(self.V_x.sub(1),0,self.interior_nodes)
        self.dofs_boundary = np.sort(np.concatenate([self.dofs_x_boundary,self.dofs_y_boundary]))
        self.dofs_interior = np.sort(np.concatenate([self.dofs_x_interior,self.dofs_y_interior]))
    
        #order the boundary using a nearest neighbor search:
        self.boundary_ordering = order_boundary_nodes(self.msh.geometry.x[self.boundary_nodes,0:2])
        self.ordered_nodes = self.boundary_nodes[self.boundary_ordering]
        self.inverse_boundary_ordering = np.argsort(self.boundary_ordering)

        #initialize warping displacement fxn space
        self._set_up_fxnspace_and_fxns()

    def _get_warping_functions(self):
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

        # #set up warping functions:
        # self._set_up_warping_functions()
        
        #set up KSP solver
        if self.verbose:
            print('Computing warping functions....')
        self._compile_forms()
        self._set_up_solver()
        self._solve_system() 

    def get_xs_stiffness_matrix(self):
        if self.verbose:
            print('Computing warping solution....')
        self._get_warping_functions()
        
        if self.verbose:
            print('Computing Beam Constitutive Matrix....')
        self._compute_xs_stiffness_matrix()

        print("DONE computing Beam Constitutive Matrix") 

    def _set_up_fxnspace_and_fxns(self):
        # Construct warping function mixed function space
        e_w = element("CG",self.msh.topology.cell_name(),self.degree,shape=(self.d,))
        self.V_w = functionspace(self.msh, mixed_element(4*[e_w]))
        
        #displacement and test functions
        self.u = Function(self.V_w)
        self.v = TestFunction(self.V_w)
        self.du = TrialFunction(self.V_w)

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
        self.V_lm = create_real_functionspace(self.msh, value_shape=(self.num_constraints,))

        #lagrange multipliers
        self.lmbda = Function(self.V_lm)
        self.mu = TestFunction(self.V_lm)
        self.dlmbda = TrialFunction(self.V_lm)

        #get maps from block vectors ---> warping function & lagrange multiplier vectors
        self.maps = [(self.V_w.dofmap.index_map, self.V_w.dofmap.index_map_bs), (self.V_lm.dofmap.index_map, self.V_lm.dofmap.index_map_bs)]
        
        #set up the warping functions:
        self.warping_functions = []
        self.lmbdas = []
        for i in range(6):
            self.warping_functions.append(self.u.copy())
            self.lmbdas.append(self.lmbda.copy())

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
        self.F00 = eq1+eq2+eq3+eq4

        #get the stiffness matrix form:
        a00 = ufl.derivative(self.F00,self.u,self.du)

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
        F00 = self.F00
        a00 = self.a00

        #construct constraint forms
        F01 = self._construct_constraint_form(lmbda,self.v)
        F10 = self._construct_constraint_form(self.mu,u)
        a01 = ufl.derivative(F01,self.lmbda,self.dlmbda)
        a10 = ufl.derivative(F10,self.u,self.du)

        F =  [[F00,F01],[F10,None]]
        a = [[a00, a01], [a10, None]]

        #construct RHS form with no body force (e.g. unchanged for each mode)
        f0 = fem.Constant(self.msh, default_scalar_type([0.0]*12)) 
        L0 = inner(self.v, f0) * self.dx

        #assemble the constraint RHS: (for each mode solve,the corresponding i entry is = 1)
        self.f1 = fem.Constant(self.msh, default_scalar_type(np.zeros((self.num_constraints,))))
        L1 = inner(self.f1, self.mu) * self.dx

        #since we have different RHS's, return the list of L1's i
        L = [L0,L1]

        self.F = F
        self.a_form = a
        self.L_form = L

    def _assemble_block(self,block=[0,0]):
        '''
        return the assembled petsc mat
        '''
        mat = fem.petsc.assemble_matrix(fem.form(self.a_form[block[0]][block[1]]))
        mat.assemble()
        
        return mat
    
    
    def _return_rhs_vec(self,mode=0):
        self.f1.value = 0
        self.f1.value[mode] = 1
        petsc_vec = fem.petsc.assemble_vector(fem.form(self.L_form[1]))
        return petsc_vec.array
    
    def _compile_forms(self):
        #assemble matrix and vector
        # pRk/puk is the system stiffness matrix 
        self.pRkpuk_form = fem.form(self.a_form)
        self.pRkpuk = fem.petsc.assemble_matrix_block(self.pRkpuk_form)
        self.pRkpuk.assemble()

        self.rhs_form = fem.form(self.L_form)

        self.residual_PDE = self.F[0][0] + self.F[0][1] - self.L_form[0]
        self.residual_constraint = self.F[1][0] - self.L_form[1]

    def _set_up_solver(self):
        # set up the solver with the LHS
        ksp = PETSc.KSP().create(self.msh.comm)
        ksp.setOperators(self.pRkpuk)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        
        self.solver = ksp

    def _solve_system(self):
        # self.solution_vectors= []
        # self.warping_functions = []
        # self.lmbdas = []
        # self.residuals = []
        bcs = []
        for idx_k in range(6):
            self.f1.value = 0           #zero out constraing mode rhs
            self.f1.value[idx_k] = 1    #select mode 

            b = fem.petsc.assemble_vector_block(self.rhs_form, self.pRkpuk_form, bcs=bcs)
            xh = fem.petsc.create_vector_block(self.rhs_form)

            #solve the linear systesm
            self.solver.solve(b, xh)
            xh.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            #populate the warping function and the lagrange multiplier vectors
            x_local = get_local_vectors(xh, self.maps)
            self.u.x.array[: len(x_local[0])] = x_local[0]
            self.lmbda.x.array[: len(x_local[1])] = x_local[1]

            self.u.x.scatter_forward()
            self.lmbda.x.scatter_forward()

            #save copies of the warping function state
            self.warping_functions[idx_k].x.array[:] = self.u.x.array
            self.lmbdas[idx_k].x.array[:] = self.lmbda.x.array

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
        self.A_form = 1.0*dx
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
        
        #apply threshholding:
        s1 = np.max(np.abs(np.diag(self.K1)))
        s2 = np.max(np.abs(np.diag(self.K2)))
        eta = 1e-8
        mask_1 = (np.abs(self.K1) < s1*eta)
        mask_2 = (np.abs(self.K2) < s2*eta)
        np.fill_diagonal(mask_1,False)
        np.fill_diagonal(mask_2,False)
        self.K1[mask_1] = 0.0
        self.K2[mask_2] = 0.0 
        
        #update cross-sectional area:
        self.A = assemble_scalar(fem.form(self.A_form))

        #update centroid coordinates:
        self.yavg = assemble_scalar(form(self.x[0]*self.dx))/self.A
        self.zavg = assemble_scalar(form(self.x[1]*self.dx))/self.A

        #TODO: for multi-material, need smarter update
        self.linear_density = self.A*self.materials[0].density

        #store K1^-1 for recovery and sensitivity computation
        self.K1inv = np.linalg.inv(self.K1)

        #stor K2^-1 for sensitivity computation
        self.K2inv = np.linalg.inv(self.K2)
        
        #compute Flexibility matrix
        self.S = self.K1inv.T@self.K2@self.K1inv

        #compute Beam Stiffness Matrix
        self.K =  self.K1@self.K2inv@self.K1.T
        # self.K =  (self.A**2)*self.K2inv

    
    def rigid_constraints(self,u):
        '''constraints on averages generalized stresses (forces + moments)'''
        ubar,uhat,_,_ = split(u)
        x1,x2 = self.x[0],self.x[1]

        ubar_r = cross(as_vector([0,x1,x2]),ubar)
        
        U = [ ubar[0],      # translation x
            ubar[1],        # translation y
            ubar[2],        # translation z
            ubar_r[0],      # rotation about x
            ubar_r[1],      # rotation about y #NOTE: THIS IS a RIGID rotation, 
            ubar_r[2]      # rotation about z
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

    def _compute_mesh_size(self):
        h_expr = ufl.CellDiameter(self.msh)
        self.mesh_size = fem.assemble_scalar(fem.form(h_expr*self.dx))/self.A
    
    #========== computing derivatives for optimization ===========#
    def apply_inverse_jacobian(self,d_output_w,d_output_l):
        '''
        solve the 6 systems for the action of the warping functions on the residual
        '''
        self.pRkpuk.assemble()
        d_residuals = self.pRkpuk.createVecLeft()
        d_residuals.setUp()
        d_outputs = self.pRkpuk.createVecRight()
        d_outputs.setUp()
        d_outputs_len=d_output_w.shape[0]

        d_residuals_w = np.zeros_like(d_output_w)
        d_residuals_lmbda = np.zeros_like(d_output_l)
        for idx in range(d_output_w.shape[1]):
            with d_outputs.localForm() as rhs_local:
                rhs_local.set(0.0)
                rhs_local[:d_outputs_len] = d_output_w[:,idx]
                rhs_local[d_outputs_len:] = d_output_l[:,idx]
            # with d_residuals.localForm() as lhs_local:
            #     lhs_local.set(0.0)

            self.solver.solveTranspose(d_outputs,d_residuals)
            # self.solver.solve(d_outputs,d_residuals)
            d_residuals_w[:,idx]= d_residuals.array[:d_outputs_len]
            d_residuals_lmbda[:,idx]= d_residuals.array[d_outputs_len:]

        return d_residuals_w,d_residuals_lmbda


    def compute_VJP(self,d_residuals_w,d_residuals_lmbda):
        '''
        d_residual_w shape : num_dofs x 6
        d_residual_l shape : num_lms x 6

        dRdx_dr = num_nodes
        '''
        #set up input vector sizes
        d_residuals_w_vec_size = d_residuals_w.shape[0]
        d_residuals_w_vec = PETSc.Vec().createSeq(d_residuals_w_vec_size, comm=PETSc.COMM_SELF)
        
        d_residuals_lmbda_vec_size = d_residuals_lmbda.shape[0]
        d_residuals_lmbda_vec = PETSc.Vec().createSeq(d_residuals_lmbda_vec_size, comm=PETSc.COMM_SELF)
        
        #set up output vector sizes
        d_inputs_vec_size = self.V_x.dofmap.index_map_bs*self.V_x.dofmap.index_map.size_global
        d_inputs_vec = PETSc.Vec().createSeq(d_inputs_vec_size, comm=PETSc.COMM_SELF)
        
        dRdx_dr = np.zeros(d_inputs_vec_size)
        
        #TODO: these really need to be re-formulated to compute actions, not full vec-mat products
        # this is actually pretty straightfoward using UFL when you get around to it

        #this looks something like:
        
        
        d_residual_w_func = fem.Function(self.V_w)
        d_residual_l_func = fem.Function(self.V_lm)
        for idx in range(d_residuals_w.shape[1]):
            #update rhs vector value
            self.f1.value = 0
            self.f1.value[idx]=1

            #select warping function value for mode
            self.u.x.array[:] = self.warping_functions[idx].x.array

            d_residual_w_func.x.array[:] = d_residuals_w[:,idx]
            dinputs_w_test = fem.petsc.assemble_vector(fem.form(ufl.derivative(ufl.action(self.residual_PDE,d_residual_w_func),self.x,self.dX)))
            
            d_residual_l_func.x.array[:] = d_residuals_lmbda[:,idx]
            dinputs_l_test = fem.petsc.assemble_vector(fem.form(ufl.derivative(ufl.action(self.residual_constraint,d_residual_l_func),self.x,self.dX)))

            dRdx_dr += dinputs_w_test.array
            dRdx_dr += dinputs_l_test.array
            # dRwdx = self._compute_spatial_partials(self.residuals[idx][0]) #num_dofs x num_nodes
            
            # d_residuals_w_vec.array = d_residuals_w[:,idx]
            # dRwdx.multTranspose(d_residuals_w_vec,d_inputs_vec) #perform vec-mat product
            # dRdx_dr += d_inputs_vec.array
            
            # dRldx = self._compute_spatial_partials(self.residuals[idx][1] )#num_lms x num_nodes
            # d_residuals_lmbda_vec.array = d_residuals_lmbda[:,idx]
            # dRldx.multTranspose(d_residuals_lmbda_vec,d_inputs_vec) #perform vec-mat product
            # dRdx_dr += d_inputs_vec.array

        return dRdx_dr
    
    def _compile_component_vjp_forms(self):
        print("compiling system component derivatives....")
        self.u_j = fem.Function(self.V_w)
        self.v_j = fem.Function(self.V_w)

        self.l_j = fem.Function(self.V_lm)

        A00_form = self.a_form[0][0]
        A10_form = self.a_form[1][0]

        self.pA00px_form = fem.form(ufl.derivative(ufl.action(ufl.action(A00_form,self.u_j),self.v_j),self.x,self.dX))
        self.pA10px_form = fem.form(ufl.derivative(ufl.action(ufl.action(A10_form,self.u_j),self.l_j),self.x,self.dX))

        print("DONE compiling system component derivatives....")

    def _compute_vjp_dA00dx(self,d_output):
        d_inputs_size = self.V_x.dofmap.index_map_bs * self.V_x.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)

        u_j = self.u_j 
        v_j = self.v_j
        
        for j in np.unique(np.nonzero(d_output)[1]):
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec

            vec = fem.assemble_vector(self.pA00px_form)
            d_input += vec.array

        return d_input


    def _compute_vjp_dA10dx(self,d_output):
        d_inputs_size = self.V_x.dofmap.index_map_bs * self.V_x.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)

        u_j = self.u_j 
        l_j = self.l_j

        for j in np.unique(np.nonzero(d_output)[1]):
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            l_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec

            vec = fem.assemble_vector(self.pA10px_form)
            d_input += vec.array

        return d_input
        
    
    def _compute_vjp_component_spatial(self,form,d_output,test_space ,trial_space = None):
        
        # dFdx = ufl.derivative(form,self.x,self.dX)

        # #create a function and set numpy
        # d_output_func = fem.Function(fxn_space)
        # d_output_func.x.array = d_output

        # vec_form = ufl.action(ufl.adjoint(form),d_output_func)

        # dFdx_dx = fem.petsc.assemble_vector(fem.form(vec_form))

        #TODO: we can probably speed this up by being a bit more intelligent with the function values. 
        # Really... we shouldn't need a loop here at all. 
        # dFormdx = ufl.derivative(form,self.x,self.dX)


        d_inputs_size = self.V_x.dofmap.index_map_bs * self.V_x.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)
        
        if trial_space==None:
            trial_space=test_space
        
        u_j = fem.Function(trial_space)
        v_j = fem.Function(test_space)

        form_cache = fem.form(ufl.derivative(ufl.action(ufl.action(form,u_j),v_j),self.x,self.dX))
        
        # for j in range(d_output.shape[0]):
        # use the nonzero column indices only (instead of all columns regardless of value)                 
        for j in np.nonzero(d_output)[1]:
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec

            # vec = fem.assemble_vector(fem.form(ufl.derivative(ufl.action(ufl.action(form,u_j),v_j),self.x,self.dX)))
            # vec = fem.assemble_vector(fem.form(ufl.action(ufl.action(dFormdx,u_j),v_j)))
            vec = fem.assemble_vector(form_cache)
            d_input += vec.array

        return d_input




    def _compute_spatial_partials(self,form):
        pfpx = fem.petsc.assemble_matrix(fem.form(derivative(form,self.x,self.dX)))
        pfpx.assemble()
        return pfpx
        # return fem.petsc.assemble_vector(fem.form(derivative(form,self.x,self.dX)))
    
    
    def _compute_function_partials(self,Kxij_form,function):
        return fem.petsc.assemble_vector(fem.form(derivative(Kxij_form,function)))


    def compute_pKpx(self):       
        self.pK1px_form = [[derivative(self.K1_form[idx1][idx2],self.x,self.dX)
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
        self.pK2px_form = [[derivative(self.K2_form[idx1][idx2],self.x,self.dX)
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
                
        self.pK1px_lol = [[petsc.assemble_vector(form(self.pK1px_form[idx1][idx2]))
                        for idx2 in range(6)] 
                            for idx1 in range(6)]
        self.pK2px_lol = [[petsc.assemble_vector(form(self.pK2px_form[idx1][idx2]))
                for idx2 in range(6)] 
                    for idx1 in range(6)]

        self.pK1px_sparse = sparse_mat_from_loflofvec(self.pK1px_lol)
        self.pK2px_sparse = sparse_mat_from_loflofvec(self.pK2px_lol)
        
        self.pK1px = self.pK1px_sparse.toarray().reshape((6,6,self.pK1px_sparse.shape[1]))
        self.pK2px = self.pK2px_sparse.toarray().reshape((6,6,self.pK2px_sparse.shape[1]))


        # #boundary dofs ([:,:,self.boundary_dofs])
        # self.boundary_nodes = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))

        #Term 1: (dK1/dx) @ K2inv @ K1^T
        term1 = np.einsum("ijm,jk,kl->ilm", self.pK1px, self.K2inv, self.K1)
        # Term 2: - K1 @ K2inv @ (dK2/dx) @ K2inv @ K1^T
        term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1,self.K2inv,self.pK2px,self.K2inv,self.K1.T)
        # Term 3: K1@ K2inv @ (dK1/dx)^T
        term3 = np.einsum("ij,jk,lkm->ilm", self.K1, self.K2inv, self.pK1px)

        #partial derivatives
        self.pKpx = term1 + term2 + term3 
        
        # #if warping functions are orthogonal, the following simplification holds:
        # self.pApx = petsc.assemble_vector(form(derivative(self.A_form,self.x,self.dX)))
        # term1 = 2*self.A*np.einsum('ij,k->ijk',self.K2inv,self.pApx.array)
        # term2 = -np.einsum("ij,jkl,km->iml", self.K2inv,self.pK2px,self.K2inv)

        # self.pKpx = term1 + self.A**2 * term2

        # #get map from vtx to dofs to restrict to boundary (this only works for CG1)
        # self.boundary_dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.V_x.value_size)
        # indices_to=[]
        # for i in range(self.V_x.num_sub_spaces):
        #     _,map_to = self.V_x.sub(i).collapse()
        #     indices_to.extend(map_to)
        # self.boundary_dof_to_vertex_map = self.boundary_dof_to_vertex_map[np.argsort(indices_to)]

        # #find all the indices where the boundary_node is in the boundary_dof_to_vertex_map and save those indices as a list
        
        # boundary_indices = []
        # for i in self.boundary_nodes:
        #     boundary_indices.extend(list(np.where(self.boundary_dof_to_vertex_map==i)[0]))
        
        # self.pKpx_boundary = self.dKpx[:,:,boundary_indices]

        return self.pKpx.reshape((36,self.pKpx.shape[-1]))
    

    def compute_pKpw(self):
        self.pK1pw_form = [[[derivative(self.K1_form[idx1][idx2],self.warping_functions[idx3])
                                for idx2 in range(6)] 
                                    for idx1 in range(6)]
                            for idx3 in range(6)] 
        self.pK2pw_form = [[[derivative(self.K2_form[idx1][idx2],self.warping_functions[idx3])
                                for idx2 in range(6)] 
                                    for idx1 in range(6)]
                            for idx3 in range(6)] 
          
        self.pK1pw_lol = [[[petsc.assemble_vector(form(self.pK1pw_form[idx3][idx1][idx2]))
                        for idx2 in range(6)] 
                            for idx1 in range(6)]
                            for idx3 in range(6)] 

        self.pK2pw_lol = [[[petsc.assemble_vector(form(self.pK2pw_form[idx3][idx1][idx2]))
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
                                        for idx3 in range(6)] 
        
        #TODO: looks like this doesn't return the derivatives in the same way that 
        # self.pKpw = np.zeros((36,self.warping_functions[0].x.array.shape[0]*6))
        self.pKpw = np.zeros((6,6,self.warping_functions[0].x.array.shape[0],6))
        # self.pKpw_original = np.zeros((6,6,6,self.warping_functions[0].x.array.shape[0]))
        warping_len = self.warping_functions[0].x.array.shape[0]

        # pApw = petsc.assemble_vector(form(derivative(self.A_form,self.warping_functions[0])))
        # term1 = 2*self.A*np.einsum('ij,k->ijk',self.K2inv,pApw.array)
        for idx3 in range(6):
            self.pK1pw_sparse = sparse_mat_from_loflofvec(self.pK1pw_lol[idx3])
            self.pK2pw_sparse = sparse_mat_from_loflofvec(self.pK2pw_lol[idx3])
            
            self.pK1pw = self.pK1pw_sparse.toarray().reshape((6,6,self.pK1pw_sparse.shape[1]))
            self.pK2pw = self.pK2pw_sparse.toarray().reshape((6,6,self.pK2pw_sparse.shape[1]))

            # #boundary dofs ([:,:,self.boundary_dofs])
            # self.boundary_nodes = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
            
            # #TODO: can simplify this
            # #compact einsums:
            # term1 = np.einsum("ijm,ik,kl->jlm", self.pK1pw, self.K2inv, self.K1)
            # term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1.T,self.K2inv,self.pK2pw,self.K2inv,self.K1)
            # term3 = np.einsum("ij,jk,lkm->ilm", self.K1.T, self.K2inv, self.pK1pw)

            #Term 1: (dK1/dw) @ K2inv @ K1^T
            term1 = np.einsum("ijm,jk,kl->ilm", self.pK1pw, self.K2inv, self.K1)
            # Term 2: - K1 @ K2inv @ (dK2/dx) @ K2inv @ K1^T
            term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1,self.K2inv,self.pK2pw,self.K2inv,self.K1.T)
            # Term 3: K1@ K2inv @ (dK1/dx)^T
            term3 = np.einsum("ij,jk,lkm->ilm", self.K1, self.K2inv, self.pK1pw)

            #full sensitivities
            # start = warping_len*idx3
            # stop = warping_len*(idx3+1)
            # self.pKpw[:,start:stop] = (term1 + term2 + term3).T.reshape((warping_len,36)).T
            self.pKpw[:,:,:,idx3] = (term1 + term2 + term3)
            # self.pKpw[:,:,idx3,] = - self.A**2 * np.einsum("ij,jkl,km->iml", self.K2inv,self.pK2pw,self.K2inv)
              
        # #get map from vtx to dofs to restrict to boundary (this only works for CG1)
        # self.boundary_dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.V_x.value_size)
        # indices_to=[]
        # for i in range(self.V_x.num_sub_spaces):
        #     _,map_to = self.V_x.sub(i).collapse()
        #     indices_to.extend(map_to)
        # self.boundary_dof_to_vertex_map = self.boundary_dof_to_vertex_map[np.argsort(indices_to)]

        # #find all the indices where the boundary_node is in the boundary_dof_to_vertex_map and save those indices as a list
        
        # boundary_indices = []
        # for i in self.boundary_nodes:
        #     boundary_indices.extend(list(np.where(self.boundary_dof_to_vertex_map==i)[0]))
        
        # self.pKpx_boundary = self.dKpx[:,:,boundary_indices]

        return self.pKpw.reshape(36,warping_len*6)
    
    def compute_pKpl(self):
        # self.pK1pl_form = [[[derivative(self.K1_form[idx1][idx2],self.lmbdas[idx3])
        #                         for idx2 in range(6)] 
        #                             for idx1 in range(6)]
        #                     for idx3 in range(6)] 
        self.pK2pl_form = [[[derivative(self.K2_form[idx1][idx2],self.lmbdas[idx3])
                                for idx2 in range(6)] 
                                    for idx1 in range(6)]
                            for idx3 in range(6)] 
          
        # self.pK1pl_lol = [[[petsc.assemble_vector(form(self.pK1pl_form[idx3][idx1][idx2]))
        #                 for idx2 in range(6)] 
        #                     for idx1 in range(6)]
        #                     for idx3 in range(6)] 

        self.pK2pl_lol = [[[petsc.assemble_vector(form(self.pK2pl_form[idx3][idx1][idx2]))
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
                                        for idx3 in range(6)] 
        
        self.pKpl = np.zeros((36,self.lmbdas[0].x.array.shape[0]*6))
        lm_len = self.lmbdas[0].x.array.shape[0]
        for idx3 in range(6):
            # self.pK1pl_sparse = sparse_mat_from_loflofvec(self.pK1pl_lol[idx3])
            self.pK2pl_sparse = sparse_mat_from_loflofvec(self.pK2pl_lol[idx3])
            
            # self.pK1pl = self.pK1pl_sparse.toarray().reshape((6,6,self.pK1pl_sparse.shape[1]))
            self.pK2pl = self.pK2pl_sparse.toarray().reshape((6,6,self.pK2pl_sparse.shape[1]))

            # #boundary dofs ([:,:,self.boundary_dofs])
            # self.boundary_nodes = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
            
            # #TODO: can simplify this
            # #compact einsums:
            # term1 = np.einsum("ijm,ik,kl->jlm", self.pK1pl, self.K2inv, self.K1)
            # term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1.T,self.K2inv,self.pK2pl,self.K2inv,self.K1)
            # term3 = np.einsum("ij,jk,lkm->ilm", self.K1.T, self.K2inv, self.pK1pl)
            
            term2 = -np.einsum("ij,jkl,km->iml", self.K2inv,self.pK2pl,self.K2inv)

            #full sensitivities
            start = lm_len*idx3
            stop = lm_len*(idx3+1)
            # self.pKpl[:,start:stop] = (term1 + term2 + term3).reshape((36,lm_len))
            self.pKpl[:,start:stop] = (self.A**2 * term2).reshape((36,lm_len))
               
        # #get map from vtx to dofs to restrict to boundary (this only works for CG1)
        # self.boundary_dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.V_x.value_size)
        # indices_to=[]
        # for i in range(self.V_x.num_sub_spaces):
        #     _,map_to = self.V_x.sub(i).collapse()
        #     indices_to.extend(map_to)
        # self.boundary_dof_to_vertex_map = self.boundary_dof_to_vertex_map[np.argsort(indices_to)]

        # #find all the indices where the boundary_node is in the boundary_dof_to_vertex_map and save those indices as a list
        
        # boundary_indices = []
        # for i in self.boundary_nodes:
        #     boundary_indices.extend(list(np.where(self.boundary_dof_to_vertex_map==i)[0]))
        
        # self.pKpx_boundary = self.dKpx[:,:,boundary_indices]

        return self.pKpl
    
    def compute_pApx(self):
        self.pApx = fem.petsc.assemble_vector(fem.form(derivative(self.A_form,self.x,self.dX)))
        return self.pApx.array
    
    def _set_up_dK_forms(self):
        print('compiling cross-sectional stiffness matrix forms...')
        self.W_1 = fem.Constant(self.msh,np.zeros((6,6)))
        self.W_2 = fem.Constant(self.msh,np.zeros((6,6)))
            
        XS = self
        W_1 = self.W_1
        W_2 = self.W_2

        self.dK_form = 0
        for idx_i in range(6):
            for idx_j in range(6):
                self.dK_form += W_1[idx_i,idx_j] * XS.K1_form[idx_i][idx_j]     
                self.dK_form -= W_2[idx_i,idx_j] * XS.K2_form[idx_i][idx_j]
        
        #spatial partials form
        self.dKdx_form = fem.form(ufl.derivative(self.dK_form,XS.x,XS.dX))

        #partials w.r.t. warping functions and lagrange multipliers
        self.dKdw_form = []
        self.dKdl_form = []
        for idx_k in range(6):
            print('Mode ',idx_k)
            self.dKdw_form.append(fem.form(ufl.derivative(self.dK_form,XS.warping_functions[idx_k])))
            self.dKdl_form.append(fem.form(ufl.derivative(self.dK_form,XS.lmbdas[idx_k])))
        
        print('DONE compiling cross-sectional stiffness matrix forms...')


        
    def _compute_pK_action(self, dK, derivative_type='x'):
        """
        Compute the action of the seed dK on the input based on derivative_type.

        Constitutive map:
            K = K1 @ K2^{-1} @ K1.T
        """
        XS = self
        K1 = self.K1
        K2inv = self.K2inv

        # Correct adjoint weights
        W_1_value = dK @ K1 @ K2inv + K2inv @ K1.T @ dK
        W_2_value = K2inv @ K1.T @ dK @ K1 @ K2inv

        self.W_1.value = W_1_value
        self.W_2.value = W_2_value

        if derivative_type == 'x':
            d_inputs = fem.petsc.assemble_vector(self.dKdx_form)
            return d_inputs.array

        if derivative_type == 'w':
            n_w = XS.V_w.dofmap.index_map_bs * XS.V_w.dofmap.index_map.size_global
            d_inputs = np.zeros((n_w, 6))
            for idx_k in range(6):
                d_inputs[:, idx_k] = fem.petsc.assemble_vector(self.dKdw_form[idx_k]).array
            return d_inputs

        if derivative_type == 'l':
            n_l = XS.V_lm.dofmap.index_map_bs * XS.V_lm.dofmap.index_map.size_global
            d_inputs = np.zeros((n_l, 6))
            for idx_k in range(6):
                d_inputs[:, idx_k] = fem.petsc.assemble_vector(self.dKdl_form[idx_k]).array
            return d_inputs        

    def _set_up_dA_form(self):
        self.dAdx_form = fem.form(ufl.derivative(self.A_form,self.x,self.dX))
    
    
    def _compute_pA_action(self,dA,derivative_type='x'):
        '''
        compute the action of the seed dK on the input based on derivative_type

        return numpy arrays
        '''
        XS = self

        if derivative_type == 'x':
            
            d_inputs = dA*fem.petsc.assemble_vector(self.dAdx_form)
            
            return d_inputs

    #TODO: NEED TO UPDATE WITH ADJOINT SENSITIVITY CODE (REQUIRES FIXES TO RESIDUAL ASSEMBLY)
    def compute_xs_stiffness_matrix_sensitivities(self):
        #TODO: combine EB and TS sensitivities...
        args = self.K1_form[0][0].arguments()
        n = max(a.number() for a in args) if args else -1
        dX = Argument(self.V_x,n+1)
        # n = max(a.number() for a in args) if args else -1
        # du2 = Argument(self.V_x,n+1)
        # du = Argument(self.V_x,0) #there are no arguments in any of these forms?
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

        # #boundary dofs ([:,:,self.boundary_dofs])
        # self.boundary_nodes = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
        
        #TODO: can simplify this
        #compact einsums:
        term1 = np.einsum("ijm,ik,kl->jlm", self.dK1dx, self.K2inv, self.K1)
        term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1.T,self.K2inv,self.dK2dx,self.K2inv,self.K1)
        term3 = np.einsum("ij,jk,lkm->ilm", self.K1.T, self.K2inv, self.dK1dx)

        #full sensitivities
        self.dKdx = term1 + term2 + term3 
               
        #get map from vtx to dofs to restrict to boundary (this only works for CG1)
        self.boundary_dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.V_x.value_size)
        indices_to=[]
        for i in range(self.V_x.num_sub_spaces):
            _,map_to = self.V_x.sub(i).collapse()
            indices_to.extend(map_to)
        self.boundary_dof_to_vertex_map = self.boundary_dof_to_vertex_map[np.argsort(indices_to)]

        #find all the indices where the boundary_node is in the boundary_dof_to_vertex_map and save those indices as a list
        
        boundary_indices = []
        for i in self.boundary_nodes:
            boundary_indices.extend(list(np.where(self.boundary_dof_to_vertex_map==i)[0]))
        
        self.dKdx_boundary = self.dKdx[:,:,boundary_indices]

    def setup_recovery(self):
        """
        Build expressions from ufl and appropriate functionspaces for
        displacement and stress recovery.

        This must be called ONCE after cross-section analysis.
        """

        #TODO: conceptually, these are the same warping coefficients as the ones used for the ufl.varibles
        # warping coefficients (load-dependent, symbolic)
        self.warping_coeffs = fem.Constant(self.msh, np.zeros(self.n_w))

        # --- Warping displacement expression ---
        # u_w(x) = sum_i c_i * phi_i(x)
        self.w_ufl = dot(self.warping_coeffs, as_tensor(self.warping_functions))
        #TODO: mixed element has to be handled differently
        # self.w_expr = Expression(self.w_ufl, self.V_w.element.interpolation_points())

        #NOTE: this uses a frame transformation to translate the beam axis to the X (Z used in beam sectional analysis) 
        # setup the displacement expression (from the ubar warping function)
        self.u_ufl = as_tensor([self.w_ufl[1],
                                 self.w_ufl[2],
                                 self.w_ufl[0]])
        self.u_expr = Expression(self.u_ufl,self.V_u.element.interpolation_points())

        #TODO: need to make sure that the stresses frame is properly translated
        # Strain and stress expressions
        self.eps_ufl = self.warping2strain(self.w_ufl,0)
        self.sigma_ufl = self.warping2stress(self.w_ufl,0)
        self.sigma_expr = Expression(self.sigma_ufl,self.V_sigma.element.interpolation_points())


        s = self.sigma_ufl - 1. / 3 * tr(self.sigma_ufl) * Identity(self.sigma_ufl.ufl_shape[0])
        self.von_Mises_ufl = sqrt(3. / 2 * inner(s, s))
        self.von_Mises_expr = Expression(self.von_Mises_ufl, self.V_vm.element.interpolation_points())

        # # --- Stress projection (L2) ---
        # sigma_trial = ufl.TrialFunction(self.V_sigma)
        # tau = ufl.TestFunction(self.V_sigma)

        # a_sigma = ufl.inner(sigma_trial, tau) * ufl.dx
        # L_sigma = ufl.inner(self.sigma_expr, tau) * ufl.dx

        # self._stress_problem = fem.petsc.LinearProblem(
        #     a_sigma,
        #     L_sigma,
        #     petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        # )

    def coeff_from_reactions(self,reactions:np.ndarray):
        assert(reactions.shape==(6,))

        return self.K1inv@reactions
    
    
    def recover_displacement(self, reactions):
        '''     
        :param reactions: the 6 sectional forces and moments from the 1d solution
        
        :return: function describing sectional displacement
        '''
        self.warping_coeffs.value = self.coeff_from_reactions(reactions)

        u = fem.Function(self.V_u)
        u.interpolate(self.u_expr)
        return u
    

    def recover_stress(self,reactions):
        '''     
        :param reactions: the 6 sectional forces and moments from the 1d solution
        
        :return: function describing sectional displacement
        '''
        self.warping_coeffs.value = self.coeff_from_reactions(reactions)

        sigma = fem.Function(self.V_sigma)
        sigma.interpolate(self.sigma_expr)
        return sigma

    def get_von_mises(self,reactions):
        '''
        Given the sectional reaction forces/moments, return a function with the von Mises stress
        
        inputs: reaction forces/moments
        
        '''
        self.warping_coeffs.value = self.coeff_from_reactions(reactions)

        von_Mises = Function(self.V_vm)
        von_Mises.interpolate(self.von_Mises_expr)
        
        return von_Mises

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
            
            V0,V0_to_V = self.V_w.sub(0).collapse()
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
            
            V0,V0_to_V = self.V_w.sub(0).collapse()
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

class CoupledCrossSection:
    '''class containing methods for gluing multiple overlapping, nonmatching meshes to
        compute combined beam cross-sectional properties'''
    def __init__(self,XSs,pen_u=1e2,pen_t=1,enable_overlap_correction=False):
        
        #assign cross-sections objects to regions 
        self.XSs = XSs
        self.regions = {i:Region(XS.msh,fxn_space=XS.V_w) for i,XS in enumerate(XSs)} 
        self.meshes = {i:XS.msh for i,XS in enumerate(XSs)}
        self.num_meshes = len(self.meshes)
        
        #base penalty parameter
        self.pen_u = pen_u
        self.pen_t = pen_t
        self.enable_overlap_correction = enable_overlap_correction

        #adjust penalty based on average mesh size
        self._set_penalty_values()

        #compute collisions between all meshes
        self._find_overlap()

        #construct mortar meshes
        self._construct_mortar_meshes()

        #construct mortar mesh functionspace and fxns:
        self._construct_mortar_mesh_fxns()

        #prepare the system sizes information:
        self._get_system_sizes()

    #TODO: this whole function could likely be removed and the functionality to set the penalty param values should be in the 
    # construction of the penalty terms
    def _set_penalty_values(self):
        #TODO: this should really be set based on the mortar mesh size, not the foreground mesh size
        #TODO: the mesh size should really be based on the mesh size in the intersection, not the overall average mesh size 
        h_avg_list = []
        for XS in self.XSs:
            XS._compute_mesh_size()
            h_avg = XS.mesh_size 
            print(f'average cell size: {h_avg}')
            h_avg_list.append(h_avg)
        # # use with area disp penalty
        self.nu_u = (self.pen_u * self.XSs[0].materials[0].E) / ((0.5*np.average(h_avg_list))**2)
        # self.nu_u = (self.pen_u * self.XSs[0].materials[0].E) / ((np.average(h_avg_list)))
        # self.nu_u = (self.pen_u  / np.average(h_avg_list) )
        # self.nu_u = (self.pen_u * self.XSs[0].materials[0].E) / np.average(h_avg_list)
        # self.nu_u = (self.pen_u ) / np.average(h_avg_list)**2
        # self.nu_u = (self.pen_u ) / np.average(h_avg_list)
        # self.nu_t = (self.pen_t ) / (np.average(h_avg_list)**2 * self.XSs[0].materials[0].E)
        # self.nu_t = (self.pen_t * np.average(h_avg_list)) / ( self.XSs[0].materials[0].E)
        # self.nu_t = (self.pen_t * self.XSs[0].materials[0].E) / np.average(h_avg_list)
        # # use with area stress penalty
        # self.nu_t = (self.pen_t  * 0.5* np.average(h_avg_list) ) / self.XSs[0].materials[0].E
        # self.nu_t = (self.pen_t  ) / ( self.XSs[0].materials[0].E * 0.5* np.average(h_avg_list))
        # use with traction term (boundary)
        self.nu_t = (self.pen_t ) / ( self.XSs[0].materials[0].E)
        # #use with area strain penalty
        # self.nu_t = self.pen_t * self.XSs[0].materials[0].E 
        # # use with boundary strain penalty
        # self.nu_t = (self.pen_t * self.XSs[0].materials[0].E) * 0.5* np.average(h_avg_list)
        # self.nu_t = (self.pen_t ) / np.average(h_avg_list)

        return
    

    def _build_foreground_systems(self):
        #construct each region's system
        self._construct_system_forms()
        self._organize_system_forms()
        # self._get_system_sizes()
        self._get_system_matrices()
        self._get_system_vectors()


    def _get_warping_functions(self):
        #assemble all the stuff on the foreground meshes and prep the coupled system size
        self._build_foreground_systems()

        #construct the penalty term mass matrices
        self._construct_interpolation_operators()
        self._construct_mortar_forms()
        self._assemble_mortar_matrices()
        # self._construct_overlap_correction()
        self._construct_coupling_terms()

        #apply the penalty terms
        self._apply_coupling()

        #construct block system
        self._set_up_solver()

        #solve for the warping functions
        self._solve_coupled_system()
    

    def get_xs_stiffness_matrix(self):
        #apply the coupling terms, construct the blocked system and solve for warping:
        self._get_warping_functions()
        
        #map elastic solutions to construct warping functions:
        self._compute_xs_stiffness_matrix()
        
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
                #constraint column
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
        self.system_RHS_forms.append(self.XSs[0].L_form[1])
    
    def _get_system_sizes(self):
        
        system_sizes = []
        for xs in self.XSs:
            size = xs.V_w.dofmap.index_map.size_global * xs.V_w.dofmap.index_map_bs
            system_sizes.append(size)
        size_lm = self.XSs[0].V_lm.dofmap.index_map.size_global * self.XSs[0].V_lm.dofmap.index_map_bs
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

    def _get_system_vectors(self):
        self.system_RHS_vectors = []
        for idx_i,system_RHS_form in enumerate(self.system_RHS_forms[:-1]):
            b0i = fem.petsc.assemble_vector(fem.form(system_RHS_form))
            self.system_RHS_vectors.append(b0i)
        self.b1_form_assembled = fem.form(self.system_RHS_forms[-1])
        self.system_RHS_vectors.append(fem.petsc.assemble_vector(self.b1_form_assembled))

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
                    # self._adjust_material(collision_ij)

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
    

    def _construct_mortar_meshes(self):
        for collision in self.collisions:
            mshA = self.meshes[collision[0]]
            mshB = self.meshes[collision[1]]
            
            #constructing mortar mesh:
            tags_A,tags_B=self.collisions[collision].celltags

            bndry_facets_A = get_overlap_boundary_facets(mshA,tags_A)
            bndry_facets_B = get_overlap_boundary_facets(mshB,tags_B)

            facet_tags_A = mesh.meshtags(mshA,mshA.topology.dim-1,bndry_facets_A,np.ones_like(bndry_facets_A))
            facet_tags_B = mesh.meshtags(mshB,mshB.topology.dim-1,bndry_facets_B,np.ones_like(bndry_facets_B))

            poly_C = compute_union_polygon(mshA, facet_tags_A, mshB, facet_tags_B)

            #TODO: this really should be based on just the overlapping cells and extract the 
            # minimum mesh size in the overlap region, not the min of both meshes' average cell size
            mesh_size = np.min([self.XSs[collision[0]].mesh_size,
                               self.XSs[collision[1]].mesh_size])
            mesh_C = mesh_from_polygon(poly_C,mesh_size=0.25*mesh_size,mesh_name='mortar_mesh')
            self.collisions[collision].mortar_mesh = MortarMesh(mesh_C)
            #TODO: the material field could be interpolated from the foreground meshes instead of just 
            #       populating straight from a single material property
            self.collisions[collision].mortar_xs = CrossSection(
                mesh_C,
                materials=self.XSs[0].materials,
                celltags=None,
                degree=1,
                verbose=False,
            )

    def _construct_mortar_mesh_fxns(self):
        for collision in self.collisions:
            #mortar mesh has previously been constructed:
            mesh_C = self.collisions[collision].mortar_mesh.msh
        
            #intialize functions on mortar mesh and add to collision
            if hasattr(self.collisions[collision], "mortar_xs"):
                mortar_xs = self.collisions[collision].mortar_xs
                self.collisions[collision].fxn_space = mortar_xs.V_w
                self.collisions[collision].u = mortar_xs.du
                self.collisions[collision].v = mortar_xs.v
                self.collisions[collision].dx = mortar_xs.dx
                VC = mortar_xs.V_w
            else:
                Ve_C = element("CG",mesh_C.topology.cell_name(),1,shape=(3,))
                self.collisions[collision].fxn_space = fem.functionspace(mesh_C, mixed_element(4*[Ve_C]))
                VC = self.collisions[collision].fxn_space
                
                self.collisions[collision].u = TrialFunction(VC)
                self.collisions[collision].v = TestFunction(VC)
                self.collisions[collision].dx = Measure("dx",domain=mesh_C)
                # uC = self.collisions[collision].u
                # vC = self.collisions[collision].v
                # dx_C = self.collisions[collision].dx

            self.collisions[collision].mortar_mesh.size = VC.dofmap.index_map.size_global * VC.dofmap.index_map_bs


    def _construct_interpolation_operators(self):
        for collision in self.collisions:
            #construct projection operators
            VC = self.collisions[collision].fxn_space

            self.collisions[collision].PA = get_interpolation_matrix(VC,self.XSs[collision[0]].V_w,mixed=True)
            self.collisions[collision].PB = get_interpolation_matrix(VC,self.XSs[collision[1]].V_w,mixed=True)
    
    
    def _construct_interpolation_operator(self,collision,mesh_id):
        VC = self.collisions[collision].fxn_space

        return get_interpolation_matrix(VC,self.XSs[collision[mesh_id]].V_w,mixed=True)
    

    def _construct_mortar_forms(self):
        for collision in self.collisions:
            i,j=self.collisions[collision].mortar_xs.i,self.collisions[collision].mortar_xs.j
            #mortar mesh has previously been constructed:
            mesh_C = self.collisions[collision].mortar_mesh.msh
            VC = self.collisions[collision].fxn_space
            uC = self.collisions[collision].u
            vC = self.collisions[collision].v
            dx_C = self.collisions[collision].dx

            #construct displacement term (penalty weighted mass matrix)
            MC_ufl = self.nu_u * inner(uC, vC) * dx_C
            # MC_ufl = self.nu_u * inner(uC, vC) * ds
            MC_form = fem.form(MC_ufl)
            MC = fem.petsc.assemble_matrix(MC_form)
            MC.assemble()
            self.collisions[collision].MC_ufl = MC_ufl
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
            eps_C = self.collisions[collision].mortar_xs.warping2strain(uC,0)
            eps1_C = self.collisions[collision].mortar_xs.warping2strain(uC,1)
            sigma_c =  as_tensor(C_C[i,j,k,l]*eps_C[k,l],(i,j))
            sigma1_c =  as_tensor(C_C[i,j,k,l]*eps1_C[k,l],(i,j))

            #test function strain/stress:
            eps_vC = self.collisions[collision].mortar_xs.warping2strain(vC,0)
            eps1_vC = self.collisions[collision].mortar_xs.warping2strain(vC,1)
            sigma_vC =  as_tensor(C_C[i,j,k,l]*eps_vC[k,l],(i,j))
            sigma1_vC =  as_tensor(C_C[i,j,k,l]*eps1_vC[k,l],(i,j))

            #traction stiffness matrix:
            SC_ufl = self.nu_t * dot(dot(sigma_c,n3),dot(sigma_vC,n3))*ds
            # SC_ufl = self.nu_t * inner(sigma_c,sigma_vC)+inner(sigma_c,sigma_vC)*dx_C
            # SC_ufl = self.nu_t * inner(sigma_c,sigma_vC)*dx_C
            # SC_ufl = self.nu_t * inner(eps_vC,eps_C)*dx_C
            # SC_ufl = self.nu_t * inner(eps_vC,eps_C)*ds
            # SC_ufl = self.nu_t * (inner(eps_vC,eps_C)+ inner(eps1_vC,eps1_C))*dx_C
            # SC_ufl = self.nu_t * (dot(dot(sigma_c,n3),dot(sigma_vC,n3))+ dot(dot(sigma1_c,n3),dot(sigma1_vC,n3)))*ds
            # SC_ufl = self.nu_t * sigma_vC[i,j]*eps_C[i,j]*dx_C
            SC_form = fem.form(SC_ufl)
            S_C = fem.petsc.assemble_matrix(SC_form)
            S_C.assemble()
            self.collisions[collision].SC_ufl = SC_ufl
            self.collisions[collision].SC_form = SC_form
            self.collisions[collision].S_C = S_C

            if self.enable_overlap_correction:
                # Construct the overlap-correction form only when requested.
                self.collisions[collision].mortar_xs._construct_xs_form()
                self.collisions[collision].mortar_xs.a_form = fem.form(
                    self.collisions[collision].mortar_xs.a00
                )
            else:
                self.collisions[collision].mortar_xs.a_form = None
            

    def _assemble_mortar_matrices(self):
        for collision in self.collisions:
            #assemble area term:
            MC_form = self.collisions[collision].MC_form 
            MC = fem.petsc.assemble_matrix(MC_form)
            MC.assemble()
            self.collisions[collision].MC = MC

            #assemble boundary term:
            SC_form = self.collisions[collision].SC_form 
            S_C = fem.petsc.assemble_matrix(SC_form)
            S_C.assemble()
            # self.collisions[collision].SC_form = S_C_form
            self.collisions[collision].S_C = S_C

            if self.enable_overlap_correction:
                self.collisions[collision].mortar_xs.K_bar = fem.petsc.assemble_matrix(
                    self.collisions[collision].mortar_xs.a_form
                )
                self.collisions[collision].mortar_xs.K_bar.assemble()
            else:
                self.collisions[collision].mortar_xs.K_bar = None
    
    # def _construct_overlap_correction(self):
    #     '''
    #     construct the mortar warping stiffness matrix used in overlap correction
        
    #     '''
    #     for collision in self.collisions:
    #         self.collisions[collision].mortar_xs._construct_xs_form()
    #         self.collisions[collision].mortar_xs.a_form = fem.form(self.collisions[collision].mortar_xs.a00)
    #         self.collisions[collision].mortar_xs.K_bar = fem.petsc.assemble_matrix(self.collisions[collision].mortar_xs.a_form)

    def _construct_coupling_terms(self):
        for collision in self.collisions:

            #interpolation operators
            PA = self.collisions[collision].PA
            PB = self.collisions[collision].PB

            #construct displacement penalty terms
            MC = self.collisions[collision].MC

            S_AA = AT_C_B(PA, MC, PA)
            S_AB = AT_C_B(PA, MC, PB)
            S_BA = AT_C_B(PB, MC, PA)
            S_BB = AT_C_B(PB, MC, PB)

            #add traction term to the penalty terms:
            S_C = self.collisions[collision].S_C 

            S_AA.axpy(1.0, AT_C_B(PA, S_C, PA) )
            S_AB.axpy(1.0, AT_C_B(PA, S_C, PB) )
            S_BA.axpy(1.0, AT_C_B(PB, S_C, PA) )
            S_BB.axpy(1.0, AT_C_B(PB, S_C, PB) )

            if self.enable_overlap_correction:
                K_C = self.collisions[collision].mortar_xs.K_bar
                S_AA.axpy(-0.5,AT_C_B(PA,K_C,PA))
                S_BB.axpy(-0.5,AT_C_B(PB,K_C,PB))

            #store to collision for later addition to overall system
            self.collisions[collision].Sij = [[S_AA,S_AB],
                                              [S_BA,S_BB]]
        
        return
    
    def _apply_coupling(self):
        for enum_idx, (msh_indices,collision) in enumerate(self.collisions.items()):
            for a_num,idx_i in enumerate(msh_indices):
                for b_num,idx_j in enumerate(msh_indices):
                    if idx_i == idx_j:
                        scale = 1.0
                    else:
                        scale = -1.0
                    #add coupling term to system matrices
                    self.system_matrices[idx_i][idx_j].axpy(scale,self.collisions[msh_indices].Sij[a_num][b_num])
    

    def _set_up_solver(self):
        #Set up the full block system
        self.system_mat = PETSc.Mat()
        self.system_mat.createNest(self.system_matrices)

        # #set up rhs vector form
        # self.b1_form = fem.form(self.XSs[0].L_form[1])

        # set up the solver with the LHS
        self.solver = PETSc.KSP().create(self.meshes[0].comm)
        self.solver.setOperators(self.system_mat)
        self.solver.setType("preonly")
        pc = self.solver.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        
        return

    def _solve_coupled_system(self):
        #================== solve constrained system for each mode ==================#           
        for idx_l in range(6):
            self.XSs[0].f1.value = 0
            self.XSs[0].f1.value[idx_l] = 1
            
            self.system_RHS_vectors[-1].array[:] = fem.petsc.assemble_vector(self.b1_form_assembled).array

            #TODO: Lucky us, no special BCS to apply rn, may change if there were any elastic foundations, etc
            b = PETSc.Vec().createNest(self.system_RHS_vectors)

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
            for xs_num,xs in enumerate(self.XSs):
                xs.warping_functions[idx_l].x.array[:] = x_local[xs_num]
                xs.lmbdas[idx_l].x.array[:] = x_local[-1]

            # uh.x.scatter_forward()
            # lmbdah.x.scatter_forward()

        return
    
    def _compile_coupling_vjp_forms(self,collision):
        print('compling coupling forms...')
        self.collisions[collision].u_j = fem.Function(self.collisions[collision].fxn_space)
        self.collisions[collision].v_j = fem.Function(self.collisions[collision].fxn_space)
        
        x = self.collisions[collision].mortar_mesh.x
        dX = self.collisions[collision].mortar_mesh.dX
        u_j = self.collisions[collision].u_j 
        v_j = self.collisions[collision].v_j 
        MC_ufl= self.collisions[collision].MC_ufl
        SC_ufl = self.collisions[collision].SC_ufl
        self.collisions[collision].pMCpx_form = fem.form(ufl.derivative(ufl.action(ufl.action(MC_ufl,u_j),v_j),x,dX))
        self.collisions[collision].pSCpx_form = fem.form(ufl.derivative(ufl.action(ufl.action(SC_ufl,u_j),v_j),x,dX))
        if self.enable_overlap_correction:
            KC_ufl = self.collisions[collision].mortar_xs.a00
            self.collisions[collision].pKCpx_form = fem.form(
                ufl.derivative(ufl.action(ufl.action(KC_ufl, u_j), v_j), x, dX)
            )
        else:
            self.collisions[collision].pKCpx_form = None
        print('DONE compling coupling forms')


    def _compute_vjp_dMC(self,d_output,collision):
        VX = self.collisions[collision].mortar_mesh.V_x
        d_inputs_size = VX.dofmap.index_map_bs * VX.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)

        u_j = self.collisions[collision].u_j 
        v_j = self.collisions[collision].v_j

        for j in np.unique(np.nonzero(d_output)[1]):
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec


            vec = fem.assemble_vector(self.collisions[collision].pMCpx_form)
            d_input += vec.array

        return d_input
    
    def _compute_vjp_dSC(self,d_output,collision):
        VX = self.collisions[collision].mortar_mesh.V_x
        d_inputs_size = VX.dofmap.index_map_bs * VX.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)

        u_j = self.collisions[collision].u_j 
        v_j = self.collisions[collision].v_j

        for j in np.unique(np.nonzero(d_output)[1]):
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec


            vec = fem.assemble_vector(self.collisions[collision].pSCpx_form)
            d_input += vec.array

        return d_input

    def _compute_vjp_dKC(self,d_output,collision):
        if not self.enable_overlap_correction:
            VX = self.collisions[collision].mortar_mesh.V_x
            d_inputs_size = VX.dofmap.index_map_bs * VX.dofmap.index_map.size_global
            return np.zeros(d_inputs_size)

        VX = self.collisions[collision].mortar_mesh.V_x
        d_inputs_size = VX.dofmap.index_map_bs * VX.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)

        u_j = self.collisions[collision].u_j 
        v_j = self.collisions[collision].v_j

        for j in np.unique(np.nonzero(d_output)[1]):
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec


            vec = fem.assemble_vector(self.collisions[collision].pKCpx_form)
            d_input += vec.array

        return d_input
    

    def _compute_vjp_component_spatial(self,form,collision,d_output,test_space ,trial_space = None):
        
        # dFdx = ufl.derivative(form,self.x,self.dX)

        # #create a function and set numpy
        # d_output_func = fem.Function(fxn_space)
        # d_output_func.x.array = d_output

        # vec_form = ufl.action(ufl.adjoint(form),d_output_func)

        # dFdx_dx = fem.petsc.assemble_vector(fem.form(vec_form))

        #TODO: we can probably speed this up by being a bit more intelligent with the function values. 
        # Really... we shouldn't need a loop here at all. 
        x = self.collisions[collision].mortar_mesh.x
        dX = self.collisions[collision].mortar_mesh.dX

        # dFormdx = ufl.derivative(form,x,dX)

        VX = self.collisions[collision].mortar_mesh.V_x
        d_inputs_size = VX.dofmap.index_map_bs * VX.dofmap.index_map.size_global
        d_input = np.zeros(d_inputs_size)
        
        if trial_space==None:
            trial_space=test_space
        
        u_j = fem.Function(trial_space)
        v_j = fem.Function(test_space)

        form_cache = fem.form(ufl.derivative(ufl.action(ufl.action(form,u_j),v_j),x,dX))
        
        # for j in range(d_output.shape[0]):
        # use the nonzero column indices only (instead of all columns regardless of value)                 
        for j in np.nonzero(d_output)[1]:
            u_j.x.array[:] = 0.0
            u_j.x.array[j] = 1.0                

            v_j.x.array[:] = d_output[:, j]     # column j = coefficients Λ_{ij}

            #this only computes the scalar value, needs to be done with the spatial derivative
            # fem.assemble_scalar(fem.form(ufl.action(ufl.action(self.a_form[0][0],u_j),v_j)))

            #TODO: update to use the actual passed form and function spaces
            #i barely understand this myself, but we start with a bilinear form, then we compute the "double-action", 
            # this gives us a form (scalar), then we take the spatial derivative of that and add to the d_inputs vec

            # vec = fem.assemble_vector(fem.form(ufl.derivative(ufl.action(ufl.action(form,u_j),v_j),self.x,self.dX)))
            # vec = fem.assemble_vector(fem.form(ufl.action(ufl.action(dFormdx,u_j),v_j)))
            vec = fem.assemble_vector(form_cache)
            d_input += vec.array

        return d_input

    def apply_inverse_jacobian(self, d_output_w_A, d_output_w_B, d_output_lmbda):
        """
        Solve the transpose of the coupled KKT system for each of the 6 RHS columns.

        Inputs:
            d_output_w_A    : (nA, 6)
            d_output_w_B    : (nB, 6)
            d_output_lmbda  : (nL, 6)

        Returns:
            d_residuals_w_A   : (nA, 6)
            d_residuals_w_B   : (nB, 6)
            d_residuals_lmbda : (nL, 6)
        """
        nA = d_output_w_A.shape[0]
        nB = d_output_w_B.shape[0]
        nL = d_output_lmbda.shape[0]

        d_residuals_w_A = np.zeros_like(d_output_w_A)
        d_residuals_w_B = np.zeros_like(d_output_w_B)
        d_residuals_lmbda = np.zeros_like(d_output_lmbda)

        offsets = np.cumsum([0] + self.system_size_list)

        for k in range(d_output_w_A.shape[1]):
            rhs = self.system_mat.createVecRight()
            lhs = self.system_mat.createVecLeft()

            rhs.array[:] = 0.0
            lhs.array[:] = 0.0

            rhs.array[offsets[0]:offsets[1]] = d_output_w_A[:, k]
            rhs.array[offsets[1]:offsets[2]] = d_output_w_B[:, k]
            rhs.array[offsets[2]:offsets[3]] = d_output_lmbda[:, k]

            self.solver.solveTranspose(rhs, lhs)

            d_residuals_w_A[:, k] = lhs.array[offsets[0]:offsets[1]].copy()
            d_residuals_w_B[:, k] = lhs.array[offsets[1]:offsets[2]].copy()
            d_residuals_lmbda[:, k] = lhs.array[offsets[2]:offsets[3]].copy()

        return d_residuals_w_A, d_residuals_w_B, d_residuals_lmbda
    
    def compute_VJP(self, d_residuals_w_A, d_residuals_w_B, d_residuals_lmbda):
        """
        Reverse-mode VJP for the coupled residual:
            (dR/dx)^T * seed

        Inputs:
            d_residuals_w_A   : (nA, 6)
            d_residuals_w_B   : (nB, 6)
            d_residuals_lmbda : (nL, 6)

        Returns:
            dRdx_A : full coordinate derivative vector for mesh A
            dRdx_B : full coordinate derivative vector for mesh B
            dRdx_C : full coordinate derivative vector for mortar mesh

        Assumptions:
            - one collision only: (0,1)
            - current geometry / state have already been updated
            - _get_warping_functions() has already been run for the current geometry
        """
        collision = (0, 1)

        # lazily compile component / coupling VJP forms if needed
        if not hasattr(self.XSs[0], "pA00px_form"):
            self.XSs[0]._compile_component_vjp_forms()
        if not hasattr(self.XSs[1], "pA00px_form"):
            self.XSs[1]._compile_component_vjp_forms()
        if not hasattr(self.collisions[collision], "pMCpx_form"):
            self._compile_coupling_vjp_forms(collision)

        # current interpolation / mortar matrices from the latest forward solve
        P_A = convert_petsc_to_numpy(self.collisions[collision].PA)
        P_B = convert_petsc_to_numpy(self.collisions[collision].PB)

        MC = convert_petsc_to_numpy(self.collisions[collision].MC)
        SC = convert_petsc_to_numpy(self.collisions[collision].S_C)
        U = MC + SC
        if self.enable_overlap_correction:
            KC = convert_petsc_to_numpy(self.collisions[collision].mortar_xs.K_bar)
            T = U - 0.5 * KC
        else:
            T = U

        dRdx_A = np.zeros(self.XSs[0].V_x.dofmap.index_map_bs * self.XSs[0].V_x.dofmap.index_map.size_global)
        dRdx_B = np.zeros(self.XSs[1].V_x.dofmap.index_map_bs * self.XSs[1].V_x.dofmap.index_map.size_global)
        dRdx_C = np.zeros(self.collisions[collision].mortar_mesh.V_x.dofmap.index_map_bs *
                        self.collisions[collision].mortar_mesh.V_x.dofmap.index_map.size_global)
        DEBUG_FOREGROUND = True
        DEBUG_MORTAR = True
        DEBUG_INTERP = True

        for k in range(6):
            # current state for mode k
            w_A = self.XSs[0].warping_functions[k].x.array.copy()
            w_B = self.XSs[1].warping_functions[k].x.array.copy()
            lmbda = self.XSs[0].lmbdas[k].x.array.copy()

            # residual seeds for mode k
            r_A = d_residuals_w_A[:, k]
            r_B = d_residuals_w_B[:, k]
            r_L = d_residuals_lmbda[:, k]

            # -------------------------
            # component seeds
            # -------------------------
            seed_K_A = np.outer(r_A, w_A)
            seed_K_B = np.outer(r_B, w_B)

            seed_C_A = np.outer(lmbda, r_A) + np.outer(r_L, w_A)
            seed_C_B = np.outer(lmbda, r_B) + np.outer(r_L, w_B)
            
            if DEBUG_FOREGROUND:
                # VJP wrt foreground geometries from uncoupled blocks
                dRdx_A += self.XSs[0]._compute_vjp_dA00dx(seed_K_A)
                dRdx_A += self.XSs[0]._compute_vjp_dA10dx(seed_C_A)

                dRdx_B += self.XSs[1]._compute_vjp_dA00dx(seed_K_B)
                dRdx_B += self.XSs[1]._compute_vjp_dA10dx(seed_C_B)

            # -------------------------
            # projected state/seed vectors on mortar mesh
            # -------------------------
            PA_wA = P_A @ w_A
            PB_wB = P_B @ w_B
            PA_rA = P_A @ r_A
            PB_rB = P_B @ r_B

            # -------------------------
            # mortar matrix seeds
            # -------------------------
            seed_MC = (
                np.outer(PA_rA, PA_wA)
                - np.outer(PA_rA, PB_wB)
                - np.outer(PB_rB, PA_wA)
                + np.outer(PB_rB, PB_wB)
            )

            # SC has the same algebraic placement as MC
            seed_SC = seed_MC.copy()

            # KC only appears on diagonal terms with -0.5 coefficient
            if DEBUG_MORTAR:
                dRdx_C += self._compute_vjp_dMC(seed_MC, collision)
                dRdx_C += self._compute_vjp_dSC(seed_SC, collision)
                if self.enable_overlap_correction:
                    seed_KC = (
                        -0.5 * np.outer(PA_rA, PA_wA)
                        -0.5 * np.outer(PB_rB, PB_wB)
                    )
                    dRdx_C += self._compute_vjp_dKC(seed_KC, collision)

            # -------------------------
            # interpolation matrix seeds
            # -------------------------
            # d/dP_A of:
            #   r_A^T P_A^T T P_A w_A
            # - r_A^T P_A^T U P_B w_B
            # - r_B^T P_B^T U P_A w_A
            seed_P_A = (
                np.outer(T @ PA_wA, r_A)
                + np.outer(T.T @ PA_rA, w_A)
                - np.outer(U @ PB_wB, r_A)
                - np.outer(U.T @ PB_rB, w_A)
            )

            # d/dP_B of:
            # - r_A^T P_A^T U P_B w_B
            # - r_B^T P_B^T U P_A w_A
            # + r_B^T P_B^T T P_B w_B
            seed_P_B = (
                - np.outer(U.T @ PA_rA, w_B)
                - np.outer(U @ PA_wA, r_B)
                + np.outer(T @ PB_wB, r_B)
                + np.outer(T.T @ PB_rB, w_B)
            )
            if DEBUG_INTERP:
                dPdx_A, dPdx_C_from_A = self._compute_vjp_dP(seed_P_A, collision, mesh_id=0)
                dPdx_B, dPdx_C_from_B = self._compute_vjp_dP(seed_P_B, collision, mesh_id=1)

                dRdx_A += dPdx_A
                dRdx_B += dPdx_B
                dRdx_C += dPdx_C_from_A + dPdx_C_from_B

        return dRdx_A, dRdx_B, dRdx_C


    def _compute_vjp_dP(self, d_output, collision, mesh_id):
        """
        Compute VJP of an interpolation matrix P wrt:
            - foreground mesh geometry (mesh_id)
            - mortar mesh geometry

        Returns:
            dRdx_foreground : full coordinate derivative vector on foreground mesh
            dRdx_mortar     : full coordinate derivative vector on mortar mesh
        """
        from ALBATROSS.nonmatching_utils import action_of_geom_on_nm_interpolation_matrix

        xs_fg = self.XSs[mesh_id]
        mortar = self.collisions[collision].mortar_mesh

        dx_fg_nodes = np.zeros((xs_fg.msh.geometry.x.shape[0], 2))
        dx_C_nodes = np.zeros((mortar.msh.geometry.x.shape[0], 2))

        # identical logic to your current explicit op
        for i in range(4):
            for j in range(3):
                sub_space_fg, sub_space_fg_dofmap = xs_fg.V_w.sub(i).sub(j).collapse()
                sub_space_C, sub_space_C_dofmap = self.collisions[collision].fxn_space.sub(i).sub(j).collapse()

                dP_block = d_output[np.ix_(sub_space_C_dofmap, sub_space_fg_dofmap)]

                dx_fg_ij, dx_C_ij = action_of_geom_on_nm_interpolation_matrix(
                    sub_space_C,
                    sub_space_fg,
                    dP=dP_block,
                )

                dx_fg_nodes += dx_fg_ij
                dx_C_nodes += dx_C_ij

        # convert node-wise x/y perturbations to the V_x dof ordering
        dRdx_fg = np.zeros(xs_fg.V_x.dofmap.index_map_bs * xs_fg.V_x.dofmap.index_map.size_global)
        dRdx_C = np.zeros(mortar.V_x.dofmap.index_map_bs * mortar.V_x.dofmap.index_map.size_global)

        dRdx_fg[xs_fg.dofs_x_boundary] = dx_fg_nodes[xs_fg.boundary_nodes, 0]
        dRdx_fg[xs_fg.dofs_y_boundary] = dx_fg_nodes[xs_fg.boundary_nodes, 1]
        dRdx_fg[xs_fg.dofs_x_interior] = dx_fg_nodes[xs_fg.interior_nodes, 0]
        dRdx_fg[xs_fg.dofs_y_interior] = dx_fg_nodes[xs_fg.interior_nodes, 1]

        dRdx_C[mortar.dofs_x_boundary] = dx_C_nodes[mortar.boundary_nodes, 0]
        dRdx_C[mortar.dofs_y_boundary] = dx_C_nodes[mortar.boundary_nodes, 1]
        dRdx_C[mortar.dofs_x_interior] = dx_C_nodes[mortar.interior_nodes, 0]
        dRdx_C[mortar.dofs_y_interior] = dx_C_nodes[mortar.interior_nodes, 1]

        return dRdx_fg, dRdx_C

    # def compute_VJP(self, d_residuals_w_a, d_residuals_w_b, d_residuals_w_c, d_residuals_lmbda):
    #     '''
    #     INPUTS:
    #     d_residual_w_a shape : num_dofs_a x 6
    #     d_residual_w_b shape : num_dofs_a x 6
    #     d_residual_w_c shape : num_dofs_a x 6
    #     d_residual_l shape : num_lms x 6

    #     OUTPUTS:
    #     dRdxa_dr = num_nodes_a
    #     dRdxb_dr = num_nodes_b
    #     dRdxc_dr = num_nodes_c
    #     '''
    #     #set up input vector sizes
    #     d_residuals_w_a_vec_size = d_residuals_w_a.shape[0]
    #     d_residuals_w_a_vec = PETSc.Vec().createSeq(d_residuals_w_a_vec_size, comm=PETSc.COMM_SELF)

    #     d_residuals_w_b_vec_size = d_residuals_w_b.shape[0]
    #     d_residuals_w_b_vec = PETSc.Vec().createSeq(d_residuals_w_b_vec_size, comm=PETSc.COMM_SELF)

    #     d_residuals_w_c_vec_size = d_residuals_w_c.shape[0]
    #     d_residuals_w_c_vec = PETSc.Vec().createSeq(d_residuals_w_c_vec_size, comm=PETSc.COMM_SELF)
        
    #     d_residuals_lmbda_vec_size = d_residuals_lmbda.shape[0]
    #     d_residuals_lmbda_vec = PETSc.Vec().createSeq(d_residuals_lmbda_vec_size, comm=PETSc.COMM_SELF)
        
    #     #set up output vector sizes
    #     d_inputs_a_vec_size = self.XSs[0].V_x.dofmap.index_map_bs*self.XSs[0].V_x.dofmap.index_map.size_global
    #     d_inputs_a_vec = PETSc.Vec().createSeq(d_inputs_a_vec_size, comm=PETSc.COMM_SELF)

    #     d_inputs_b_vec_size = self.XSs[1].V_x.dofmap.index_map_bs*self.XSs[1].V_x.dofmap.index_map.size_global
    #     d_inputs_b_vec = PETSc.Vec().createSeq(d_inputs_b_vec_size, comm=PETSc.COMM_SELF)

    #     d_inputs_c_vec_size = self.collisions[(0,1)].mortar_mesh.V_x.dofmap.index_map_bs*self.collisions[(0,1)].mortar_mesh.V_x.dofmap.index_map.size_global
    #     d_inputs_c_vec = PETSc.Vec().createSeq(d_inputs_c_vec_size, comm=PETSc.COMM_SELF)
        
    #     dRdxa_dr = np.zeros(d_inputs_a_vec_size)
    #     dRdxb_dr = np.zeros(d_inputs_b_vec_size)
    #     dRdxc_dr = np.zeros(d_inputs_c_vec_size)
        
    #     #TODO: these really need to be re-formulated to compute actions, not full vec-mat products
    #     #TODO: self._compute_spatial_partials_penalty_term() needs to be implemented
    #     #TODO: need to store residual forms for each individual mesh region (this is done in the ._solve_system() method for the uncoupled case)
    #     for idx in range(d_residuals_w_a.shape[1]):
    #         #====== dRadx_dr =======#
    #         dRwadxa = self._compute_spatial_partials(self.residuals[idx][0]) #num_dofs x num_nodes
    #         dRwbdxa = self._compute_spatial_partials_penalty_term(of=coupled_residual_b,wrt=x_a)
    #         dRwcdxa = self._compute_spatial_partials_penalty_term(of=coupled_residual_c,wrt=x_a)
            
    #         #"uncoupled" portion over the foreground mesh
    #         d_residuals_w_a_vec.array = d_residuals_w_a[:,idx]
    #         dRwadxa.multTranspose(d_residuals_w_a_vec,d_inputs_a_vec) #perform vec-mat product
    #         dRdxa_dr += d_inputs_a_vec.array
            
    #         dRldx = self._compute_spatial_partials(self.residuals[idx][1] )#num_lms x num_nodes
    #         d_residuals_lmbda_vec.array = d_residuals_lmbda[:,idx]
    #         dRldx.multTranspose(d_residuals_lmbda_vec,d_inputs_a_vec) #perform vec-mat product
    #         dRdxa_dr += d_inputs_a_vec.array

    #         #coupled portion:
    #         d_residuals_w_b_vec.array = d_residuals_w_b[:,idx]
    #         dRwbdxa.multTranspose(d_residuals_w_b_vec,d_inputs_a_vec) #perform vec-mat product
    #         dRdxa_dr += d_inputs_a_vec.array

    #         d_residuals_w_c_vec.array = d_residuals_w_c[:,idx]
    #         dRwcdxa.multTranspose(d_residuals_w_c_vec,d_inputs_a_vec) #perform vec-mat product
    #         dRdxa_dr += d_inputs_a_vec.array


    #         #====== dRbdx_dr =======#
    #         dRwadxb = self._compute_spatial_partials_penalty_term(of=coupled_residual_a,wrt=x_b)
    #         dRwbdxb = self._compute_spatial_partials(self.residuals[idx][0]) #num_dofs x num_nodes
    #         dRwcdxb = self._compute_spatial_partials_penalty_term(of=coupled_residual_c,wrt=x_b)
            
    #         #"uncoupled" portion over the foreground mesh
    #         d_residuals_w_b_vec.array = d_residuals_w_b[:,idx]
    #         dRwbdxb.multTranspose(d_residuals_w_b_vec,d_inputs_b_vec) #perform vec-mat product
    #         dRdxb_dr += d_inputs_b_vec.array
            
    #         dRldx = self._compute_spatial_partials(self.residuals[idx][1] )#num_lms x num_nodes
    #         d_residuals_lmbda_vec.array = d_residuals_lmbda[:,idx]
    #         dRldx.multTranspose(d_residuals_lmbda_vec,d_inputs_b_vec) #perform vec-mat product
    #         dRdxb_dr += d_inputs_b_vec.array

    #         #coupled portion:
    #         d_residuals_w_a_vec.array = d_residuals_w_a[:,idx]
    #         dRwadxb.multTranspose(d_residuals_w_b_vec,d_inputs_b_vec) #perform vec-mat product
    #         dRdxb_dr += d_inputs_b_vec.array

    #         d_residuals_w_c_vec.array = d_residuals_w_c[:,idx]
    #         dRwcdxb.multTranspose(d_residuals_w_c_vec,d_inputs_b_vec) #perform vec-mat product
    #         dRdxb_dr += d_inputs_b_vec.array
            
    #         #====== dRcdx_dr =======#
    #         dRwadxc = self._compute_spatial_partials_penalty_term(of=coupled_residual_a,wrt=x_c)
    #         dRwbdxc = self._compute_spatial_partials_penalty_term(of=coupled_residual_b,wrt=x_c)
    #         dRwcdxc = self._compute_spatial_partials(self.residuals[idx][0]) #num_dofs x num_nodes

    #         #"uncoupled" portion over the mortar mesh
    #         d_residuals_w_c_vec.array = d_residuals_w_c[:,idx]
    #         dRwcdxc.multTranspose(d_residuals_w_c_vec,d_inputs_c_vec) #perform vec-mat product
    #         dRdxc_dr += d_inputs_c_vec.array
            
    #         dRldx = self._compute_spatial_partials(self.residuals[idx][1] )#num_lms x num_nodes
    #         d_residuals_lmbda_vec.array = d_residuals_lmbda[:,idx]
    #         dRldx.multTranspose(d_residuals_lmbda_vec,d_inputs_c_vec) #perform vec-mat product
    #         dRdxc_dr += d_inputs_c_vec.array

    #         #coupled portion:
    #         d_residuals_w_a_vec.array = d_residuals_w_a[:,idx]
    #         dRwadxc.multTranspose(d_residuals_w_b_vec,d_inputs_c_vec) #perform vec-mat product
    #         dRdxc_dr += d_inputs_c_vec.array

    #         d_residuals_w_b_vec.array = d_residuals_w_b[:,idx]
    #         dRwbdxc.multTranspose(d_residuals_w_b_vec,d_inputs_c_vec) #perform vec-mat product
    #         dRdxc_dr += d_inputs_c_vec.array


    #     return dRdx_dr
    
    # def _compute_pA_action(self,dK,mesh_id=0,derivative_type='x'):
    #     '''
    #     compute the action of the seed dK on the input based on derivative_type

    #     return numpy arrays
    #     '''
    #     XS = self.XSs[mesh_id]
    #     K1 = self.K1
    #     K2inv = self.K2inv
    #     K2 = self.K2

    #     #get K1 and K2 adjoint loads
    #     W_1 = dK @ K1 @ K2inv + dK.T @ K2inv @ K1
    #     W_2 = K2inv @ K1.T @ dK @ K1 @ K2inv

    #     if derivative_type == 'x':
    #         d_form = 0
            
    #         for idx_i,idx_j in np.argwhere(W_1):
    #             d_form += W_1[idx_i,idx_j] * XS.K1_form[idx_i][idx_j]     
    #         for idx_i,idx_j in np.argwhere(W_2):
    #             d_form -= W_2[idx_i,idx_j] * XS.K2_form[idx_i][idx_j]

    #         d_inputs = fem.petsc.assemble_vector(fem.form(ufl.derivative(d_form,XS.x,XS.dX)))

    #         return d_inputs.array

    def _compute_pK_action(self,dK,mesh_id=0,derivative_type='x'):
        '''
        compute the action of the seed dK on the input based on derivative_type

        return numpy arrays
        '''
        XS = self.XSs[mesh_id]
        K1 = self.K1
        K2inv = self.K2inv
        K2 = self.K2

        #get K1 and K2 adjoint loads for the full coupled matrix
        W_1 = dK @ K1 @ K2inv + K2inv @ K1.T @ dK
        W_2 = K2inv @ K1.T @ dK @ K1 @ K2inv

        if derivative_type == 'x':
            # d_form = 0
            
            # for idx_i,idx_j in np.argwhere(W_1):
            #     d_form += W_1[idx_i,idx_j] * XS.K1_form[idx_i][idx_j]     
            # for idx_i,idx_j in np.argwhere(W_2):
            #     d_form -= W_2[idx_i,idx_j] * XS.K2_form[idx_i][idx_j]

            # d_inputs = fem.petsc.assemble_vector(fem.form(ufl.derivative(d_form,XS.x,XS.dX)))

            # return d_inputs.array

            #update xs adjoint load weights
            XS.W_1.value = W_1
            XS.W_2.value = W_2
            
            #assemble vector
            d_inputs = fem.petsc.assemble_vector(XS.dKdx_form)

            return d_inputs.array
        
        if derivative_type == 'w':
            # d_form = 0

            d_inputs = np.zeros((XS.V_w.dofmap.index_map_bs*XS.V_w.dofmap.index_map.size_global,6))
            # indices_i,indices_j = np.nonzero(dK)
            #loop over warping functions:
            for idx_k in range(6):
                # for idx_i,idx_j in np.argwhere(W_1):
                #     d_form += W_1[idx_i,idx_j] * XS.K1_form[idx_i][idx_j]     
                # for idx_i,idx_j in np.argwhere(W_2):
                #     d_form -= W_2[idx_i,idx_j] * XS.K2_form[idx_i][idx_j]

                # d_inputs[:,idx_k] = fem.petsc.assemble_vector(fem.form(ufl.derivative(d_form,XS.warping_functions[idx_k])))
                
                #update xs adjoint load weights
                XS.W_1.value = W_1
                XS.W_2.value = W_2

                d_inputs[:,idx_k] = fem.petsc.assemble_vector(XS.dKdw_form[idx_k]).array

            return d_inputs
        
        if derivative_type == 'l':
            # d_form = 0

            d_inputs = np.zeros((XS.V_lm.dofmap.index_map_bs*XS.V_lm.dofmap.index_map.size_global,6))
            # indices_i,indices_j = np.nonzero(dK)
            #loop over lagrange multipliers:
            for idx_k in range(6):
                # for idx_i,idx_j in np.argwhere(W_1):
                #     d_form += W_1[idx_i,idx_j] * XS.K1_form[idx_i][idx_j]     
                # for idx_i,idx_j in np.argwhere(W_2):
                #     d_form -= W_2[idx_i,idx_j] * XS.K2_form[idx_i][idx_j]

                # d_inputs[:,idx_k] = fem.petsc.assemble_vector(fem.form(ufl.derivative(d_form,XS.lmbdas[idx_k])))
                
                #update xs adjoint load weights
                XS.W_1.value = W_1
                XS.W_2.value = W_2

                d_inputs[:,idx_k] = fem.petsc.assemble_vector(XS.dKdl_form[idx_k]).array

            return d_inputs

        # self.pK1px_form = [[derivative(self.XSs[mesh_id].K1_form[idx1][idx2],self.x,self.dX)
        #                     for idx2 in range(6)] 
        #                         for idx1 in range(6)]
        # self.pK2px_form = [[derivative(self.K2_form[idx1][idx2],self.x,self.dX)
        #                     for idx2 in range(6)] 
        #                         for idx1 in range(6)]
                
        # self.pK1px_lol = [[petsc.assemble_vector(form(self.pK1px_form[idx1][idx2]))
        #                 for idx2 in range(6)] 
        #                     for idx1 in range(6)]
        # self.pK2px_lol = [[petsc.assemble_vector(form(self.pK2px_form[idx1][idx2]))
        #         for idx2 in range(6)] 
        #             for idx1 in range(6)]

        # #Term 1: (dK1/dx) @ K2inv @ K1^T
        # term1 = np.einsum("ijm,jk,kl->ilm", self.pK1px, self.K2inv, self.K1)
        # # Term 2: - K1 @ K2inv @ (dK2/dx) @ K2inv @ K1^T
        # term2 = -np.einsum("ij,jk,klm,ln,np->ipm", self.K1,self.K2inv,self.pK2px,self.K2inv,self.K1.T)
        # # Term 3: K1@ K2inv @ (dK1/dx)^T
        # term3 = np.einsum("ij,jk,lkm->ilm", self.K1, self.K2inv, self.pK1px)

        # #partial derivatives
        # self.pKpx = term1 + term2 + term3 

        # return self.pKpx.reshape((36,self.pKpx.shape[-1]))


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
            cells = celltags.find(1)
            # cells = np.concatenate([celltags.find(1),celltags.find(2)])
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

    def _compute_xs_stiffness_matrix(self):
        '''
        for each region, get the elastic solution modes and compute the stiffness
        store the accumulated matrices for recovery, etc
        '''
        self.K = np.zeros((6,6))
        self.K1 = np.zeros((6,6))
        self.K2 = np.zeros((6,6))
        # self.S = np.zeros((6,6))
        self.A = 0

        for i,region in zip(self.regions,self.regions.values()):
            self.XSs[i]._compute_xs_stiffness_matrix()
            self.K1 += self.XSs[i].K1
            self.K2 += self.XSs[i].K2

            self.A += self.XSs[i].A

        if self.enable_overlap_correction:
            for collision in self.collisions:
                for idx in range(6):
                    self.collisions[collision].PA.mult(
                        self.XSs[0].warping_functions[idx].x.petsc_vec,
                        self.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec,
                    )
                self.collisions[collision].mortar_xs._compute_xs_stiffness_matrix()
                self.K1 -= 0.5 * self.collisions[collision].mortar_xs.K1
                self.K2 -= 0.5 * self.collisions[collision].mortar_xs.K2

                for idx in range(6):
                    self.collisions[collision].PB.mult(
                        self.XSs[1].warping_functions[idx].x.petsc_vec,
                        self.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec,
                    )
                self.collisions[collision].mortar_xs._compute_xs_stiffness_matrix()

                self.K1 -= 0.5 * self.collisions[collision].mortar_xs.K1
                self.K2 -= 0.5 * self.collisions[collision].mortar_xs.K2
        
        #apply threshholding:
        s1 = np.max(np.abs(np.diag(self.K1)))
        s2 = np.max(np.abs(np.diag(self.K2)))
        eta = 1e-8
        mask_1 = (np.abs(self.K1) < s1*eta)
        mask_2 = (np.abs(self.K2) < s2*eta)
        np.fill_diagonal(mask_1,False)
        np.fill_diagonal(mask_2,False)
        self.K1[mask_1] = 0.0
        self.K2[mask_2] = 0.0 

        self.K1inv = np.linalg.inv(self.K1)

        self.K2inv = np.linalg.inv(self.K2)

        self.K = self.K1 @ self.K2inv @ self.K1.T

        self.A -= self.get_overlap_area()

        self.linear_density = self.A*self.XSs[0].materials[0].density
           

    def get_overlap_area(self,approach='under'):
        for idx,val in np.ndenumerate(self.adjacency):
            if val == 0:
                continue
            else:
                dx_overlap = Measure("dx", domain=self.regions[idx[0]].msh, subdomain_data=self.collisions[idx].celltags[1])

                if approach == 'under': 
                    return fem.assemble_scalar(fem.form(1.0*dx_overlap((1))))
                elif approach == 'over': 
                    return fem.assemble_scalar(fem.form(1.0*dx_overlap((1,2))))
                elif approach == 'average':
                    A_minus = fem.assemble_scalar(fem.form(1.0*dx_overlap((1))))
                    A_plus = fem.assemble_scalar(fem.form(1.0*dx_overlap((1,2)))) 
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

                V0,V0_to_V = xs.V_w.sub(0).collapse()
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
                
                V0,V0_to_V = xs.V_w.sub(0).collapse()
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


    def setup_recovery(self):
        """
        Build expressions from ufl and appropriate functionspaces for
        displacement and stress recovery.

        This must be called ONCE after cross-section analysis.
        """
        for xs in self.XSs:
            xs.setup_recovery()
        
    def coeff_from_reactions(self,reactions:np.ndarray):
        assert(reactions.shape==(6,))

        return self.K1inv@reactions
    
    
    def recover_displacement(self, reactions):
        '''     
        :param reactions: the 6 sectional forces and moments from the 1d solution
        
        :return: functions describing sectional displacement
        '''
        disps = []
        for xs in self.XSs:
            xs.warping_coeffs.value = self.coeff_from_reactions(reactions)

            u = fem.Function(xs.V_u)
            u.interpolate(xs.u_expr)

            disps.append(u)

        return disps
    

    def recover_stress(self,reactions):
        '''     
        :param reactions: the 6 sectional forces and moments from the 1d solution
        
        :return: function describing sectional displacement
        '''

        sigmas = []
        for xs in self.XSs:
            xs.warping_coeffs.value = self.coeff_from_reactions(reactions)

            sigma = fem.Function(xs.V_sigma)
            sigma.interpolate(xs.sigma_expr)

            sigmas.append(sigma)

        return sigmas

    def get_von_mises(self,reactions):
        '''
        Given the sectional reaction forces/moments, return a function with the von Mises stress
        
        inputs: reaction forces/moments
        
        '''
        vm_list = []
        for xs in self.XSs:
            xs.warping_coeffs.value = self.coeff_from_reactions(reactions)

            von_Mises = Function(xs.V_vm)
            von_Mises.interpolate(xs.von_Mises_expr)

            vm_list.append(von_Mises)
        
        return vm_list


    def _ensure_section_vjp_forms_compiled(self, collision=(0, 1)):
        """
        Compile-once helper for all UFL derivative forms needed by the coupled
        constitutive VJP.

        This should be called lazily and never inside a tight loop more than once.
        """
        # Foreground XS constitutive derivative forms
        for xs in self.XSs:
            if not hasattr(xs, "dKdx_form"):
                xs._set_up_dK_forms()

        # Mortar constitutive derivative forms are only needed when the
        # overlap-correction constitutive term is active.
        if self.enable_overlap_correction:
            mortar_xs = self.collisions[collision].mortar_xs
            if not hasattr(mortar_xs, "K1_form"):
                mortar_xs._compute_xs_stiffness_matrix()
            if not hasattr(mortar_xs, "dKdx_form"):
                mortar_xs._set_up_dK_forms()

        # Coupling / interpolation VJP forms
        if not hasattr(self.collisions[collision], "pMCpx_form"):
            self._compile_coupling_vjp_forms(collision)


    def _compute_global_section_adjoint_weights(self, dK):
        """
        For K = K1 @ K2^{-1} @ K1.T, return reverse weights W_1, W_2 such that

            <dK, dK_total> = <W_1, dK1_total> - <W_2, dK2_total>

        under the Frobenius inner product.
        """
        K1 = self.K1
        K2inv = self.K2inv

        W_1 = dK @ K1 @ K2inv + dK.T @ K1 @ K2inv
        W_2 = K2inv @ K1.T @ dK @ K1 @ K2inv

        return W_1, W_2


    def _apply_section_adjoint_weights(self, XS, W_1, W_2, derivative_type='x', scale=1.0):
        """
        Apply the *global* coupled constitutive adjoint weights to a target
        CrossSection's local K1/K2 forms.

        Parameters
        ----------
        XS : CrossSection
            Section whose K1_form / K2_form contribution is being differentiated.
        W_1, W_2 : (6,6) ndarray
            Global coupled constitutive adjoint weights.
        derivative_type : str
            'x' or 'w'
        scale : float
            Optional scaling, e.g. -0.5 for overlap-correction terms.

        Returns
        -------
        derivative_type == 'x' :
            ndarray of shape (n_x_dofs,)
        derivative_type == 'w' :
            ndarray of shape (n_w_dofs, 6)
        """
        XS.W_1.value = scale * W_1
        XS.W_2.value = scale * W_2

        if derivative_type == 'x':
            vec = fem.petsc.assemble_vector(XS.dKdx_form)
            return vec.array.copy()

        elif derivative_type == 'w':
            n_w = XS.V_w.dofmap.index_map_bs * XS.V_w.dofmap.index_map.size_global
            out = np.zeros((n_w, 6))
            for k in range(6):
                out[:, k] = fem.petsc.assemble_vector(XS.dKdw_form[k]).array
            return out

        else:
            raise ValueError(f"Unsupported derivative_type '{derivative_type}'")


    def _save_mortar_warping_state(self, collision=(0, 1)):
        mortar_xs = self.collisions[collision].mortar_xs
        return [wf.x.array.copy() for wf in mortar_xs.warping_functions]


    def _restore_mortar_warping_state(self, state, collision=(0, 1)):
        mortar_xs = self.collisions[collision].mortar_xs
        for k in range(6):
            mortar_xs.warping_functions[k].x.array[:] = state[k]


    def _load_projected_mortar_warping(self, collision=(0, 1), mesh_id=0):
        """
        Load mortar_xs.warping_functions with projected foreground warping state.

        mesh_id = 0 -> use PA and foreground A
        mesh_id = 1 -> use PB and foreground B

        Returns
        -------
        P_np : ndarray
            Dense interpolation matrix used in the projection.
        w_fg : ndarray
            Foreground warping state with shape (n_fg_dofs, 6)
        """
        mortar_xs = self.collisions[collision].mortar_xs

        if mesh_id == 0:
            P = self.collisions[collision].PA
            xs_fg = self.XSs[collision[0]]
        elif mesh_id == 1:
            P = self.collisions[collision].PB
            xs_fg = self.XSs[collision[1]]
        else:
            raise ValueError("mesh_id must be 0 or 1")

        w_fg = np.column_stack([xs_fg.warping_functions[k].x.array.copy() for k in range(6)])

        for k in range(6):
            P.mult(
                xs_fg.warping_functions[k].x.petsc_vec,
                mortar_xs.warping_functions[k].x.petsc_vec
            )

        return convert_petsc_to_numpy(P), w_fg


    def compute_section_vjp(self, dK, collision=(0, 1)):
        """
        Reverse-mode VJP for the explicit coupled constitutive map

            K = K(x_A, x_B, x_C, w_A, w_B)

        evaluated at the current state already loaded into self.

        Assumptions
        -----------
        - Foreground and mortar geometries are already updated to the current inputs.
        - Foreground warping states w_A and w_B are already updated to the current inputs.
        - Current interpolation operators PA/PB correspond to the current geometry.
        - self._compute_xs_stiffness_matrix() has already been run for this same state.

        Returns
        -------
        dict with keys:
            'x_A', 'x_B', 'x_C' : full V_x derivative vectors
            'w_A', 'w_B'        : arrays of shape (n_w_dofs, 6)
        """
        assert collision == (0, 1), "Current implementation assumes a single collision (0,1)."

        self._ensure_section_vjp_forms_compiled(collision)

        xs_A = self.XSs[collision[0]]
        xs_B = self.XSs[collision[1]]
        mortar = self.collisions[collision].mortar_mesh
        mortar_xs = self.collisions[collision].mortar_xs

        # Use current coupled K1/K2
        W_1, W_2 = self._compute_global_section_adjoint_weights(dK)

        # Allocate accumulators
        nXA = xs_A.V_x.dofmap.index_map_bs * xs_A.V_x.dofmap.index_map.size_global
        nXB = xs_B.V_x.dofmap.index_map_bs * xs_B.V_x.dofmap.index_map.size_global
        nXC = mortar.V_x.dofmap.index_map_bs * mortar.V_x.dofmap.index_map.size_global

        nWA = xs_A.V_w.dofmap.index_map_bs * xs_A.V_w.dofmap.index_map.size_global
        nWB = xs_B.V_w.dofmap.index_map_bs * xs_B.V_w.dofmap.index_map.size_global

        dRdx_A = np.zeros(nXA)
        dRdx_B = np.zeros(nXB)
        dRdx_C = np.zeros(nXC)

        dRdw_A = np.zeros((nWA, 6))
        dRdw_B = np.zeros((nWB, 6))

        # ==========================================================
        # 1) Direct foreground contributions
        # ==========================================================
        dRdx_A += self._apply_section_adjoint_weights(xs_A, W_1, W_2, derivative_type='x', scale=1.0)
        dRdw_A += self._apply_section_adjoint_weights(xs_A, W_1, W_2, derivative_type='w', scale=1.0)

        dRdx_B += self._apply_section_adjoint_weights(xs_B, W_1, W_2, derivative_type='x', scale=1.0)
        dRdw_B += self._apply_section_adjoint_weights(xs_B, W_1, W_2, derivative_type='w', scale=1.0)

        if not self.enable_overlap_correction:
            return {
                'x_A': dRdx_A,
                'x_B': dRdx_B,
                'x_C': dRdx_C,
                'w_A': dRdw_A,
                'w_B': dRdw_B,
            }

        # Save mortar warping state because we overwrite it twice below
        mortar_state = self._save_mortar_warping_state(collision)

        try:
            # ==========================================================
            # 2) A-side overlap correction:
            #
            #   -0.5 * K_C( x_C, P_A w_A )
            # ==========================================================
            P_A_np, w_A = self._load_projected_mortar_warping(collision=collision, mesh_id=0)

            # Recompute mortar constitutive quantities at projected A-state
            mortar_xs._compute_xs_stiffness_matrix()

            # Direct constitutive sensitivities wrt mortar geometry / mortar warping
            dRdx_C_A_direct = self._apply_section_adjoint_weights(
                mortar_xs, W_1, W_2, derivative_type='x', scale=-0.5
            )
            dRdw_C_A = self._apply_section_adjoint_weights(
                mortar_xs, W_1, W_2, derivative_type='w', scale=-0.5
            )

            dRdx_C += dRdx_C_A_direct

            # Chain rule through w_C^A = P_A w_A
            seed_P_A = np.zeros_like(P_A_np)
            for k in range(6):
                dRdw_A[:, k] += P_A_np.T @ dRdw_C_A[:, k]
                seed_P_A += np.outer(dRdw_C_A[:, k], w_A[:, k])

            # Projection-operator geometry VJP
            dPdx_A, dPdx_C_from_A = self._compute_vjp_dP(seed_P_A, collision, mesh_id=0)
            dRdx_A += dPdx_A
            dRdx_C += dPdx_C_from_A

            # ==========================================================
            # 3) B-side overlap correction:
            #
            #   -0.5 * K_C( x_C, P_B w_B )
            # ==========================================================
            P_B_np, w_B = self._load_projected_mortar_warping(collision=collision, mesh_id=1)

            mortar_xs._compute_xs_stiffness_matrix()

            dRdx_C_B_direct = self._apply_section_adjoint_weights(
                mortar_xs, W_1, W_2, derivative_type='x', scale=-0.5
            )
            dRdw_C_B = self._apply_section_adjoint_weights(
                mortar_xs, W_1, W_2, derivative_type='w', scale=-0.5
            )

            dRdx_C += dRdx_C_B_direct

            seed_P_B = np.zeros_like(P_B_np)
            for k in range(6):
                dRdw_B[:, k] += P_B_np.T @ dRdw_C_B[:, k]
                seed_P_B += np.outer(dRdw_C_B[:, k], w_B[:, k])

            dPdx_B, dPdx_C_from_B = self._compute_vjp_dP(seed_P_B, collision, mesh_id=1)
            dRdx_B += dPdx_B
            dRdx_C += dPdx_C_from_B

        finally:
            self._restore_mortar_warping_state(mortar_state, collision)

        return {
            'x_A': dRdx_A,
            'x_B': dRdx_B,
            'x_C': dRdx_C,
            'w_A': dRdw_A,
            'w_B': dRdw_B,
        }


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
        
