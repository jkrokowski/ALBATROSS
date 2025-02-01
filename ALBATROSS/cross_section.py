from ufl import (Argument,derivative,dot,cross,Identity,sqrt,inner,tr,variable,
                 diff,grad,sin,cos,as_matrix,SpatialCoordinate,FacetNormal,
                 Measure,as_tensor,indices,
                 TrialFunction,TestFunction,split)
from basix.ufl import element,mixed_element
from dolfinx.fem import (Constant,Expression,assemble_scalar,form,Function,
                         functionspace,assemble_vector,petsc)
from dolfinx import fem
import numpy as np
from petsc4py import PETSc
from dolfinx.mesh import locate_entities_boundary
from dolfinx import geometry # import compute_collisions_trees
from scipy.sparse.linalg import inv,lsqr,spsolve
import sparseqr
from scipy.sparse import csr_matrix
import ufl 
import pyvista
from dolfinx import plot

from ALBATROSS.material import getMatConstitutiveIsotropic
from ALBATROSS.utils import plot_xdmf_mesh,get_vtx_to_dofs,sparseify
from ALBATROSS.nonmatching_utils import (Region,Separation,Collision,
                                         get_bbtrees,get_collision_celltags,
                                         pts_to_dofs,get_petsc_system,
                                         celltags_to_dofs,
                                         get_interpolation_matrix,
                                         convert_petsc_to_numpy,
                                         get_points_from_cells)
default_scalar_type = PETSc.ScalarType    

#TODO: allow user to specify a point to find xs props about
#TODO: provide a method to translate between different xs values?
#TODO: update sensitivities plotting for higher order basis functions
#TODO: sparse multiply for QR decomposition

class CrossSection:
    def __init__(self, msh, materials ,celltags=None,verbose=False):
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
        self.degree = 1

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
        if self.ct is not None:
            #check that the number and values of celltags match those specified in the material objects
            mesh_ct = np.unique(self.ct.values)
            mat_ct = np.unique([self.materials[_i].id for _i in range(self.num_mat)] )
            assert(np.logical_and.reduce(mesh_ct==mat_ct))
            
            #material property functions
            self.Q = functionspace(self.msh,('DG',0))
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

        elif self.ct is None:
            self.dx = Measure("dx",domain=self.msh)
        #     for material in self.materials:
        #         print(self.ct.find(material.id))
        #         material_facets=meshtags(self.msh,self.tdim,self.ct.find(material.id),material.id)
        #         material.dx = Measure("dx",domain=self.msh,subdomain_data=material_facets)
            self.C = getMatConstitutiveIsotropic(self.msh,self.materials[0].E,self.materials[0].nu)
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
        #TODO: compute density weight areas and areas of each subdomain?
        
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
        self._construct_residual()

        if self.verbose:
            print('Assembling System Matrix....')   
        self._assemble_system_matrix()

        if self.verbose:
            print('Computing non-trivial solutions....')
        self._get_modes()

        if self.verbose:
            print('Orthogonalizing w.r.t. elastic modes...')
        self._decouple_modes()
        self._build_elastic_solution_modes()
        
        if self.verbose:
            print('Computing Beam Constitutive Matrix....')
        self._compute_xs_stiffness_matrix()

        print("DONE computing Beam Constitutive Matrix") 


    def get_xs_stiffness_matrix_EB(self):
        
        #construct material constitutive tensor field
        # self.constructConstitutiveField()

        if self.verbose:
            print('Constructing Cross-Section System...')
        self._construct_residual()

        if self.verbose:
            print('Assembling System Matrix....')   
        self._assemble_system_matrix()

        if self.verbose:
            print('Computing non-trivial solutions....')
        self._get_modes()

        if self.verbose:
            print('Orthogonalizing w.r.t. elastic modes...')
        self._decouple_modes()
        self._build_elastic_solution_modes_EB()
        
        if self.verbose:
            print('Computing Beam Constitutive Matrix....')
        self._compute_xs_stiffness_matrix_EB()

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
        #rotation about Z-axis
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

    def _construct_residual(self,dx=None,return_residual=False):

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
        L1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
            + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
            - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
            - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \
            # + Tbar
        
        # # equation 4,5,6
        L2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
            + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
            - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
            - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \
            # + That

        # equation 7,8,9
        L3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
            - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
            - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\
            # + Ttilde

        #equation 10,11,12
        L4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\
            # + Tbreve
        
        #construct residual
        residual = L1+L2+L3+L4

        if return_residual is False:
            self.Residual = residual
        else:
            return residual

    def _assemble_system_matrix(self,residual = None):
        if residual is None:
            self.system_mat = petsc.assemble_matrix(form(self.Residual))
            self.system_mat.assemble()
        else:
            system_mat = petsc.assemble_matrix(form(residual))
            system_mat.assemble()
            return system_mat

    def _get_modes(self):
        
        m,n1=self.system_mat.getSize()
        if self.verbose:
            print('Computing QR factorization')
        Acsr = csr_matrix(self.system_mat.getValuesCSR()[::-1], shape=self.system_mat.size)
        
        #perform QR factorization and store as struct in householder form
        QR= sparseqr.qr_factorize( Acsr.transpose() )

        #build matrix of unit vectors for selecting last 12 columns
        X = np.zeros((m,12))
        for i in range(12):
            X[m-1-i,11-i]=1

        #perform matrix multiplication implicitly to construct orthogonal nullspace basis
        self.sols = sparseqr.qmult(QR,X)
        # Q,_ = np.linalg.qr(Acsr.transpose().toarray())
        # self.sols  = Q[:,-12:]
        self.sparse_sols = sparseify(self.sols,sparse_format='csc')
        # self.sols = self.sparse_sols.toarray()

        #TODO: maybe instead of storing the sols as a numpy array, i should initialize the warping function space here and directly populate
        #       It may be easier to get values in and out....

    def _decouple_modes(self,basis_matrix_only=False):
        #this is a change of basis operation from the standard R^12 basis to
        #   the basis defined by the 6 rigid body modes and the 6 elastic modes
        #
        #the change of basis matrix can be easily computed by simply evaluating
        #   the functions defining the rigid+elastic basis at all the dofs
        x = self.x
        dx = self.dx
        C = self.C
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B

        # get collapsed subspace and maps from subspaces to parent space 
        UBAR,self.ubar_vtx_to_dof = self.V.sub(0).collapse()
        UHAT,self.uhat_vtx_to_dof = self.V.sub(1).collapse()
        # self.ubar_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(0)).flatten()
        # self.uhat_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(1)).flatten()
        # self.utilde_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(2))
        # self.ubreve_vtx_to_dof = get_vtx_to_dofs(self.msh,self.V.sub(3))

        #GET UBAR AND UHAT RELATED MODES
        ubar_modes = self.sols[self.ubar_vtx_to_dof,:]
        uhat_modes = self.sols[self.uhat_vtx_to_dof,:]

        #CONSTRUCT FUNCTION FOR UBAR AND UHAT SOLUTIONS GIVEN EACH MODE
        # UBAR = self.V.sub(0).collapse()[0]
        # UHAT = self.V.sub(1).collapse()[0]
        ubar_mode = Function(UBAR)
        uhat_mode = Function(UHAT)
        vbar_mode = TrialFunction(UBAR)

        # Sketch of a newer approach:
        #We can construct the warping functions in a much more intelligent way, by using the approach of the
        # self._orthonormalize_rbd() to return the 6 elastic warping modes directly instead of looping through
        # the existing modes and explicitly constructing the warping functions. 
        # The key here is set of expression that describe the warping 

        #INITIALIZE DECOUPLING MATRIX (12X12)
        mat = np.zeros((6,12))

        #HERES THE NEW APPROACH:
        #what we want is the set of warping functions Nbar and Nhat
        # the other warping functions have no effect on the beam stiffness matrix or sensitivities
        # so we'll first extract ubar and uhat
        # then we'll use the gram-schmidt process to factor out the rigid body modes from ubar
        # rigid body translation and displacement only affect ubar, no other warping function
        # so... we can orthogonalize ubar and explicitly construct a reduced basis transformation matrix M_e
        # that only considers the elastic modes, which we can decouple with a 6x6 matrix in the same manner as below
                
        #LOOP THROUGH MAT'S COLUMN (EACH MODE IS A COLUMN OF MAT):
        for mode in range(mat.shape[1]):
            #construct function from mode
            ubar_mode.vector.array = ubar_modes[:,mode]
            uhat_mode.vector.array = uhat_modes[:,mode]

            #filter rigid body modes out using GS as these are just a function of ubar
            # ubar_mode = self._orthonormalize_rbm(ubar_mode)
            # uhat_mode = self._orthonormalize_rbm(uhat_mode)

            #TODO: cannot just update the ubar values without accounting for how this affects the 
            # properties of the sols matrix. 
            # self.sols[self.ubar_vtx_to_dof,mode] = ubar_mode.vector.array
            # self.sols[self.uhat_vtx_to_dof,mode] = uhat_mode.vector.array
            
            #get stress from warping functions
            # sigma_avg = self.warping2stress(ubar_mode_avg,uhat_mode)
            sigma = self.warping2stress(ubar_mode,uhat_mode)

            #relevant components of stress tensor
            sigma11 = sigma[0,0]
            sigma12 = sigma[1,0]
            sigma13 = sigma[2,0]

            #integrate stresses over cross-section at "root" of beam and construct xs load vector
            P1 = assemble_scalar(form(sigma11*dx))
            V2 = assemble_scalar(form(sigma12*dx))
            V3 = assemble_scalar(form(sigma13*dx))
            
            T1 = assemble_scalar(form( (((x[0])*(sigma13)) - ((x[1])*(sigma12)))*dx))
            M2 = assemble_scalar(form((x[1])*(sigma11)*dx))          
            M3 = assemble_scalar(form(-(x[0])*(sigma11)*dx))  
            
            #THIRD THREE ROWS: AVERAGE FORCE (COMPUTED WITH UBAR AND UHAT)
            mat[0,mode]=P1
            mat[1,mode]=V2
            mat[2,mode]=V3   

            #FOURTH THREE ROWS: AVERAGE MOMENTS (COMPUTED WITH UBAR AND UHAT)
            mat[3,mode]=T1
            mat[4,mode]=M2
            mat[5,mode]=M3
        

        # for i in range(6):
        #     for j in range(6):
        #         print(f"Dot product of mode {i} and mode {j}: {np.dot(mat[i, :], mat[j, :])}")

        #normalize the orthogonal rows and transpose to get the decoupling matrix
        self.mat = mat
        # self.mat = (mat.T/np.linalg.norm(mat,axis=1)).T
        # print('--------------------------')
        # print('basis transformation matrix after normalization:')
        # print('--------------------------')
        # for i in range(6):
        #     for j in range(6):
        #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.mat[i, :], self.mat[j, :])}")
        # print('--------------------------')
        # print('sols before basis transformation:')
        # print('--------------------------')
        # print("CHECK Normalization")
        # for i in range(6):
        #     print(np.linalg.norm(self.sols[:,i]))

        # print("Check orthogonality:")
        # for i in range(12):
        #     for j in range(12):
        #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.sols[:,i], self.sols[:,j])}")


        # Q,_ = np.linalg.qr(mat.T,mode='complete')
        # self.mat = Q[-6:,:]
        # print('---------------')
        # for i in range(6):
        #     for j in range(6):
        #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.mat[i, :], self.mat[j, :])}")

        if basis_matrix_only is False:
            mat_sparse = sparseify(self.mat,sparse_format='csc')

            # self.sols_decoup = (self.sparse_sols.dot(inv(mat_sparse))).toarray()
            # self.sols_decoup = self.sols@np.linalg.inv(mat)
            # self.sols_decoup = self.sols@self.mat.T
            # ubar_uhat_dofs = np.concatenate([self.ubar_vtx_to_dof,self.uhat_vtx_to_dof])
            # sparse_sols = sparseify(self.sols[ubar_uhat_dofs,:])
            # # # self.sols_decoup = self.sols[ubar_uhat_dofs,:]@self.mat.T
            # self.sols_decoup = sparse_sols.dot(mat_sparse.T).toarray()

            ubar_uhat_dofs = np.concatenate([self.ubar_vtx_to_dof,self.uhat_vtx_to_dof])
            # # self.sols_decoup = self.sols[ubar_uhat_dofs,:]@self.mat.T
            # self.sols_decoup = (self.sparse_sols.dot(mat_sparse.T).toarray())[ubar_uhat_dofs,:]
            # self.sols_decoup = (self.sparse_sols.dot(mat_sparse.T).toarray())

            #USING PSEUDOINVERSE
            mat_pinv = sparseify(np.linalg.pinv(mat_sparse.toarray()))
            self.sols_decoup = self.sparse_sols.dot(mat_pinv).toarray()
            # self.sols_decoup=mat@self.sols

            # print('--------------------------')
            # print('sols after basis transformation:')
            # print('--------------------------')
            # print("CHECK Normalization")
            # for i in range(6):
            #     print(np.linalg.norm(self.sols_decoup[:,i]))

            # print("Check orthogonality:")
            # for i in range(6):
            #     for j in range(6):
            #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.sols_decoup[:,i], self.sols_decoup[:,j])}")
            # #TODO: think about how and why to store some portion of the sols
            # # ubar_modes = self.sols_decoup[:len(self.ubar_vtx_to_dof),:]
            
            # for mode in range(6):
            #     #construct function from mode
            #     # ubar_mode.vector.array = self.sols_decoup[:len(self.ubar_vtx_to_dof),mode]
            #     ubar_mode.vector.array = self.sols_decoup[self.ubar_vtx_to_dof,mode]
            #     # uhat_mode.vector.array = uhat_modes[:,mode]

            #     #filter rigid body modes out using GS as these are just a function of ubar
            #     ubar_mode = self._orthonormalize_rbm(ubar_mode)
            #     # uhat_mode = self._orthonormalize_rbm(uhat_mode)

            #     #update decoupled solutions with the rigid body modes removed
            #     self.sols_decoup[self.ubar_vtx_to_dof,mode] = ubar_mode.vector.array
            #     # self.sols_decoup[self.uhat_vtx_to_dof,mode] = uhat_mode.vector.array
            

            # print('--------------------------')
            # print('sols after RBM removal:')
            # print('--------------------------')
            # print("CHECK Normalization")
            # for i in range(6):
            #     print(np.linalg.norm(self.sols_decoup[:,i]))

            # print("Check orthogonality:")
            # for i in range(6):
            #     for j in range(6):
            #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.sols_decoup[:,i], self.sols_decoup[:,j])}")

            # print()

            # self.sols_decoup = self.sols_decoup/np.linalg.norm(self.sols_decoup,axis=0)

            # #NEED TO RENORMALIZE THE WARPING FUNCTIONS
            # print('--------------------------')
            # print('sols after renormalization:')
            # print('--------------------------')
            # print("CHECK Normalization")
            # for i in range(6):
            #     print(np.linalg.norm(self.sols_decoup[:,i]))

            # print("Check orthogonality:")
            # for i in range(6):
            #     for j in range(6):
            #         print(f"Dot product of mode {i} and mode {j}: {np.dot(self.sols_decoup[:,i], self.sols_decoup[:,j])}")

            # print()


            #USING LSQR:
            # self.sols_decoup2 = lsqr(mat_sparse.T,sparse_sols.T).T
            # self.sols_decoup = sparseify(np.linalg.lstsq(mat_sparse.T.toarray(),sparse_sols.T.toarray())[0].T).toarray()

            # from scipy.sparse.linalg import norm
            # diff = self.sols_decoup-self.sols_decoup2
            # diff_norm = norm(diff)


    def _build_elastic_solution_modes(self):
        #Initialize a tensor element and mixed tensor function space 
        # for the elastic solution modes
        Ne = element('CG',self.msh.topology.cell_name(),self.degree,shape=(3,6))
        self.N_space = functionspace(self.msh,mixed_element(2*[Ne]))
        self.N = Function(self.N_space)
        
        #extract portions of elastic solution mode function related to each warping fxn
        self.N_bar, self.N_hat = self.N.split() 

        #get map of function dofs 
        # N_bar_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(0)).flatten()
        # N_hat_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(1)).flatten()
        N_bar_vtx_to_dofs = self.N_space.sub(0).collapse()[1]
        N_hat_vtx_to_dofs = self.N_space.sub(1).collapse()[1]
        # N_tilde_vtx_to_dofs = self.N_space.sub(2).collapse()[1]
        # N_breve_vtx_to_dofs = self.N_space.sub().collapse()[1]

        #get separate elastic solution mode values
        N_bar_vals = sparseify(self.sols_decoup[self.ubar_vtx_to_dof,:]).toarray().flatten()
        N_hat_vals = sparseify(self.sols_decoup[self.uhat_vtx_to_dof,:]).toarray().flatten()
        

        # N_bar_vals = self.sols_decoup[self.ubar_vtx_to_dof,:].flatten()/np.linalg.norm(self.sols_decoup[self.ubar_vtx_to_dof,:])
        # N_hat_vals = self.sols_decoup[self.uhat_vtx_to_dof,:].flatten()/np.linalg.norm(self.sols_decoup[self.uhat_vtx_to_dof,:])

        #populate elastic solution modes to elastic solution mode function
        self.N_bar.vector.array[N_bar_vtx_to_dofs] = N_bar_vals
        self.N_hat.vector.array[N_hat_vtx_to_dofs] = N_hat_vals

    # def _build_elastic_solution_modes(self):
    #     #Initialize a tensor element and mixed tensor function space 
    #     # for the elastic solution modes
    #     Ne = element('CG',self.msh.topology.cell_name(),self.degree,shape=(3,6))
    #     self.N_space = functionspace(self.msh,mixed_element(4*[Ne]))
    #     self.N = Function(self.N_space)
        
    #     #extract portions of elastic solution mode function related to each warping fxn
    #     self.N_bar, self.N_hat, self.N_tilde, self.N_breve = self.N.split() 

    #     #unpack elastic solution modes
    #     elastic_sols = self.sols_decoup[:,6:]

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

    def _compute_xs_stiffness_matrix(self):             
        #unpacking values
        x = self.x
        dx = self.dx
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
   
        #elastic solution mode function related to each warping fxn
        N_bar = self.N_bar
        N_hat = self.N_hat
        # N_tilde = self.N_tilde
        # N_breve = self.N_breve 

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
        P1 =sigma11_c*dx
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
        self.K1_form = [[diff(P[idx1],c[idx2]) for idx2 in range(6)] 
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
        
        self.K1 = sparseify(self.K1).toarray()
        self.K2 = sparseify(self.K2).toarray()
        
        #store K1^-1 for recovery and sensitivity computation
        self.K1inv = np.linalg.inv(self.K1)
        self.K1inv = sparseify(self.K1inv).toarray()

        #stor K2^-1 for sensitivity computation
        self.K2inv = np.linalg.inv(self.K2)
        self.K2inv = sparseify(self.K2inv).toarray()
        
        #compute Flexibility matrix
        # self.S = self.K1inv.T@self.K2@self.K1inv
        # self.S = sparseify(self.S).toarray()
        self.S = self.K2
        
        #invert Flexibility matrix to find beam constitutive matrix
        # self.K = np.linalg.inv(self.S)
        # self.K = sparseify(self.K).toarray()

        #an alternative approach to avoid multiple inversion of products of inversions
        # self.K = self.K1@sparseify(self.K2inv).toarray()@self.K1.T
        # self.K = sparseify(self.K).toarray()
        self.K = self.K2inv


    def _build_elastic_solution_modes_EB(self):
        #Initialize a tensor element and mixed tensor function space 
        # for the elastic solution modes
        Ne = element('CG',self.msh.topology.cell_name(),self.degree,shape=(3,4))
        self.N_space = functionspace(self.msh,mixed_element(4*[Ne]))
        self.N = Function(self.N_space)
        
        #extract portions of elastic solution mode function related to each warping fxn
        self.N_bar, self.N_hat, self.N_tilde, self.N_breve = self.N.split() 

        #unpack elastic solution modes
        elastic_sols = np.concatenate([self.sols_decoup[:,6:7],self.sols_decoup[:,9:]],axis=1)

        #get map of function dofs 
        N_bar_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(0))
        N_hat_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(1))
        N_tilde_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(2))
        N_breve_vtx_to_dofs = get_vtx_to_dofs(self.msh,self.N_space.sub(3))

        #get separate elastic solution mode values
        N_bar_vals = elastic_sols[self.ubar_vtx_to_dof.flatten(),:]
        N_hat_vals = elastic_sols[self.uhat_vtx_to_dof.flatten(),:]
        N_tilde_vals = elastic_sols[self.utilde_vtx_to_dof.flatten(),:]
        N_breve_vals = elastic_sols[self.ubreve_vtx_to_dof.flatten(),:]

        #populate elastic solution modes to elastic solution mode function
        self.N_bar.vector.array[N_bar_vtx_to_dofs.flatten()] = N_bar_vals.flatten()
        self.N_hat.vector.array[N_hat_vtx_to_dofs.flatten()] = N_hat_vals.flatten()
        self.N_tilde.vector.array[N_tilde_vtx_to_dofs.flatten()] = N_tilde_vals.flatten()
        self.N_breve.vector.array[N_breve_vtx_to_dofs.flatten()] = N_breve_vals.flatten()

    def _compute_xs_stiffness_matrix_EB(self):             
        #unpacking values
        x = self.x
        dx = self.dx
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.B
   
        #elastic solution mode function related to each warping fxn
        N_bar = self.N_bar
        N_hat = self.N_hat
        N_tilde = self.N_tilde
        N_breve = self.N_breve 

        #construct fenicsx variables pertaining to elastic solution modes
        c7 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        # c8 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        # c9 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c10 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c11 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c12 = variable(Constant(self.msh,PETSc.ScalarType((0.0))))
        c = as_tensor([c7,c10,c11,c12])

        #construct general warping displacement functions in terms of the 
        #   elastic solution modes and elastic solution mode coefficients
        ubar_c = dot(N_bar,c)
        uhat_c = dot(N_hat,c)
        utilde_c = dot(N_tilde,c)
        ubreve_c = dot(N_breve,c)

        #these elastic solution modes are related by the general expression 
        # for the displacement as:
        # u_c = ubar_c + uhat_c * x1 + utilde_c * x1**2 + ubreve_c * x1**3
        # wereh x1 is the beam axis direction
                
        # expressions for the stress and strain in terms of the polynomial 
        # from expansion above:
        eps_c = self.warping2strain(ubar_c,uhat_c,utilde_c,ubreve_c)
        sigma_c = self.warping2stress(ubar_c,uhat_c,utilde_c,ubreve_c)

        #only stresses with a 1x component are of concern:
        sigma11_c = sigma_c[0,0]
        sigma12_c = sigma_c[0,1]
        sigma13_c = sigma_c[0,2]

        #construct expression for the load applied to a cross-section in 
        # terms of stress and strain expressions defined based on  the 
        # polynomial expansion:
        P1 = sigma11_c*dx
        # V2 = sigma12_c*dx
        # V3 = sigma13_c*dx
        T1 = -((x[0])*sigma13_c - (x[1])*sigma12_c)*dx
        M2 = -(x[1])*sigma11_c*dx
        M3 = (x[0])*sigma11_c*dx

        #store loads in a list instead of a ufl vector as we cannot take 
        # variable derivatives of non-scalar forms
        P = [P1,T1,M2,M3]
        
        # construct expression for the internal energy of the beam based on
        # the polynomial expansion:
        Uc = 0.5*sigma_c[i,j]*eps_c[i,j]*dx

        #now we begin the differentiation, form construction, and form assembly
        # to get K1 & K2 as well as dK1dx & dK2dx (used for shape optimization)
        self.K1_form = [[diff(P[idx1],c[idx2]) for idx1 in range(4)] 
                        for idx2 in range(4)]
        self.K2_form = [[diff(diff(Uc,c[idx1]),c[idx2]) for idx1 in range(4)]
                        for idx2 in range(4)]
        
        self.K1 = np.array([[assemble_scalar(form(self.K1_form[idx1][idx2]))
                     for idx1 in range(4)] 
                        for idx2 in range(4)])
        self.K2 = np.array([[assemble_scalar(form(self.K2_form[idx1][idx2]))
                     for idx1 in range(4)] 
                        for idx2 in range(4)])
        
        #store K1^-1 for recovery and sensitivity computation
        self.K1inv = np.linalg.inv(self.K1)
        
        #compute Flexibility matrix
        self.S = self.K1inv.T@self.K2@self.K1inv
        
        #invert Flexibility matrix to find beam constitutive matrix
        self.K = np.linalg.inv(self.S)
    
    def compute_xs_stiffness_matrix_sensitivities(self):
        #TODO: combine EB and TS sensitivities...
        args = self.K1_form[0][0].arguments()
        n = max(a.number() for a in args) if args else -1
        du1 = Argument(self.VX,n+1)
        n = max(a.number() for a in args) if args else -1
        du2 = Argument(self.VX,n+1)
        # du = Argument(self.VX,0) #there are no arguments in any of these forms?
        self.dK1dx_form = [[derivative(self.K1_form[idx1][idx2],self.x,du1)
                            for idx2 in range(6)] 
                                for idx1 in range(6)]
        self.dK2dx_form = [[derivative(self.K2_form[idx1][idx2],self.x,du2)
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
        
        # #TODO: need to confirm that these einsums are computing what we want them to
        # #ARCHIVAL:
        # self.dK1invT = -np.einsum('ijk,jl->ilk',
        #                      self.K1inv.T @ self.dK1dx.transpose(1,0,2),
        #                        self.K1inv.T @ self.K2 @ self.K1inv ) 
        # #second term of dSdx
        # self.dK2 = np.einsum('ijk,jl->ilk',
        #                 self.K1inv.T@self.dK2dx,
        #                 self.K1inv)
        
        # #third term of dSdx
        # self.dK1inv = -np.einsum('ijk,jl->ilk',
        #                     self.K1inv.T @ self.K2 @ self.K1inv @ self.dK1dx,
        #                       self.K1inv)
        # #END ARCHIVAL

        #use chain rule for derivative of flexibility matrix dSdx:
        #first term of dSdx

        dK1invT_dK1dx= np.einsum('ij,kjl->ikl',
                                 self.K1inv.T,
                                   self.dK1dx)
        self.dK1invT = -np.einsum('ijk,jl->ilk',
                             dK1invT_dK1dx,
                               self.S ) 
        #second term of dSdx
        K1invT_dK2dx = np.einsum('ij,jkl->ikl',
                        self.K1inv.T,
                        self.dK2dx)
        self.dK2 = np.einsum('ijk,jl->ilk',
                        K1invT_dK2dx,
                        self.K1inv)
        
        #third term of dSdx
        S_dK1dx = np.einsum('ij,jkl->ikl',
                            self.S,
                            self.dK1dx)
        self.dK1inv = -np.einsum('ijk,jl->ilk',
                            S_dK1dx,
                              self.K1inv)
    
        #add terms to get dSdx
        self.dSdx = self.dK1invT + self.dK2 + self.dK1inv.transpose(1,0,2)
        # self.dSdx = self.dK1inv.transpose(1,0,2) + self.dK2 + self.dK1inv
        # self.dSdx = self.dK2 + 2*self.dK1inv
        # self.dSdx = -self.dK2 
        # self.dSdx = self.dK2 
        # self.dSdx = self.dK1inv
        # self.dSdx = self.dK1invT
        # self.dSdx = self.dK1dx
        # self.dSdx = self.dK2dx
        # self.dSdx = -self.dK1inv-self.dK1invT.transpose(1,0,2)
        # self.dSdx = 2*self.dK1invT

        # #APPROACH TO LIMIT MATRIX MULTIPLICATIONS:
        # #use chain rule for derivative of flexibility matrix dSdx:
        # #first term of dSdx
        # # self.dK1invT = -np.einsum('ijk,ij->ijk',
        # #                       self.dK1dx.transpose(1,0,2),
        # #                        self.K1inv.T @ self.K2 ) 
        # # self.dK1invT = -np.einsum('ijk,ij->ijk',
        # #                       self.dK1dx.transpose(1,0,2),
        # #                        self.K1inv.T )
        # # #second term of dSdx
        # # self.dK2 = np.einsum('ijk,ij->ijk',
        # #                 self.K1inv.T@self.dK2dx,
        # #                 self.K1inv)
        
        # #third term of dSdx
        # # self.dK1inv =  self.K2 @ self.K1inv @ self.dK1dx

        # self.dK1_term = self.K1inv @ self.dK1dx

        # self.K2dK1 = self.K2 @self.dK1_term
        # self.dK1TK2 = np.einsum('ijk,ij->ijk',
        #                       self.dK1_term.transpose(1,0,2),
        #                        self.K2 )

        # #add terms to get dSdx
        # # self.dSdx = np.einsum('ijk,ij->ijk',
        # #                       self.K1inv.T @ (- self.dK1invT + self.dK2dx - self.dK1inv ),
        # #                         self.K1inv)
        # # self.dSdx = np.einsum('ijk,ij->ijk',
        # #                       self.K1inv.T @ (- self.dK1inv.transpose(1,0,2) + self.dK2dx - self.dK1inv ),
        # #                         self.K1inv)
        # self.dSdx = np.einsum('ijk,ij->ijk',
        #                       self.K1inv.T @ (- self.dK1TK2 + self.dK2dx - self.K2dK1 ),
        #                         self.K1inv)

        #compute derivative of stiffness matrix (dKdx) from derivative of flexibility matrix (dSdx)
        # self.dKdx = - np.einsum('ijk,ij->ijk',
        #                         self.K @ self.dSdx,
        #                         self.K)
        K_dSdx = np.einsum('ij,jkl->ikl',
                           self.K,
                           self.dSdx)
        # self.dKdx = - np.einsum('ijk,jl->ilk',
        #                         K_dSdx,
        #                         self.K)

        #first term of dKdx
        dK1dxK2invK1T = np.einsum('ijl,jk->ikl',
                           self.dK1dx,
                           self.K2inv@self.K1.T)

        #second term of dKdx
        K1K2invdK2dx = np.einsum('ij,jkl->ikl',
                        self.K1@self.K2inv,
                        self.dK2dx)
        K1K2invdK2dxK2invK1T = np.einsum('ijk,jl->ilk',
                        K1K2invdK2dx,
                        self.K2inv@self.K1.T)
        
        #third term of dKdx
        K1K3invdK1dxT = np.einsum('ij,kjl->ikl',
                            self.K1@self.K2inv,
                            self.dK1dx)
        
        #add terms to get dSdx
        # self.dKdx = K1K2invdK2dxK2invK1T 
        # self.dKdx = dK1dxK2invK1T + K1K3invdK1dxT
        # self.dKdx = dK1dxK2invK1T + K1K2invdK2dxK2invK1T + K1K3invdK1dxT
        # self.dKdx =  K1K3invdK1dxT
        # self.dKdx = dK1dxK2invK1T + K1K2invdK2dxK2invK1T + K1K3invdK1dxT
        
        self.dKdx = np.einsum('ijk,ji->ijk',
                                self.K @ self.dK2dx,
                                self.K)
        # self.dKdx = self.dK2dx
        
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

    def compute_xs_stiffness_matrix_sensitivities_EB(self):
        args = self.K1_form[0][0].arguments()
        n = max(a.number() for a in args) if args else -1
        du = Argument(self.VX,n+1)
        # du = Argument(self.VX,0) #there are no arguments in any of these forms?

        m = 4
        self.dK1dx_form = [[derivative(self.K1_form[idx1][idx2],self.x,du)
                            for idx1 in range(m)] 
                                for idx2 in range(m)]
        self.dK2dx_form = [[derivative(self.K2_form[idx1][idx2],self.x,du)
                            for idx1 in range(m)] 
                                for idx2 in range(m)]
        self.dK1dx = np.array([[petsc.assemble_vector(form(self.dK1dx_form[idx1][idx2]))
                        for idx1 in range(m)] 
                            for idx2 in range(m)])     
        self.dK2dx = np.array([[petsc.assemble_vector(form(self.dK2dx_form[idx1][idx2]))
                for idx1 in range(m)] 
                    for idx2 in range(m)])
        
        #boundary dofs ([:,:,self.boundary_dofs])
        self.boundary_dofs = locate_entities_boundary(self.msh,0,lambda x: np.ones_like(x[0]))
        
        #use chain rule for derivative of flexibility matrix dSdx:
        #first term of dSdx
        self.dK1invT = -np.einsum('ijk,ij->ijk',
                             self.K1inv.T @ self.dK1dx.transpose(1,0,2),
                               self.K1inv.T @ self.K2 @ self.K1inv ) 
        #second term of dSdx
        self.dK2 = np.einsum('ijk,ij->ijk',
                        self.K1inv.T@self.dK2dx,
                        self.K1inv)
        
        #third term of dSdx
        self.dK1inv = -np.einsum('ijk,ij->ijk',
                            self.K1inv.T @ self.K2 @ self.K1inv @ self.dK1dx,
                              self.K1inv)

        #add terms to get dSdx
        self.dSdx = self.dK1invT + self.dK2 + self.dK1inv

        #compute derivative of stiffness matrix (dKdx) from derivative of flexibility matrix (dSdx)
        self.dKdx = - np.einsum('ijk,ij->ijk',
                                self.K @ self.dSdx,
                                self.K)
        
    def warping2strain(self,ubar,uhat):
        gradubar=grad(ubar)

        #derivatives of displacement
        #this is known from our displacement expression
        # dubxdx = uhat[0]
        # dubxdy = gradubar[0,0]
        # dubxdz = gradubar[0,1]
        # dubydx = uhat[1]
        # dubydy = gradubar[1,0]
        # dubydz = gradubar[1,1]
        # dubzdx = uhat[2]
        # dubzdy = gradubar[2,0]
        # dubzdz = gradubar[2,1]
        dubxdx = uhat[0]
        dubxdy = uhat[1]
        dubxdz = uhat[2]
        dubydx = gradubar[0,0]
        dubydy = gradubar[1,0]
        dubydz = gradubar[2,0]
        dubzdx = gradubar[0,1]
        dubzdy = gradubar[1,1]
        dubzdz = gradubar[2,1]

        #form ufl displacement for grad(u_i)
        gradu = as_tensor([[dubxdx,dubxdy,dubxdz],
                        [dubydx,dubydy,dubydz],
                        [dubzdx,dubzdy,dubzdz]])
        
        #ensure that strains are symmetric
        eps = 0.5 * (gradu + gradu.T)
        # eps = gradu

        return eps 

    def warping2stress(self,ubar,uhat):
        i,j,k,l=self.i,self.j,self.k,self.l
        eps = self.warping2strain(ubar,uhat)

        stress = as_tensor(self.C[i,j,k,l]*eps[k,l],(i,j))
        
        return stress 
    
    # def warping2loads(self,ubar,uhat):

    def recover_stress(self,reactions):
        c = self.K1inv@reactions

        c_const=Constant(self.msh,PETSc.ScalarType(c))
        ubar = dot(self.N_bar,c_const)
        uhat = dot(self.N_hat,c_const)
        utilde = dot(self.N_tilde,c_const)
        ubreve = dot(self.N_breve,c_const)

        stress = self.warping2stress(ubar,uhat,utilde,ubreve)
        return stress
        # V_stress = TensorFunctionSpace(self.msh, ("DG", 0),shape=(3,3))
        # stress_expr = Expression(stress, V_stress.element.interpolation_points())
        # stresses = Function(V_stress)
        # stresses.interpolate(stress_expr)
        # return stresses
    
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
    # def coeff_to_field(self,fxn,coeff,vtx_to_dof):
    #     #uses vtx_to_dof map to populate field with correct solution coefficients
    #     fxn.vector.array = coeff.flatten()[vtx_to_dof].flatten()
    #     fxn.vector.destroy

    # def construct_warping_fxns(self,u_c,N):
    #     #utility to populate data from decoupled modes to warping displacement functions
    #     Uc = u_c.function_space #fxn space associated with fxn

    #     #loop through subspaces and map values from nullspace to specified fxn values
    #     vtx_to_dofs = []
    #     for i in range(Uc.num_sub_spaces):
    #         vtx_to_dofs.append(get_vtx_to_dofs(self.msh,Uc.sub(0)))
    #         vtx_to_dofs_flat = vtx_to_dofs[i].flatten()
    #         u_c.vector.array[vtx_to_dofs_flat] = N.flatten()[vtx_to_dofs_flat]

    #     # ubar_c,uhat_c,utilde_c,ubreve_c = split(u_c)
    #     # ubar_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(0))
    #     # uhat_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(1))
    #     # utilde_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(2))
    #     # ubreve_c_vtx_to_dof = get_vtx_to_dofs(self.msh,Uc.sub(3))
   
    def plot_mesh(self):
        plot_xdmf_mesh(self.msh)

    def plot_warping_fxns(self,rigid=True,coup=False):
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()
        
        if rigid is True:
            if coup is True:
                elastic_sols = self.sols[:,:6]
            else:
                ubar_uhat_dofs = np.concatenate([self.ubar_vtx_to_dof,self.uhat_vtx_to_dof])
                elastic_sols = self.sols_decoup[ubar_uhat_dofs,:6]
        else:
            if coup is True:
                elastic_sols = self.sols[:,6:]
            else:
                elastic_sols = self.sols_decoup[:,6:]
        
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
            # topology, cell_types, geom = plot.vtk_mesh(self.msh, tdim)
            # grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))
            
            V0,V0_to_V = self.V.sub(0).collapse()
            topology, cell_types, geom = plot.vtk_mesh(V0)
            grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))
            
            c = np.zeros((6,1))
            c[i,:] = 1

            warping_sol = elastic_sols[:len(self.ubar_vtx_to_dof):,:]@c
            # ubar = Function(V0)
            # ubar.vector.array = warping_sol.flatten()
            solution_mode = warping_sol.reshape((geom.shape[0], 3))[:,[1,2,0]]
            grids[i][name]= solution_mode/np.max(np.linalg.norm(solution_mode,axis=1))
            # grids[i][name]= ubar.vector.array
            warped.append(grids[i].warp_by_vector(name,factor=.1))


            plotter.add_mesh(warped[i],show_edges=True,opacity=.9)
            plotter.add_mesh(grids[i],show_edges=True,opacity=.5,scalar_bar_args={'title': f'warping mode {i}'})
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
class CoupledXSProblem:
    '''class containing methods for gluing multiple overlapping, nonmatching meshes to 
        compute combined beam cross-sectional properties'''
    def __init__(self,XSs,pen=1e2):

        self.XSs = XSs

        self.regions = {i:Region(XS.msh,fxn_space=XS.V) for i,XS in enumerate(XSs)} 
        self.meshes = {i:XS.msh for i,XS in enumerate(XSs)}
        # self.form_construction = form_construction
        # self.bcs = {i:bc for i,bc in enumerate(bc_args)}
        self.pen = pen

        self.num_meshes = len(self.meshes)

        #compute collisions between all meshes
        self._find_overlap()

        
    def get_xs_stiffness_matrix(self):
        #assemble each region's system matrix
        self._assemble_system_mats()

        #apply the penalty parameter
        self._construct_coupled_system_matrix()

        #total system size:
        print("overall system size:")
        print(self.system_mat.getSize())

        #individual system sizes:
        print("overall system size:")
        for XS in self.XSs:
            print(XS.system_mat.getSize())

        #use QR factorization to get null modes:
        self._get_modes()
        print("null modes found!")
        
        #populate each individual cross-section's solution modes:
        # self._map_modes_to_region()


        #need to "decouple" the modes
        self._decouple_modes()
        
        #map elastic solutions to construct warping functions
        self._compute_xs_stiffness_matrix()
        
        #populate each individual region with the null mode corresponding


    def _find_overlap(self):
        '''
        Construct collision objects 
        '''
        bb_trees = get_bbtrees(list(self.meshes.values()))
        self.bb_trees = {i:bbtree for i,bbtree in zip(self.meshes.keys(),bb_trees)}

        #compute all collisions
        # TODO: some collision detection computational time can be saved by avoiding 
        # the collision detection on the inverse mesh combination with a non-overlapping section         
        collisions = {}
        separations = {}
        adjacency = np.zeros((self.num_meshes,self.num_meshes),dtype=int)
        for i in self.meshes.keys():
            collisions_i = {}
            separations_i = {}
            for j in self.meshes.keys():
                if i==j:
                    continue
                #get collisions 
                collisions_bbtree_ij = geometry.compute_collisions_trees(self.bb_trees[i], self.bb_trees[j])
                
                if collisions_bbtree_ij.size != 0:
                    #update the adjaceny matrix
                    adjacency[i,j] = 1

                    meshptsi = self.meshes[i].geometry.x
                    meshptsj = self.meshes[j].geometry.x

                    bbleaves_ij = geometry.compute_collisions_points(self.bb_trees[j],meshptsi)
                    bbleaves_ji = geometry.compute_collisions_points(self.bb_trees[i],meshptsj)
                    celltags_i,celltags_j = get_collision_celltags(self.meshes[i],self.meshes[j],collisions_bbtree_ij)
                    
                    #cells making up the collision zone
                    # cells_i = np.unique(collisions_bbtree_ij[:,0])
                    # cells_j = np.unique(collisions_bbtree_ij[:,1])

                    #points of the cells in collision zone
                    # pts_i = get_points_from_cells(self.regions[i].msh,cells_i)
                    # pts_j = get_points_from_cells(self.regions[j].msh,cells_j)
                    

                    # # Compute actual colliding cells for each input point
                    # potential_colliding_cells = geometry.compute_collisions_points(self.bb_trees[i], pts_j)
                    adj_list_ij = geometry.compute_colliding_cells(self.regions[j].msh, bbleaves_ij, meshptsi)
                    adj_list_ji = geometry.compute_colliding_cells(self.regions[i].msh, bbleaves_ji, meshptsj)

                    # Filter out points that actually intersect the mesh
                    pts_i = [i for i in range(len(meshptsi)) if len(adj_list_ij.links(i)) > 0]
                    pts_j = [i for i in range(len(meshptsj)) if len(adj_list_ji.links(i)) > 0]

                    #check if pts i are in mesh j
                    # collision_points_ij = geometry.compute_collisions_points(self.bb_trees[j],pts_i)
                    # collision_points_ji =geometry.compute_collisions_points(self.bb_trees[i],pts_j)

                    #create vertex-to-cell connectivity has been created if it hasn't been done yet
                    if self.regions[i].msh.topology.connectivity(0,2) is None:
                        self.regions[i].msh.topology.create_connectivity(0,2)
                    if self.regions[j].msh.topology.connectivity(0,2) is None:
                        self.regions[j].msh.topology.create_connectivity(0,2)
                    
                    #get the penalty dofs:
                    penalty_dofs_i=fem.locate_dofs_topological(self.regions[i].fxn_space,0,pts_i)
                    penalty_dofs_j=fem.locate_dofs_topological(self.regions[j].fxn_space,0,pts_j)

                    #TODO: just need to remove celltags for any 
                    #JJK 1/27/25: this is not returning the proper penalty dofs!
                    # penalty_dofs = pts_to_dofs(self.regions[i].fxn_space,collision_points_ij)

                    #JJK 1/29/25: this returns additional dofs outside of the 
                    #get the dofs correspondings to cells that are part of the overlap
                    # penalty_dofs_i = celltags_to_dofs(self.regions[i].fxn_space,celltags_i)
                    # penalty_dofs_j = celltags_to_dofs(self.regions[j].fxn_space,celltags_j)

                    #information about a collision of mesh i on mesh j
                    collision_ij = Collision(collisions_bbtree_ij,
                                            #  collision_points_ij,
                                             (celltags_i,celltags_j),
                                             (pts_i,pts_j),
                                             (penalty_dofs_i,penalty_dofs_j))

                    pen_vec = self._build_penalty_vector(self.regions[i],collision_ij)
                    
                    collision_ij.add_pen_vec(pen_vec)

                    collisions_i[j]=collision_ij

                elif collisions_bbtree_ij.size == 0:
                    separations_i[j]=Separation()
                    
            #add all collisions to dictionary list
            # TODO: JJK need to not add empty dictionaries     
            collisions[i] = collisions_i
            separations[i] = separations_i

        self.collisions = collisions
        self.separations = separations
        self.adjacency = adjacency
    
    
    def _build_penalty_vector(self,region_i,collision_ij):
        '''
        Build the vector of penalty terms per dof
        This is a PETSc Vector that can be directly multiplied by the interpolation matrix
        '''
        #initialize empty PETSc vector
        pen_vec = PETSc.Vec().create()
        vec_size = region_i.fxn_space.dofmap.index_map.size_global #* region_i.fxn_space.num_sub_spaces
        pen_vec.setSizes(vec_size)
        pen_vec.setFromOptions()

        #compute areas of each element in the overlapping subdomain using a DG0 space
        DG0 = functionspace(region_i.msh,("DG",0))
        v = ufl.TestFunction(DG0)
        # dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_id = 1, subdomain_data=collision_ij.celltags)
        dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_data=collision_ij.celltags[0])
        cell_area_form = form(v*dx_overlap(1))
        cell_areas = assemble_vector(cell_area_form)

        #create connectivity between cells and vertices (if not already created)
        region_i.msh.topology.create_connectivity(0,2)
        pen_values = np.zeros((len(collision_ij.penalty_dofs[0]),),dtype=float)
        #for each pt, update the penalty value for that vertex
        for i,pt in enumerate(collision_ij.pts[0]):
            #get the corresponding vertex for a specific dof
            # vtx = region_i.dof_to_vertex_map[dof]

            #get the cells connected to the penalty dof
            cells = region_i.msh.topology.connectivity(0,2).links(pt)

            dofs = fem.locate_dofs_topological(region_i.fxn_space,0,[pt])
            
            #add up area of all cells that are incident to the penalty dof
            #  adjust penalty proportionately to the supported area
            indices=np.where(np.isin(collision_ij.penalty_dofs[0],dofs))
            pen_values[indices] = self.pen * np.sum(cell_areas.array[cells])

        #populate the PETSc vector with the values at the proper indices
        for idx,val in zip(collision_ij.penalty_dofs[0],pen_values):
            pen_vec.setValue(idx,val)
        
        return pen_vec


    def _assemble_system_mats(self):
        #construct the residudal and assemble the system mat for each region
        for XS in self.XSs:
            XS._construct_residual()
            XS._assemble_system_matrix()

        #compile system matrices for each individual region into a list 
        #   accessible by the coupled problem class
        # system_mats = []
        offset = 0
        for i,region in zip(self.regions,self.regions.values()):
            region.system_mat = self.XSs[i].system_mat
            #store offset values for the computed 
            region.offset_start = offset
            offset += region.system_mat.getSize()[0]
            region.offset_end = offset
        #     system_mats.append(region.system_mat)
        # self.system_mats = system_mats


    def _construct_coupled_system_matrix(self):
        '''
        Given collisions and regions, 
        set up the coupled system matrix with the penalty terms
        '''
        #TODO: for each collision, subtract off half the assembled stiffness of the ovelapping section 
        # (use restricted integration measure and collision information to assemble these corrections) 

        # for each collision, compute the interpolation matrices and add the penalty terms to the corresponding dofs
        for idx,val in np.ndenumerate(self.adjacency):
            if val == 1:
                #get the interpolation matrix
                self.collisions[idx[0]][idx[1]].inter_mat = get_interpolation_matrix(self.regions[idx[1]].fxn_space,
                                                                                     self.regions[idx[0]].fxn_space,
                                                                                     mixed=True)
                #copy the dimensions of the interpolation matrix:
                self.collisions[idx[0]][idx[1]].pen_mat = self.collisions[idx[0]][idx[1]].inter_mat.duplicate()
                #prepopulate the penalty matrix term with the interpolation matrix
                # self.collisions[idx[0]][idx[1]].pen_mat.copy(self.collisions[idx[0]][idx[1]].inter_mat.duplicate())

            elif val == 0 and idx[0] != idx[1]:
                self.separations[idx[0]][idx[1]].mat.createAIJ([self.regions[idx[1]].system_mat.getSize()[0],
                                                                self.regions[idx[0]].system_mat.getSize()[1]])
                self.separations[idx[0]][idx[1]].mat.assemble()

        #populate an array of the same size as the adjacency matrix of the petsc matrices
        #  using a *nearly* incomprehensible list "comprehension" 
        # this adds the unadultered system to the diagonals, the interpolation matrices where there is a collision
        # and the assembled empty matrices where there is a "separation"
        A_list = [ [self.regions[i].system_mat if i==j
                    else self.separations[i][j].mat if self.adjacency[i][j] == 0 and i!=j
                    else self.collisions[i][j].pen_mat 
                        for i in range(self.num_meshes)]
                     for j in range(self.num_meshes) ]
        
        #assemble uncoupled nested matrix for debugging
        A_uncoupled_petsc = PETSc.Mat()
        A_uncoupled_petsc.createNest(A_list)
        A_uncoupled_petsc.assemble()
        A_aij = A_uncoupled_petsc.convert('aij')
        A_uncoupled = csr_matrix(A_aij.getValuesCSR()[::-1], shape=A_aij.size).toarray()
        print(f"uncoupled system zero body modes: {A_uncoupled.shape[0]-np.linalg.matrix_rank(A_uncoupled)}")
        
        #TODO: it seems that the penalty dofs and the penalty term are not matching up well
        #       there is definitely some bug here that needs some inspection
        # add the penalty to the relevant block of A_list
        for idx,val in np.ndenumerate(self.adjacency):
            # if idx[0]==idx[1]:
            if val==1:
                pen_term = PETSc.Mat().createAIJ(A_list[idx[0]][idx[0]].getSize())
                pen_term.assemble()
                # diag = pen_term.getDiagonal()
                #set diagonal values to the pre-computed penalty vector values
                # for val in self.collisions[idx[0]][idx[1]].penalty_dofs[0]:
                #     diag[val] = self.collisions[idx[0]][idx[1]].pen_vec[val]
                # pen_term.setDiagonal(diag)
                pen_term.setDiagonal(self.collisions[idx[0]][idx[1]].pen_vec)
                pen_term.assemble()
                # penalty_term = PETSc.Mat().createAIJ(I_mat.getSize())
                # I_mat.multTranspose(self.collisions[idx[0]][idx[1]].pen_vec,penalty_term)

                #add penalty term to diagonal block
                A_list[idx[0]][idx[0]].axpy(1.0,pen_term)

                #TODO: need to come up with a better way of populating the 
                #   nested list than simply filling with the interpolation matrix, then overwriting it...

                #add penalty term to off diagonal block                
                A_list[idx[0]][idx[1]] = pen_term.matMult(self.collisions[idx[1]][idx[0]].inter_mat)
                # A_list[idx[0]][idx[1]] = self.collisions[idx[1]][idx[0]].inter_mat
                A_list[idx[0]][idx[1]].assemble()
                A_list[idx[0]][idx[1]].scale(-1.0)

                # #apply the correction for the overlap
                # dx_correction = ufl.Measure("dx", 
                #                             domain=self.XSs[idx[0]].msh,
                #                             subdomain_data= self.collisions[idx[0]][idx[1]].celltags[0])
                # res = self.XSs[idx[0]]._construct_residual(dx=dx_correction,
                #                                            return_residual=True)
                # correction = self.XSs[idx[0]]._assemble_system_matrix(residual=res)
                # # correction.view()
                # A_list[idx[0]][idx[0]].axpy(-0.5,correction)


        A = PETSc.Mat()
        A.createNest(A_list)
        
        A.assemble()
        A_aij = A.convert('aij')
        A_coupled = csr_matrix(A_aij.getValuesCSR()[::-1], shape=A_aij.size).toarray()
        print(f"coupled system zero body modes: {A_coupled.shape[0]-np.linalg.matrix_rank(A_coupled)}")

        self.system_mat = A

    def _get_modes(self):
        m,n1=self.system_mat.getSize()
        print('Computing QR factorization')
        A_aij = self.system_mat.convert('aij')
        Acsr = csr_matrix(A_aij.getValuesCSR()[::-1], shape=self.system_mat.size)
        
        #perform QR factorization and store as struct in householder form
        QR= sparseqr.qr_factorize( Acsr.transpose() )

        #build matrix of unit vectors for selecting last 12 columns
        X = np.zeros((m,12))
        for i in range(12):
            X[m-1-i,11-i]=1

        #perform matrix multiplication implicitly to construct orthogonal nullspace basis
        self.sols = sparseqr.qmult(QR,X)
        self.sparse_sols = sparseify(self.sols,sparse_format='csc')
    
    #TODO: Decoupling routine at the xs level needs to be cleaned up a bit
    #       then, construct a decoupling matrix for the full system modes

    #       next, efficiently get the warping functions

    #       then to compute the stiffness matrix, we can compile K1_form and K2_form for each region
    #       K1 and K2 can be assembled and added together (as they both contribute to the loading and the internal energy)
    #       finally, K can be determined from the coupled system level K1 and K2 in the standard way
    
    def _decouple_modes(self):
        ''' 
        for each region, decouple the modes corresponding to that region
        '''
        #intialize empty basis transformation matrix
        self.basis_trans_matrix = np.zeros((6,12))

        #compute contribution to basis transformation matrix for each region
        for i,region in zip(self.regions,self.regions.values()):
            self.XSs[i].sols = self.sols[region.offset_start:region.offset_end,:]
            self.XSs[i]._decouple_modes(basis_matrix_only=True)
            self.basis_trans_matrix += self.XSs[i].mat
        #perform the basis transformation (use the sparse matrix to prevent numerical inaccuracies during inversion)
        # self.sols_decoup = self.sols@np.linalg.inv(self.basis_trans_matrix)
        self.basis_trans_matrix_sparse = sparseify(self.basis_trans_matrix)#,sparse_format='csc')
        
        self.basis_trans_matrix_pinv = sparseify(np.linalg.pinv(self.basis_trans_matrix_sparse.toarray()))

        self.sols_decoup = (self.sparse_sols.dot(self.basis_trans_matrix_pinv)).toarray()

        #get the decoupled basis
        for i,region in zip(self.regions,self.regions.values()):
            # ubar_uhat_dofs = np.concatenate([self.XSs[i].ubar_vtx_to_dof,self.XSs[i].uhat_vtx_to_dof])
            # self.XSs[i].sols_decoup = self.sols_decoup[region.offset_start:region.offset_end,:][ubar_uhat_dofs,:]
            self.XSs[i].sols_decoup = self.sols_decoup[region.offset_start:region.offset_end,:]


    def _compute_xs_stiffness_matrix(self):
        '''
        for each region, get the elastic solution modes
        '''
        self.K = np.zeros((6,6))
        # self.K1 = np.zeros((6,6))
        # self.K2 = np.zeros((6,6))
        # self.S = np.zeros((6,6))
        for i,region in zip(self.regions,self.regions.values()):
            self.XSs[i]._build_elastic_solution_modes()
            self.XSs[i]._compute_xs_stiffness_matrix()
            # self.S += self.XSs[i].S
            self.K += self.XSs[i].K
        #     self.K1 += self.XSs[i].K1
        #     self.K2 += self.XSs[i].K2
        # self.K1inv = np.linalg.inv(self.K1)
        
        # #compute Flexibility matrix
        # self.S = self.K1inv.T@self.K2@self.K1inv

        # #invert Flexibility matrix to find beam constitutive matrix
        # self.K = np.linalg.inv(self.S)

    def plot_warping_fxns(self):
        for xs in self.XSs:
            xs.plot_warping_fxns()

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
        