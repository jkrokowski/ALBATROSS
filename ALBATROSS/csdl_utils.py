import csdl_alpha as csdl
import ALBATROSS
# from scipy.spatial import cKDTree
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

class WarpingFunctionState(csdl.experimental.CustomImplicitOperation):
    '''
    inputs: nodal positions of cross-sectional mesh'''
    def __init__(self,xs,boundary_nodes,interior_nodes):
        super().__init__()
        self.xs = xs
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.w = self.create_output('w', (self.xs.V_w.dofmap.index_map.size_global,6))
        outputs.lmbda = self.create_output('lmbda', (self.xs.V_lm.value_size,6))

        return outputs
    
    def solve_residual_equations(self, inputs, outputs):
        print("solve residual equations:")
        mesh_geometry = self.xs.msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        #compute warping functions
        self.xs._get_warping_functions()
        
        outputs['w'] = np.vstack([self.xs.warping_functions[i].x.array for i in range(6)]).T
        outputs['lmbda'] = np.vstack([self.xs.lmbdas[i].x.array for i in range(6)]).T

        #return mesh geometry to original state:
        self.xs.msh.geometry.x[:] = mesh_geometry


    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        # print("apply_inverse_jacobian:")
        #TODO: do we need to update the inputs, etc (eg. does the mesh update need to happen here?)
        mesh_geometry = self.xs.msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        #update the warping functions
        for i in range(6):
            self.xs.warping_functions[i].x.array[:] = outputs['w'][:,i]
            self.xs.lmbdas[i].x.array[:] = outputs['lmbda'][:,i]    

        # for mode = rev:
        # d_outputs --> d_residuals
    
        if mode == 'rev':    
            # compute d_residuals = (dr_du^-1)*d_outputs

            # dr_du is simply the finite element stiffness matrix in this case
            # these are just the A00, A01, and A10 blocks of the assembled stiffness matrix
            # we can leverage the already existing ksp solver and compute the multMatTranspose() using petsc,
            # then we output these two terms to numpy matrices
            d_residuals['w'],d_residuals['lmbda'] = self.xs.apply_inverse_jacobian(d_outputs['w'],d_outputs['lmbda'])
            # #which does this under the hood: 
            #     d_outputs_petsc = stack(d_outputs['w'],d_outputs['lmbda'])
            #     self.xs.solver.solveTranspose(d_outputs_petsc,d_residuals_petsc)
            #     d_residuals_numpy = convert_petsc_to_numpy(d_residuals_petsc)
            #====
            # d_residuals['w'] =  d_residuals_numpy[w_slice]
            # d_residuals['lmbda']  = d_residuals_numpy[lmbda_slice]

            # d_residuals['w'] = drw_dw_inv @ d_outputs['w'] + drw_dl_inv @ d_outputs['lmbda']
            # d_residuals['lmbda'] = drl_dl_inv @ d_outputs['lmbda'] # + drl_dw_inv @ d_outputs['w'] <-- this term is = 
        
        #return mesh geometry to original state:
        self.xs.msh.geometry.x[:] = mesh_geometry


    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # print("compute vector-jacobian product:")
        #TODO: do we need to update the inputs, etc (eg. does the mesh update need to happen here?)
        mesh_geometry = self.xs.msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        #update the warping functions
        for i in range(6):
            self.xs.warping_functions[i].x.array[:] = outputs['w'][:,i]
            self.xs.lmbdas[i].x.array[:] = outputs['lmbda'][:,i]   

        # for mode = rev
        # d_residuals --> d_inputs
        if mode == 'rev':
            # compute d_input = (dr_dinput)*d_residuals
            #TODO: can also just return d_inputs as a numpy matrix here and prevent the memory overhead of converting to numpy,etc
            # dRwdx,dRldx = self.xs.compute_dRdx() #return numpy matrices 

            # d_inputs = dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']

            # d_inputs['xy'] = d_inputs['boundary']
            # d_inputs['xy_interior'] = d_inputs['interior']

            dRdx_dr = self.xs.compute_VJP(d_residuals['w'],d_residuals['lmbda'])

            #TODO: map to boundary or interior nodes
            d_inputs['xy'] = np.vstack([dRdx_dr[self.xs.dofs_x_boundary],
                                        dRdx_dr[self.xs.dofs_y_boundary]]).T
            d_inputs['xy_interior'] = np.vstack([dRdx_dr[self.xs.dofs_x_interior],
                                                 dRdx_dr[self.xs.dofs_y_interior]]).T
            
            # d_inputs['xy'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda'] )['boundary']
            # d_inputs['xy_interior'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']) ['interior']
        
        #return mesh geometry to original state:
        self.xs.msh.geometry.x[:] = mesh_geometry


class BeamMatrixFromWarping(csdl.CustomExplicitOperation):
    def __init__(self,xs,boundary_nodes=None,interior_nodes=None,check_partials='False'):
        super().__init__()
        self.xs = xs
        self.check_partials =check_partials
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes
        self.xs._set_up_dK_forms()
        self.xs._set_up_dA_form()

    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        if self.check_partials != 'w':
            self.declare_input('xy',inputs.xy)
            self.declare_input('xy_interior',inputs.xy_interior)
        if self.check_partials != 'x':
            self.declare_input('w',inputs.w)
            self.declare_input('lmbda',inputs.lmbda)
        
        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K', (6,6))
        outputs.K.name = 'beam stiffness matrix'
        outputs.A = self.create_output('A',(1,))
        outputs.A.name = 'beam xs area'

        return outputs
    
    def compute(self, inputs, outputs):
        print('compute beam matrix from warping function state')
        mesh_geometry = self.xs.msh.geometry.x.copy()

        if self.check_partials != 'w':
            #update boundary nodes:
            self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

            #update interior nodes
            self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']


        if self.check_partials != 'x':
            for i in range(6):
                self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
                self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]    

        
        # self.xs.plot_mesh()
        self.xs._compute_xs_stiffness_matrix()

        print('beam cross-sectional area:',self.xs.A)
        
        outputs['K'] = self.xs.K
        outputs['A'] = self.xs.A

        #return mesh geometry to original state:
        self.xs.msh.geometry.x[:] = mesh_geometry
    
    # def compute_derivatives(self, inputs, outputs, derivatives):
    #     print('compute beam matrix derivatives...')
    #     #update boundary nodes:
    #     # if self.boundary_nodes is not None: 
    #     #     self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
    #     # #update interior nodes
    #     # if self.interior_nodes is not None: 
    #     #     self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
    #     # else: 
    #     #     self.xs.msh.geometry.x[:,0:2]=inputs['xy']
        
    #     # for i in range(6):
    #     #     self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
    #     #     self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        
    #     # self.xs._compute_xs_stiffness_matrix()
    #     if self.check_partials != 'w':                
    #         pKpx = self.xs.compute_pKpx()
    #         derivatives['K', 'xy'] = pKpx[:,self.xs.dofs_boundary]
    #         derivatives['K', 'xy_interior'] = pKpx[:,self.xs.dofs_interior]
        
    #         pApx = self.xs.compute_pApx()
    #         derivatives['A','xy'] = pApx[self.xs.dofs_boundary]
    #         derivatives['A','xy_interior'] = pApx[self.xs.dofs_interior]
            
    #     if self.check_partials != 'x':
    #         pKpw = self.xs.compute_pKpw()
    #         pKpl = self.xs.compute_pKpl()
    #         derivatives['K', 'w'] = pKpw #return (36 x num_warping_function_dofs*6) but need to be ordered  
    #         derivatives['K', 'lmbda'] = pKpl #return (36 x 30*6)

    #     # derivatives['A','xy'] = np.vstack([pApx[self.xs.dofs_x_boundary],
    #     #                                 pApx[self.xs.dofs_y_boundary]]).T
    #     # derivatives['A','xy_interior'] = np.vstack([pApx[self.xs.dofs_x_interior],
    #     #                                 pApx[self.xs.dofs_y_interior]]).T
    
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        # dxA = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dxB = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dwA = np.zeros_like(self.d_outputs['w_A'])
        # dwB = np.zeros_like(self.d_outputs['w_B'])
        # dlmbda = np.zeros_like(self.d_outputs['lmbda'])
        print('compute vjp:')
        mesh_geometry = self.xs.msh.geometry.x.copy()
        
        if self.check_partials != 'w':
            num_spatial_dofs = self.xs.V_x.dofmap.index_map_bs*self.xs.V_x.dofmap.index_map.size_global
            dx = np.zeros((num_spatial_dofs,))
            
            if len(np.nonzero(d_outputs['K'])[0])>0:
                dx += self.xs._compute_pK_action(d_outputs['K'],
                                                derivative_type='x')
            if not np.isclose(d_outputs['A'][0],0):
                dx += self.xs._compute_pA_action(d_outputs['A'])

            d_inputs['xy'] = np.vstack([dx[self.xs.dofs_x_boundary],
                                            dx[self.xs.dofs_y_boundary]]).T
            d_inputs['xy_interior'] = np.vstack([dx[self.xs.dofs_x_interior],
                                            dx[self.xs.dofs_y_interior]]).T
            
        if self.check_partials != 'x':
            if len(np.nonzero(d_outputs['K'])[0])>0:
                d_inputs['w'] = self.xs._compute_pK_action(d_outputs['K'],
                                                            derivative_type='w')
                d_inputs['lmbda'] = self.xs._compute_pK_action(d_outputs['K'],
                                                            derivative_type='l')
            
        #return mesh geometry to original state:
        self.xs.msh.geometry.x[:] = mesh_geometry

class EllipticSmoothing(csdl.CustomExplicitOperation):

    def __init__(
        self,
        domain,
        boundary_nodes,
        interior_nodes,
        filename='xs',
        directory='output/',
        write_mesh_history=True,
        verbose=False,
    ):
        super().__init__()
        self.domain = domain
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes
        self.write_mesh_history = write_mesh_history
        self.verbose = verbose
        # self.step = 0.0
        # self.filename =filename
        # self.original_boundary = self.domain.geometry.x[self.boundary_nodes,0:2]

        #initialize mesh motion solver
        self.mesh_motion = ALBATROSS.mesh.MeshMotion(self.domain,
                                  self.boundary_nodes,
                                  self.interior_nodes,
                                  directory=directory,
                                  write_initial_mesh=write_mesh_history)
        

    def evaluate(self, inputs: csdl.VariableGroup):
        #boundary node position inputs:
        self.declare_input('xy',inputs.xy)
        # self.declare_input('xy_interior',inputs.xy)

        # construct output of the model
        output = csdl.VariableGroup()

        #create output for new interior node positions
        output.xy_interior = self.create_output('xy_interior',(self.interior_nodes.shape[0],2))
        # output.xy_interior.name = 'xy_interior, output'

        return output

    def compute(self, inputs, outputs):
        if self.verbose:
            print(f'perform elliptic smoothing (step {self.mesh_motion.step})')
        # displacement = inputs['xy']-inputs['xy_prev']
        # print('xy values (mesh motion op):')
        # print(inputs['xy'].flatten())
        # print('xy mesh coords')
        # print(self.domain.geometry.x[self.boundary_nodes,:2])
        # displacement = inputs['xy']-self.original_boundary
        # displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        xy_interior = self.mesh_motion.smooth_mesh(inputs['xy'])

        if self.write_mesh_history:
            self.mesh_motion.write_mesh_deformation()
        # self.mesh_motion.plot_current_mesh_state()
        # xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
        #                                             self.boundary_nodes,
        #                                             displacement,
        #                                             self.interior_nodes,
        #                                             mode='lin_elas',
        #                                             plot_result=True,
        #                                             step=self.step,
        #                                             filename=self.filename)
        # self.step += 1.0

        outputs['xy_interior'] = xy_interior

    def compute_derivatives(self, inputs, outputs, derivatives):
        #returns displacement w.r.t to boundary nodes

        # # displacement = inputs['xy']-self.original_boundary
        # displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]

        # xy_interior,duhdx = ALBATROSS.mesh.smooth_mesh(self.domain,
        #                                                 self.boundary_nodes,
        #                                                 displacement,
        #                                                 self.interior_nodes,
        #                                                 plot_result=False,
        #                                                 get_deriv=True,
        #                                                 mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        # print('duhdx shape:')
        # print(duhdx.shape)

        duhdx = self.mesh_motion.get_derivatives()

        derivatives['xy_interior','xy'] = duhdx
        # derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
        #                                                  inputs['xy'].flatten().shape[0]))

class NonmatchingInterpolationMatrix(csdl.CustomExplicitOperation):
    '''
    return the interpolation matrix that maps the dofs from mesh = mesh_id to 
    the mortar mesh corresponding to collision = collsion


    '''
    def __init__(self,xs,
                    mesh_id=0,
                    collision=(0,1),
                    foreground_boundary = None,
                    foreground_interior = None,
                    mortar_boundary = None,
                    mortar_interior = None):
        super().__init__()
        self.xs = xs
        self.mesh_id = mesh_id
        self.collision = collision

        self.foreground_boundary = foreground_boundary
        self.foreground_interior = foreground_interior
        self.mortar_boundary = mortar_boundary
        self.mortar_interior = mortar_interior

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy_foreground',inputs.xy_foreground)
        self.declare_input('xy_interior_foreground',inputs.xy_interior_foreground)
        self.declare_input('xy_mortar',inputs.xy_mortar)
        self.declare_input('xy_interior_mortar',inputs.xy_interior_mortar)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.P = self.create_output('P', (self.xs.collisions[self.collision].mortar_mesh.size,
                                             self.xs.system_sizes[self.mesh_id][self.mesh_id][0]))
        outputs.P.name = 'interpolation_matrix'+str(self.mesh_id)

        return outputs
    
    def compute(self, inputs, outputs):
        print('compute beam matrix from warping function state')
        foreground_geometry = self.xs.XSs[self.mesh_id].msh.geometry.x.copy()
        mortar_geometry = self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.copy()

        #FOREGROUND MESH
        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_boundary,0:2]=inputs['xy_foreground']
        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_interior,0:2]=inputs['xy_interior_foreground']
        
        #MORTAR MESH
        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_boundary,0:2]=inputs['xy_mortar']
        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_interior,0:2]=inputs['xy_interior_mortar']

        #get the interpolation matrix from the foreground mesh to the mortar mesh:
        P_petsc = self.xs._construct_interpolation_operator(self.collision,self.mesh_id)

        # if self.collision.index(self.mesh_id) == 0:
        #     P_petsc = self.xs.collisions[self.collision].PA
        # elif self.collision.index(self.mesh_id) == 1:
        #     P_petsc = self.xs.collisions[self.collision].PB

        outputs['P'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(P_petsc)

        #return mesh geometry to original state:
        self.xs.XSs[self.mesh_id].msh.geometry.x[:] = foreground_geometry
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[:] = mortar_geometry

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        print('computing interpolation matrix derivatives...')
        foreground_geometry = self.xs.XSs[self.mesh_id].msh.geometry.x.copy()
        mortar_geometry = self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.copy()

        #FOREGROUND MESH
        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_boundary,0:2]=inputs['xy_foreground']
        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_interior,0:2]=inputs['xy_interior_foreground']
        
        #MORTAR MESH
        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_boundary,0:2]=inputs['xy_mortar']
        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_interior,0:2]=inputs['xy_interior_mortar']

        #need to loop over subspaces and accumulate the effect of the d_outputs entries :)
        #get subspace portion of d_outputs:
        #subspace and subsubspace numbers should match between A and C
        
        dxA = np.zeros((self.xs.XSs[self.mesh_id].msh.geometry.x.shape[0],2))
        dxC = np.zeros((self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.shape[0],2))
        for i in range(4):
            for j in range(3):
                sub_space_A,sub_space_A_dofmap = self.xs.XSs[self.mesh_id].V_w.sub(i).sub(j).collapse()
                sub_space_C,sub_space_C_dofmap = self.xs.collisions[self.collision].fxn_space.sub(i).sub(j).collapse()
                dP = d_outputs['P'][np.ix_(sub_space_C_dofmap,sub_space_A_dofmap)]
                dxAij,dxCij = ALBATROSS.nonmatching_utils.action_of_geom_on_nm_interpolation_matrix(sub_space_C,
                                                                                sub_space_A,
                                                                                dP=dP)
                dxA += dxAij
                dxC += dxCij
                
        d_inputs['xy_foreground'] = dxA[self.xs.XSs[self.mesh_id].boundary_nodes,:]
        d_inputs['xy_interior_foreground'] = dxA[self.xs.XSs[self.mesh_id].interior_nodes,:]
        
        d_inputs['xy_mortar'] = dxC[self.xs.collisions[self.collision].mortar_mesh.boundary_nodes]
        d_inputs['xy_interior_mortar'] = dxC[self.xs.collisions[self.collision].mortar_mesh.interior_nodes]
        
        #return mesh geometry to original state:
        self.xs.XSs[self.mesh_id].msh.geometry.x[:] = foreground_geometry
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[:] = mortar_geometry

        # dPdx_A  = ALBATROSS.nonmatching_utils.derivative_of_interpolation_matrix_nonmatching_meshes(target_space,
        #                                                                                             source_space,
        #                                                                                             wrt='FROM')
    
        # dPdx_C  = ALBATROSS.nonmatching_utils.derivative_of_interpolation_matrix_nonmatching_meshes(target_space,
        #                                                                                             source_space,
        #                                                                                             wrt='TO')


class CrossSectionSystemComponents(csdl.CustomExplicitOperation):
    '''
    '''
    def __init__(self,xs,mesh_id=None,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs
        self.mesh_id = mesh_id
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

        #construct forms of uncoupled problem
        self.xs.XSs[self.mesh_id]._construct_xs_form()
        self.xs.XSs[self.mesh_id]._construct_KKT_forms()
        self.xs.XSs[self.mesh_id]._compile_component_vjp_forms()

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K',self.xs.system_sizes[self.mesh_id][self.mesh_id])
        outputs.K.name = 'foreground_stiffness_matrix_'+str(self.mesh_id)
        outputs.C = self.create_output('C',self.xs.system_sizes[-1][self.mesh_id])
        outputs.C.name = 'foreground_constraint_matrix_'+str(self.mesh_id)

        return outputs

    def compute(self,inputs,outputs):
        geometry = self.xs.XSs[self.mesh_id].msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        K_petsc = self.xs.XSs[self.mesh_id]._assemble_block([0,0])
        C_petsc = self.xs.XSs[self.mesh_id]._assemble_block([1,0])
        
        outputs['K'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(K_petsc)
        outputs['C'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(C_petsc)

        #return mesh geometry to original state:
        self.xs.XSs[self.mesh_id].msh.geometry.x[:] = geometry

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        
        '''
        Derivatives of K and C w.r.t. the spatial coordinates
        in reverse mode, this is derivative of outputs w.r.t. d_inputs
        '''
        print("getting system component derivatives!")

        geometry = self.xs.XSs[self.mesh_id].msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        
        #return the two vectors for the 
        # pKpxT_dK = self.xs.XSs[self.mesh_id]._compute_vjp_component_spatial(self.xs.XSs[self.mesh_id].a_form[0][0],
        #                                                                     d_outputs['K'],
        #                                                                     self.xs.XSs[self.mesh_id].V)
        # #TODO: this needs to have the proper functions (not square, so test != trial function)
        # pCpxT_dC = self.xs.XSs[self.mesh_id]._compute_vjp_component_spatial(self.xs.XSs[self.mesh_id].a_form[1][0],
        #                                                                     d_outputs['C'],
        #                                                                     test_space = self.xs.XSs[self.mesh_id].LM,
        #                                                                     trial_space = self.xs.XSs[self.mesh_id].V)
        pKpxT_dK = self.xs.XSs[self.mesh_id]._compute_vjp_dA00dx(d_outputs['K'])
        pCpxT_dC = self.xs.XSs[self.mesh_id]._compute_vjp_dA10dx(d_outputs['C'])

        d_inputs_full = pKpxT_dK + pCpxT_dC

        d_inputs['xy'] = np.vstack([d_inputs_full[self.xs.XSs[self.mesh_id].dofs_x_boundary],
                                        d_inputs_full[self.xs.XSs[self.mesh_id].dofs_y_boundary]]).T
        d_inputs['xy_interior'] = np.vstack([d_inputs_full[self.xs.XSs[self.mesh_id].dofs_x_interior],
                                                 d_inputs_full[self.xs.XSs[self.mesh_id].dofs_y_interior]]).T

        #return mesh geometry to original state:
        self.xs.XSs[self.mesh_id].msh.geometry.x[:] = geometry        

        print("DONE getting system component derivatives!")


class CrossSectionCouplingComponents(csdl.CustomExplicitOperation):
    '''
    '''
    def __init__(self,xs,collision=None,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs
        self.collision = collision
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes
        self.xs._compile_coupling_vjp_forms(self.collision)
        self.xs._construct_mortar_forms()

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        outputs = csdl.VariableGroup()
        size_c = self.xs.collisions[self.collision].mortar_mesh.size
        outputs.MC = self.create_output('MC',(size_c,size_c))
        outputs.MC.name = 'mass_coupling_matrix'+str(self.collision[0])+str(self.collision[1])
        outputs.SC = self.create_output('SC',(size_c,size_c))
        outputs.SC.name = 'boundary_coupling_matrix'+str(self.collision[0])+str(self.collision[1])
        outputs.KC = self.create_output('KC',(size_c,size_c))
        outputs.KC.name = 'overlap_correction_matrix'+str(self.collision[0])+str(self.collision[1])

        return outputs

    def compute(self,inputs,outputs):

        geometry = self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.copy()

        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        self.xs._assemble_mortar_matrices()

        MC_petsc = self.xs.collisions[self.collision].MC
        SC_petsc = self.xs.collisions[self.collision].S_C
        KC_petsc = self.xs.collisions[self.collision].mortar_xs.K_bar

        outputs['MC'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(MC_petsc)
        outputs['SC'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(SC_petsc)
        if self.xs.enable_overlap_correction and KC_petsc is not None:
            outputs['KC'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(KC_petsc)
        else:
            outputs['KC'] = np.zeros_like(outputs['MC'])

        #return mesh geometry to original state:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[:] = geometry


    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        
        '''
        Derivatives of M_C and S_C w.r.t. the spatial coordinates
        '''
        print('getting coupling component derivatives:')
        geometry = self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.copy()

        #return the two vectors for the 
        # pMCpxT_dMC = self.xs._compute_vjp_component_spatial(self.xs.collisions[self.collision].MC_form,
        #                                                                     self.collision,
        #                                                                     d_outputs['MC'],
        #                                                                     test_space = self.xs.collisions[self.collision].fxn_space)
        
        # pSCpxT_dSC = self.xs._compute_vjp_component_spatial(self.xs.collisions[self.collision].SC_form,
        #                                                                     self.collision,
        #                                                                     d_outputs['SC'],
        #                                                                     test_space = self.xs.collisions[self.collision].fxn_space)
        
        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        pMCpxT_dMC = self.xs._compute_vjp_dMC(d_outputs['MC'],self.collision)
                
        pSCpxT_dSC = self.xs._compute_vjp_dSC(d_outputs['SC'],self.collision)
        
        pKCpxT_dSC = self.xs._compute_vjp_dKC(d_outputs['KC'],self.collision)

        d_inputs_full = pMCpxT_dMC + pSCpxT_dSC + pKCpxT_dSC

        d_inputs['xy'] = np.vstack([d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_x_boundary],
                                        d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_y_boundary]]).T
        d_inputs['xy_interior'] = np.vstack([d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_x_interior],
                                                 d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_y_interior]]).T
        
        #return mesh geometry to original state:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[:] = geometry


class CoupledBeamMatrixFromWarping(csdl.CustomExplicitOperation):
    """
    Explicit map:
        K = K(x_A, x_B, x_C, w_A, w_B)

    Important:
        This op mutates self.xs in-place, so all mutable state is saved/restored.
        Geometry-dependent coupled operators must be rebuilt at the current geometry
        before evaluating K.

    Notes:
        - lmbda is intentionally omitted as an explicit input unless later FD checks
          show K depends explicitly on it.
        - reverse-mode wrt the full coupled section assembly requires a coupled-level
          VJP implementation on CoupledCrossSection.
    """
    def __init__(self, xs, collision=(0, 1), check_partials=None, verbose=False):
        super().__init__()
        self.xs = xs
        self.collision = collision
        self.check_partials = check_partials
        self.verbose = verbose

        self.xs_A = self.xs.XSs[self.collision[0]]
        self.xs_B = self.xs.XSs[self.collision[1]]
        self.mortar = self.xs.collisions[self.collision].mortar_mesh
        self.mortar_xs = self.xs.collisions[self.collision].mortar_xs

        # Foreground dK forms are still useful pieces, but they are NOT sufficient
        # for the full coupled reverse pass by themselves.
        for xs_i in self.xs.XSs:
            xs_i._set_up_dK_forms()

    def evaluate(self, inputs: csdl.VariableGroup):

        # Geometry inputs
        if self.check_partials not in ('w',):
            self.declare_input('xy_A', inputs.xy_A)
            self.declare_input('xy_A_interior', inputs.xy_A_interior)

        if self.check_partials not in ('w',):
            self.declare_input('xy_B', inputs.xy_B)
            self.declare_input('xy_B_interior', inputs.xy_B_interior)

        if self.check_partials not in ('w',):
            self.declare_input('xy_C', inputs.xy_C)
            self.declare_input('xy_C_interior', inputs.xy_C_interior)

        # Warping inputs
        if self.check_partials not in ('xA','xB','xC'):
            self.declare_input('w_A', inputs.w_A)
            self.declare_input('w_B', inputs.w_B)

        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K', (6, 6))
        outputs.K.name = 'beam stiffness matrix'
        return outputs

    # ------------------------------------------------------------------
    # state helpers
    # ------------------------------------------------------------------
    def _save_state(self):
        return {
            'geom_A': self.xs_A.msh.geometry.x.copy(),
            'geom_B': self.xs_B.msh.geometry.x.copy(),
            'geom_C': self.mortar.msh.geometry.x.copy(),
            'w_A': [wf.x.array.copy() for wf in self.xs_A.warping_functions],
            'w_B': [wf.x.array.copy() for wf in self.xs_B.warping_functions],
            'lm_A': [lm.x.array.copy() for lm in self.xs_A.lmbdas],
            'lm_B': [lm.x.array.copy() for lm in self.xs_B.lmbdas],
            'w_C': [wf.x.array.copy() for wf in self.mortar_xs.warping_functions],
        }

    def _restore_state(self, state):
        self.xs_A.msh.geometry.x[:] = state['geom_A']
        self.xs_B.msh.geometry.x[:] = state['geom_B']
        self.mortar.msh.geometry.x[:] = state['geom_C']

        for i in range(6):
            self.xs_A.warping_functions[i].x.array[:] = state['w_A'][i]
            self.xs_B.warping_functions[i].x.array[:] = state['w_B'][i]
            self.xs_A.lmbdas[i].x.array[:] = state['lm_A'][i]
            self.xs_B.lmbdas[i].x.array[:] = state['lm_B'][i]
            self.mortar_xs.warping_functions[i].x.array[:] = state['w_C'][i]

    def _update_geometry_from_inputs(self, inputs):
        if self.check_partials in (None, 'xA'):
            self.xs_A.msh.geometry.x[self.xs_A.boundary_nodes,0:2] = inputs['xy_A']
            self.xs_A.msh.geometry.x[self.xs_A.interior_nodes,0:2] = inputs['xy_A_interior']

        if self.check_partials in (None, 'xB'):
            self.xs_B.msh.geometry.x[self.xs_B.boundary_nodes,0:2] = inputs['xy_B']
            self.xs_B.msh.geometry.x[self.xs_B.interior_nodes,0:2] = inputs['xy_B_interior']

        if self.check_partials in (None, 'xC'):
            self.mortar.msh.geometry.x[self.mortar.boundary_nodes,0:2] = inputs['xy_C']
            self.mortar.msh.geometry.x[self.mortar.interior_nodes,0:2] = inputs['xy_C_interior']
    
    def _update_state_from_inputs(self, inputs):
        for i in range(6):
            self.xs_A.warping_functions[i].x.array[:] = inputs['w_A'][:, i]
            self.xs_B.warping_functions[i].x.array[:] = inputs['w_B'][:, i]

    def _refresh_coupled_geometry_dependent_data(self):
        """
        Rebuild geometry-dependent objects used by coupled section stiffness evaluation.

        This is necessary because PA/PB and mortar matrices/forms depend on geometry.
        """
        # Interpolation operators depend on current foreground/mortar geometry
        self.xs._construct_interpolation_operators()

        # Mortar forms/matrices depend on current mortar geometry
        self.xs._construct_mortar_forms()
        self.xs._assemble_mortar_matrices()

        # Not strictly needed for _compute_xs_stiffness_matrix itself unless other
        # code depends on Sij being current, but harmless to refresh consistently.
        self.xs._construct_coupling_terms()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def compute(self, inputs, outputs):
        if self.verbose:
            print('compute beam matrix from warping function state')
        state = self._save_state()
        try:
            if self.check_partials != 'w':
                self._update_geometry_from_inputs(inputs)

            if self.check_partials not in ('xA','xB','xC'):
                self._update_state_from_inputs(inputs)

            # Geometry-dependent operators must be current before computing K
            self._refresh_coupled_geometry_dependent_data()

            # Compute full coupled section stiffness from current geometry + warping
            self.xs._compute_xs_stiffness_matrix()
            outputs['K'] = self.xs.K.copy()

        finally:
            self._restore_state(state)

    # ------------------------------------------------------------------
    # reverse-mode VJP
    # ------------------------------------------------------------------
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        if mode != 'rev':
            return

        if self.verbose:
            print("getting coupled beam matrix derivatives...")
        state = self._save_state()
        try:
            # IMPORTANT: update all current geometry and all current warping state
            # BEFORE any derivative action is evaluated.
            if self.check_partials in (None,'xA','xB','xC'):
                self._update_geometry_from_inputs(inputs)

            if self.check_partials not in ('xA','xB','xC'):
                self._update_state_from_inputs(inputs)

            # Must match the same current geometry-dependent coupled data as forward
            self._refresh_coupled_geometry_dependent_data()
            self.xs._compute_xs_stiffness_matrix()

            dK_seed = d_outputs['K']

            # ------------------------------------------------------------------
            # Correct implementation target:
            #   CoupledCrossSection must provide the full VJP for
            #   K(x_A, x_B, x_C, w_A, w_B)
            #
            # Suggested signature:
            #   dXA, dXB, dXC, dWA, dWB = self.xs.compute_section_vjp(dK_seed)
            #
            # The old self.xs._compute_pK_action(...) is NOT sufficient because it
            # misses mortar/interpolation correction paths.
            # ------------------------------------------------------------------
            if not hasattr(self.xs, "compute_section_vjp"):
                raise NotImplementedError(
                    "CoupledBeamMatrixFromWarping reverse-mode is incomplete without "
                    "CoupledCrossSection.compute_section_vjp(dK_seed). "
                    "The old _compute_pK_action foreground-only path is not sufficient "
                    "for the coupled section matrix."
                )

            results = self.xs.compute_section_vjp(dK_seed)

            # Expect:
            # results = {
            #   'x_A': full_Vx_vector_on_A,
            #   'x_B': full_Vx_vector_on_B,
            #   'x_C': full_Vx_vector_on_C,
            #   'w_A': ndarray shape (nA, 6),
            #   'w_B': ndarray shape (nB, 6),
            # }

            if self.check_partials in (None,'xA','xB','xC'):
                dxA = results['x_A']
                dxB = results['x_B']
                dxC = results['x_C']

                d_inputs['xy_A'] = np.vstack([
                    dxA[self.xs_A.dofs_x_boundary],
                    dxA[self.xs_A.dofs_y_boundary],
                ]).T
                d_inputs['xy_A_interior'] = np.vstack([
                    dxA[self.xs_A.dofs_x_interior],
                    dxA[self.xs_A.dofs_y_interior],
                ]).T

                d_inputs['xy_B'] = np.vstack([
                    dxB[self.xs_B.dofs_x_boundary],
                    dxB[self.xs_B.dofs_y_boundary],
                ]).T
                d_inputs['xy_B_interior'] = np.vstack([
                    dxB[self.xs_B.dofs_x_interior],
                    dxB[self.xs_B.dofs_y_interior],
                ]).T

                d_inputs['xy_C'] = np.vstack([
                    dxC[self.mortar.dofs_x_boundary],
                    dxC[self.mortar.dofs_y_boundary],
                ]).T
                d_inputs['xy_C_interior'] = np.vstack([
                    dxC[self.mortar.dofs_x_interior],
                    dxC[self.mortar.dofs_y_interior],
                ]).T

            if self.check_partials not in ('xA','xB','xC'):
                d_inputs['w_A'] = results['w_A']
                d_inputs['w_B'] = results['w_B']

        finally:
            self._restore_state(state)


class NonmatchingWarpingFunctionState(csdl.experimental.CustomImplicitOperation):
    """
    Implicit state solve for one coupled nonmatching cross-section
    made from:
        - foreground mesh A
        - foreground mesh B
        - one mortar mesh for collision (0,1)

    Inputs:
        xy_A, xy_A_interior
        xy_B, xy_B_interior
        xy_C, xy_C_interior

    Outputs:
        w_A, w_B, lmbda
    """

    def __init__(self, xs, collision=(0, 1), verbose=False):
        super().__init__()
        self.xs = xs
        self.collision = collision
        self.verbose = verbose

        self.xs_A = self.xs.XSs[self.collision[0]]
        self.xs_B = self.xs.XSs[self.collision[1]]
        self.mortar = self.xs.collisions[self.collision].mortar_mesh

    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input('xy_A', inputs.xy_A)
        self.declare_input('xy_A_interior', inputs.xy_A_interior)

        self.declare_input('xy_B', inputs.xy_B)
        self.declare_input('xy_B_interior', inputs.xy_B_interior)

        self.declare_input('xy_C', inputs.xy_C)
        self.declare_input('xy_C_interior', inputs.xy_C_interior)

        outputs = csdl.VariableGroup()

        outputs.w_A = self.create_output(
            'w_A',
            (self.xs.system_size_list[0], 6)
        )
        outputs.w_B = self.create_output(
            'w_B',
            (self.xs.system_size_list[1], 6)
        )
        outputs.lmbda = self.create_output(
            'lmbda',
            (self.xs.system_size_list[-1], 6)
        )

        return outputs

    # -----------------------------
    # geometry/state helpers
    # -----------------------------
    def _save_geometry(self):
        return (
            self.xs_A.msh.geometry.x.copy(),
            self.xs_B.msh.geometry.x.copy(),
            self.mortar.msh.geometry.x.copy(),
        )

    def _restore_geometry(self, geom_A, geom_B, geom_C):
        self.xs_A.msh.geometry.x[:] = geom_A
        self.xs_B.msh.geometry.x[:] = geom_B
        self.mortar.msh.geometry.x[:] = geom_C

    def _update_geometry_from_inputs(self, inputs):
        # mesh A
        self.xs_A.msh.geometry.x[self.xs_A.boundary_nodes, 0:2] = inputs['xy_A']
        self.xs_A.msh.geometry.x[self.xs_A.interior_nodes, 0:2] = inputs['xy_A_interior']

        # mesh B
        self.xs_B.msh.geometry.x[self.xs_B.boundary_nodes, 0:2] = inputs['xy_B']
        self.xs_B.msh.geometry.x[self.xs_B.interior_nodes, 0:2] = inputs['xy_B_interior']

        # mortar mesh
        self.mortar.msh.geometry.x[self.mortar.boundary_nodes, 0:2] = inputs['xy_C']
        self.mortar.msh.geometry.x[self.mortar.interior_nodes, 0:2] = inputs['xy_C_interior']

    def _update_state_from_outputs(self, outputs):
        for i in range(6):
            self.xs_A.warping_functions[i].x.array[:] = outputs['w_A'][:, i]
            self.xs_B.warping_functions[i].x.array[:] = outputs['w_B'][:, i]

            # shared LM block is stored on both foreground XSs
            self.xs_A.lmbdas[i].x.array[:] = outputs['lmbda'][:, i]
            self.xs_B.lmbdas[i].x.array[:] = outputs['lmbda'][:, i]

    def _extract_outputs_from_xs(self, outputs):
        outputs['w_A'] = np.vstack(
            [self.xs_A.warping_functions[i].x.array for i in range(6)]
        ).T
        outputs['w_B'] = np.vstack(
            [self.xs_B.warping_functions[i].x.array for i in range(6)]
        ).T
        outputs['lmbda'] = np.vstack(
            [self.xs_A.lmbdas[i].x.array for i in range(6)]
        ).T

    # -----------------------------
    # implicit solve
    # -----------------------------
    def solve_residual_equations(self, inputs, outputs):
        if self.verbose:
            print("solve nonmatching warping residual equations")

        geom_A, geom_B, geom_C = self._save_geometry()
        try:
            self._update_geometry_from_inputs(inputs)

            # Full coupled assembly + solve lives here
            self.xs._get_warping_functions()

            self._extract_outputs_from_xs(outputs)

        finally:
            self._restore_geometry(geom_A, geom_B, geom_C)

    # -----------------------------
    # inverse Jacobian action
    # -----------------------------
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        if mode != 'rev':
            return

        geom_A, geom_B, geom_C = self._save_geometry()
        try:
            self._update_geometry_from_inputs(inputs)
            self._update_state_from_outputs(outputs)

            dres_w_A, dres_w_B, dres_lmbda = self.xs.apply_inverse_jacobian(
                d_outputs['w_A'],
                d_outputs['w_B'],
                d_outputs['lmbda'],
            )

            d_residuals['w_A'] = dres_w_A
            d_residuals['w_B'] = dres_w_B
            d_residuals['lmbda'] = dres_lmbda

        finally:
            self._restore_geometry(geom_A, geom_B, geom_C)

    # -----------------------------
    # VJP wrt geometry inputs
    # -----------------------------
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        if mode != 'rev':
            return

        geom_A, geom_B, geom_C = self._save_geometry()
        try:
            self._update_geometry_from_inputs(inputs)
            self._update_state_from_outputs(outputs)

            dRdx_A, dRdx_B, dRdx_C = self.xs.compute_VJP(
                d_residuals['w_A'],
                d_residuals['w_B'],
                d_residuals['lmbda'],
            )

            # map full coordinate derivative vectors back to boundary / interior node arrays
            d_inputs['xy_A'] = np.vstack([
                dRdx_A[self.xs_A.dofs_x_boundary],
                dRdx_A[self.xs_A.dofs_y_boundary],
            ]).T
            d_inputs['xy_A_interior'] = np.vstack([
                dRdx_A[self.xs_A.dofs_x_interior],
                dRdx_A[self.xs_A.dofs_y_interior],
            ]).T

            d_inputs['xy_B'] = np.vstack([
                dRdx_B[self.xs_B.dofs_x_boundary],
                dRdx_B[self.xs_B.dofs_y_boundary],
            ]).T
            d_inputs['xy_B_interior'] = np.vstack([
                dRdx_B[self.xs_B.dofs_x_interior],
                dRdx_B[self.xs_B.dofs_y_interior],
            ]).T

            d_inputs['xy_C'] = np.vstack([
                dRdx_C[self.mortar.dofs_x_boundary],
                dRdx_C[self.mortar.dofs_y_boundary],
            ]).T
            d_inputs['xy_C_interior'] = np.vstack([
                dRdx_C[self.mortar.dofs_x_interior],
                dRdx_C[self.mortar.dofs_y_interior],
            ]).T

        finally:
            self._restore_geometry(geom_A, geom_B, geom_C)


#TODO: this is a massive operation that requires a lot of validation of individual components, 
# but would perfom much better than the current coupled approach where the system matrices are provided along with derivatives 
# class WarpingFunctionStateCoupled(csdl.experimental.CustomImplicitOperation):
#     '''
#     inputs: nodal positions of cross-sectional meshes (both overlapping and mortarmesh)
#     '''
#     def __init__(self,xs):
#         super().__init__()
#         #TODO: 
#         #this is a coupled cross-section problem:
#         self.xs = xs

#     def evaluate(self,inputs: csdl.VariableGroup):
#         # assign method inputs to input dictionary
#         #TODO: this is rigidly fixed to two overlapping meshes & one intersection; make more general
#         self.declare_input('xy_A',inputs.xy_A)
#         self.declare_input('xy_A_interior',inputs.xy_A_interior)
#         self.declare_input('xy_B',inputs.xy_B)
#         self.declare_input('xy_B_interior',inputs.xy_B_interior)
#         self.declare_input('xy_C',inputs.xy_C)
#         self.declare_input('xy_C_interior',inputs.xy_C_interior)

#         # construct output of the model
#         outputs = csdl.VariableGroup()
#         outputs.w_A = self.create_output('w_A', (self.xs.XSs[0].V.dofmap.index_map.size_global,6))
#         outputs.w_B = self.create_output('w_B', (self.xs.XSs[1].V.dofmap.index_map.size_global,6))
#         outputs.lmbda = self.create_output('lmbda', (self.xs.XSs[0].LM.value_size,6))

#         return outputs
    
#     def solve_residual_equations(self, inputs, outputs):
#         print("solve residual equations:")
#         #update boundary nodes:
#         self.xs.XSs[0].msh.geometry.x[self.xs.XSs[0].boundary_nodes,0:2]=inputs['xy_A']
#         self.xs.XSs[1].msh.geometry.x[self.xs.XSs[1].boundary_nodes,0:2]=inputs['xy_B']
#         self.xs.collisions[(0,1)].mortar_mesh.msh.geometry.x[self.xs.collisions[(0,1)].mortar_mesh.boundary_nodes,0:2]=inputs['xy_C']
        
#         #update interior nodes
#         self.xs.XSs[0].msh.geometry.x[self.xs.XSs[0].interior_nodes,0:2]=inputs['xy_A_interior']
#         self.xs.XSs[1].msh.geometry.x[self.xs.XSs[1].interior_nodes,0:2]=inputs['xy_B_interior']
#         self.xs.collisions[(0,1)].mortar_mesh.msh.geometry.x[self.xs.collisions[(0,1)].mortar_mesh.interior_nodes,0:2]=inputs['xy_C_interior']
       
#         #compute warping functions
#         self.xs._get_warping_functions()
        
#         outputs['w_A'] = np.vstack([self.xs.XSs[0].warping_functions[i].x.array for i in range(6)]).T
#         outputs['w_B'] = np.vstack([self.xs.XSs[1].warping_functions[i].x.array for i in range(6)]).T
#         outputs['lmbda'] = np.vstack([self.xs.XSs[0].lmbdas[i].x.array for i in range(6)]).T
    
#     def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
#         # print("apply_inverse_jacobian:")
#         xy = inputs['xy']
#         xy_interior = inputs['xy_interior']
#         w = outputs['w']
#         lmbda = outputs['lmbda']

    
#         if mode == 'rev':    
#             # d_outputs --> d_residuals
#             # compute d_residuals = (dr_du^-1)*d_outputs

#             # dr_du is simply the finite element stiffness matrix in this case
#             # these are just the A00, A01, and A10 blocks of the assembled stiffness matrix
#             # we can leverage the already existing ksp solver and compute the multMatTranspose() using petsc,
#             # then we output these two terms to numpy matrices
#             d_residuals['w'],d_residuals['lmbda'] = self.xs.apply_inverse_jacobian(d_outputs['w'],d_outputs['lmbda'])
#             # #which does this under the hood: 
#             #     d_outputs_petsc = stack(d_outputs['w'],d_outputs['lmbda'])
#             #     self.xs.solver.solveTranspose(d_outputs_petsc,d_residuals_petsc)
#             #     d_residuals_numpy = convert_petsc_to_numpy(d_residuals_petsc)
#             #====
#             # d_residuals['w'] =  d_residuals_numpy[w_slice]
#             # d_residuals['lmbda']  = d_residuals_numpy[lmbda_slice]

#             # d_residuals['w'] = drw_dw_inv @ d_outputs['w'] + drw_dl_inv @ d_outputs['lmbda']
#             # d_residuals['lmbda'] = drl_dl_inv @ d_outputs['lmbda'] # + drl_dw_inv @ d_outputs['w'] <-- this term is = 


#     def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
#         # print("compute vector-jacobian product:")
#         xy = inputs['xy']
#         xy_interior = inputs['xy_interior']
#         w = outputs['w']
#         lmbda = outputs['lmbda']

#         # for mode = rev
#         # d_residuals --> d_inputs
#         if mode == 'rev':
#             # compute d_input = (dr_dinput)*d_residuals
#             #TODO: can also just return d_inputs as a numpy matrix here and prevent the memory overhead of converting to numpy,etc
#             # dRwdx,dRldx = self.xs.compute_dRdx() #return numpy matrices 

#             # d_inputs = dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']

#             # d_inputs['xy'] = d_inputs['boundary']
#             # d_inputs['xy_interior'] = d_inputs['interior']

#             dRdx_dr = self.xs.compute_VJP(d_residuals['w'],d_residuals['lmbda'])

#             #TODO: map to boundary or interior nodes
#             d_inputs['xy'] = np.vstack([dRdx_dr[self.xs.dofs_x_boundary],
#                                         dRdx_dr[self.xs.dofs_y_boundary]]).T
#             d_inputs['xy_interior'] = np.vstack([dRdx_dr[self.xs.dofs_x_interior],
#                                                  dRdx_dr[self.xs.dofs_y_interior]]).T
            
#             # d_inputs['xy'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda'] )['boundary']
#             # d_inputs['xy_interior'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']) ['interior']


class BeamDeflection(csdl.experimental.CustomImplicitOperation):
    '''
    initialization inputs: beam object

    evaluation inputs: loads (at some subset of points?)
    
    outputs: deflection
    '''
    def __init__(self, beam, output_pts, verbose=True, write_deformation=True):
        super().__init__()
        self.beam = beam
        self.output_pts = output_pts
        self.verbose = verbose
        self.write_deformation = write_deformation
        self.output_dofs_disp = [beam._get_dofs(output_pt,'disp') for output_pt in output_pts]
        self.output_dofs_rot = [beam._get_dofs(output_pt,'rot') for output_pt in output_pts]

        
    def evaluate(self,inputs: csdl.VariableGroup):
        #stack of flattened beam matrix values (ordered by the beam cross-section)
        self.declare_input('K',inputs.K)
        # self.declare_input('xy')
        # self.declare_input('F',inputs.F)

        outputs = csdl.VariableGroup()
        outputs.d = self.create_output('d', (self.beam.beam_element.W.dofmap.index_map.size_global,))
        outputs.d.name = 'tip_deflection'

        return outputs
    
    def solve_residual_equations(self, inputs, outputs):
        #update cross-section values in the cross-sectional objects
        # self.beam.xs_list[0].K = inputs['K']
        for idx in range(self.beam.numxs):
            self.beam.xs_list[idx].K[:,:] =inputs['K'][idx,:].reshape((6,6))
        
        #apply the new cross-sectional values to the beam model
        self.beam._update_xs_field()
        #update the forms for the beam model
        self.beam.elastic_energy()
        
        self.beam.solve()
        if self.verbose:
            print('tip deflections (x,y,z): ',self.beam.w.x.array[self.output_dofs_disp])
            print('tip rotations (theta_x,theta_y,theta_z): ',self.beam.w.x.array[self.output_dofs_rot])
        outputs['d']  = self.beam.w.x.array
        
        if self.write_deformation:
            self.beam.write_deformation()
    
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        # for mode = rev:
        # d_outputs --> d_residuals

        '''
        This is just the transpose solve with  K_1d ^ T d_residuals['d] = d_outputs['d] 

        we can use similar machinery (just the KSP.solveTranspose() method)

        '''

        d_residuals['d'] = self.beam.apply_inverse_jacobian(d_outputs['d'])    

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # for mode = rev
        # d_residuals --> d_inputs
        d_inputs['K'] = self.beam.compute_vjp(d_residuals['d'])

    # def compute_derivatives(self, inputs, outputs, derivatives):
        # return super().compute_derivatives(inputs, outputs, derivatives)


class SectionPropertyMapper(csdl.CustomExplicitOperation):
    def __init__(self,beam,section_list):
        super().__init__()
        self.beam = beam
        self.section_list = section_list
        self.num_sections = len(section_list)

    def evaluate(self):
        return super().evaluate()
    
    def compute(self, inputs, outputs):
        return super().compute(inputs, outputs)
    
    def compute_jacvec_product(self, inputs, outputs, derivatives, d_inputs, d_outputs, mode):
        return super().compute_jacvec_product(inputs, outputs, derivatives, d_inputs, d_outputs, mode)


class BeamMass(csdl.CustomExplicitOperation):
    '''
    Compute the mass of a beam based on cross-sectional areas and mass properties
    '''
    def __init__(self,beam):
        super().__init__()
        self.beam = beam

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('A',inputs.A)

        outputs = csdl.VariableGroup()

        outputs.M = self.create_output('M', (1,))
        outputs.M.name = 'mass'

        return outputs


    def compute(self,inputs,outputs):
        #Do we need to do this? hmm?
        self.beam.xs_list[0].A = inputs['A']
        
        #TODO: move this to a more general method that handles multiple sections?
        # self.beam._update_linear_density()
        
        self.beam.linear_density.x.array[:] = self.beam.xs_list[0].A*self.beam.xs_list[0].materials[0].density

        self.beam.get_mass()
        outputs['M'] = self.beam.M


    def compute_derivatives(self, inputs, outputs, derivatives):
        #this is a simplified mass model for a constant cross-section beam
        # we would need to integrate along the span based on the interpolated cross-sectional area value
        # to get the accurate value for a tapered/variable cross-section
        L  = self.beam.get_length()
        rho = self.beam.xs_list[0].materials[0].density
        
        derivatives['M','A'] = np.array([rho*L])


class MeshQuality(csdl.CustomExplicitOperation):
    def __init__(self,msh,boundary_nodes,interior_nodes):
        super().__init__()

        self.msh = msh
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

        self.mesh_quality = ALBATROSS.mesh.MeshQuality(self.msh,
                                                       self.boundary_nodes,
                                                       self.interior_nodes)
        

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)

        outputs = csdl.VariableGroup()
        outputs.Q = self.create_output('Q',(1,))

        return outputs
    
    
    def compute(self,inputs,outputs):
        self.mesh_quality.get_mesh_metric(inputs['xy'],inputs['xy_interior'])
        outputs['Q'] = self.mesh_quality.Q
        print('mesh quality metric: ',self.mesh_quality.Q)


    def compute_derivatives(self, inputs, outputs, derivatives):
        self.mesh_quality.get_derivatives(inputs['xy'],inputs['xy_interior'])
        dQdx = self.mesh_quality.dQdx
        derivatives['Q', 'xy'] = dQdx[self.mesh_quality.dofs_boundary]
        derivatives['Q', 'xy_interior'] = dQdx[self.mesh_quality.dofs_interior]
