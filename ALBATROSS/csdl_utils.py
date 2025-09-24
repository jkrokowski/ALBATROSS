import csdl_alpha as csdl
import ALBATROSS
# from scipy.spatial import cKDTree
import numpy as np

# custom cross-sectional model
class CrossSection(csdl.CustomExplicitOperation):

    def __init__(self, 
                 domain,
                 xs_analysis_type,
                 material_type,
                 material_name,
                 mech_props,
                 boundary_nodes=None,
                 interior_nodes=None):
        
        super().__init__()

        self.domain = domain
        self.xs_analysis_type = xs_analysis_type
        self.material_type = material_type
        self.material_name =material_name
        self.mech_props = mech_props

        self.material = ALBATROSS.material.Material(name=self.material_name,
                            mat_type=self.material_type,
                            mech_props=self.mech_props,
                            density=2700)
        
        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes

        self.xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        # declare output variables
        if self.xs_analysis_type == 'TS':
            shape = (6,6)
        elif self.xs_analysis_type == 'EB':
            shape = (4,4)
        K = self.create_output('K', shape)
        A = self.create_output('A',(1,))

        # construct output of the model
        output = csdl.VariableGroup()
        output.K = K

        output.A = A

        return output
    
    def compute(self, inputs, outputs):     
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=inputs['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
        outputs['K'] = self.xs.K
        outputs['A'] = self.xs.A

    def compute_derivatives(self, inputs, outputs_vals, derivatives):
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=inputs['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
            self.xs.compute_xs_stiffness_matrix_sensitivities()
            #TODO: need to restrict to just derivatives on boundary
            if self.boundary_nodes is not None: 
                # derivatives['K', 'xy'] = xs.dKdx_boundary
                derivatives['K', 'xy'] = self.xs.dKdx_boundary.reshape((36,inputs['xy'].flatten().shape[0]))
            else: 
                derivatives['K', 'xy'] = self.xs.dKdx
            # derivatives['K', 'xy'] = xs.dKdx.reshape((36,xy.flatten().shape[0]))
            # derivatives['K', 'xy'] = xs.dKdx.reshape((xy.flatten().shape[0],36))
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
            self.xs.compute_xs_stiffness_matrix_sensitivities_EB()

class WarpingFunctionState(csdl.experimental.CustomImplicitOperation):
    '''
    inputs: nodal positions of cross-sectional mesh'''
    def __init__(self,xs,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs

        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes

    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.w = self.create_output('w', (self.xs.V.dofmap.index_map.size_global,6))
        outputs.lmbda = self.create_output('lmbda', (self.xs.LM.value_size,6))

        return outputs
    
    def solve_residual_equations(self, inputs, outputs):
        print("solve residual equations:")
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        else: 
            self.xs.msh.geometry.x[:,0:2]=inputs['xy']

        #compute warping functions
        self.xs._get_warping_functions()
        
        outputs['w'] = np.vstack([self.xs.warping_functions[i].x.array for i in range(6)]).T
        outputs['lmbda'] = np.vstack([self.xs.lmbdas[i].x.array for i in range(6)]).T
    
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        # print("apply_inverse_jacobian:")
        #TODO: do we need to update the inputs, etc (eg. does the mesh update need to happen here?)
        xy = inputs['xy']
        xy_interior = inputs['xy_interior']
        w = outputs['w']
        lmbda = outputs['lmbda']

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


    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # print("compute vector-jacobian product:")
        xy = inputs['xy']
        xy_interior = inputs['xy_interior']
        w = outputs['w']
        lmbda = outputs['lmbda']

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



class BeamMatrixFromWarping(csdl.CustomExplicitOperation):
    def __init__(self,xs,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs

        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes


    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        # self.declare_input('xy',inputs.xy)
        # self.declare_input('xy_interior',inputs.xy_interior)
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
        # #update boundary nodes:
        # if self.boundary_nodes is not None: 
        #     self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        # #update interior nodes
        # if self.interior_nodes is not None: 
        #     self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        # else: 
        #     self.xs.msh.geometry.x[:,0:2]=inputs['xy']

        for i in range(6):
            self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
            self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        # print('mesh coords:')
        # print(self.xs.msh.geometry.x[:,0:2])
        self.xs._compute_xs_stiffness_matrix()

        outputs['K'] = self.xs.K
        outputs['A'] = self.xs.A
    
    def compute_derivatives(self, inputs, outputs, derivatives):
        print('compute beam matrix derivatives...')
        #update boundary nodes:
        # if self.boundary_nodes is not None: 
        #     self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        # #update interior nodes
        # if self.interior_nodes is not None: 
        #     self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        # else: 
        #     self.xs.msh.geometry.x[:,0:2]=inputs['xy']
        
        # for i in range(6):
        #     self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
        #     self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        
        # self.xs._compute_xs_stiffness_matrix()
                
        # pKpx = self.xs.compute_pKpx()
        pKpw = self.xs.compute_pKpw()
        pKpl = self.xs.compute_pKpl()

        # print('input vals:')
        # print(inputs['xy'])
        # print('mesh coords:')
        # print(self.xs.msh.geometry.x[:,0:2])
        
        #declare derivatives
        # derivatives['K', 'xy'] = pKpx[:,self.xs.dofs_boundary]
        # derivatives['K', 'xy_interior'] = pKpx[:,self.xs.dofs_interior]
        derivatives['K', 'w'] = pKpw #return (36 x num_warping_function_dofs*6) but need to be ordered  
        derivatives['K', 'lmbda'] = pKpl #return (36 x 30*6)

        # derivatives['K', 'xy'] = pKpx[:,np.concatenate([self.xs.dofs_x_boundary,self.xs.dofs_y_boundary])]
        # derivatives['K', 'xy'] = np.hstack([pKpx[:,self.xs.dofs_x_boundary],
        #                                 pKpx[:,self.xs.dofs_y_boundary]]) #return (36 x num_boundary_nodes*2)
        # derivatives['K', 'xy_interior'] = np.hstack([pKpx[:,self.xs.dofs_x_interior],
        #                                 pKpx[:,self.xs.dofs_y_interior]]) #return (36 x num_interior_nodes*2)

class EllipticSmoothing(csdl.CustomExplicitOperation):

    def __init__(self,domain,boundary_nodes,interior_nodes):
        super().__init__()
        self.domain = domain
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes
        self.step = 0.0


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
        print(f'perform elliptic smoothing (step {self.step})')
        # displacement = inputs['xy']-inputs['xy_prev']
        displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        
        #
        xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                    self.boundary_nodes,
                                                    displacement,
                                                    self.interior_nodes,
                                                    mode='lin_elas',
                                                    plot_result=False,
                                                    step=self.step)
        self.step += 1.0

        outputs['xy_interior']=xy_interior

    def compute_derivatives(self, inputs, outputs, derivatives):
        #returns displacement w.r.t to boundary nodes

        displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]

        xy_interior,duhdx = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                        self.boundary_nodes,
                                                        displacement,
                                                        self.interior_nodes,
                                                        plot_result=False,
                                                        get_deriv=True,
                                                        mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        # print('duhdx shape:')
        # print(duhdx.shape)
        derivatives['xy_interior','xy'] = duhdx
        # derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
        #                                                  inputs['xy'].flatten().shape[0]))

# class OversetMeshManager(csdl.CustomExplicitOperation):
#     """
#     Manages mesh connectivity and interpolation weights for overlapping meshes
#     during optimization iterations.
#     """
    
#     def initialize(self):
#         # Design variables that affect mesh positions/shapes
#         self.add_input('design_vars', shape=(n_design_vars,))
        
#         # Current mesh geometry states
#         self.add_input('mesh_A_coords', shape=(n_nodes_A, 2))
#         self.add_input('mesh_B_coords', shape=(n_nodes_B, 2))
        
#         # Outputs
#         self.add_output('connectivity_changed', shape=(1,))  # Boolean flag
#         self.add_output('interpolation_weights', shape=(n_interp_weights,))
#         self.add_output('mortar_mesh_coords', shape=(n_mortar_nodes, 2))
#         self.add_output('collision_matrix', shape=(n_elements_A, n_elements_B))
        
#         # Cached states for comparison
#         self.previous_connectivity = None
#         self.previous_design_vars = None
#         self.tolerance = 1e-6  # Connectivity change threshold
    
#     def compute(self, inputs, outputs):
#         design_vars = inputs['design_vars']
#         mesh_A = inputs['mesh_A_coords']
#         mesh_B = inputs['mesh_B_coords']
        
#         # 1. Check if significant geometry change occurred
#         connectivity_changed = self._check_connectivity_change(design_vars, mesh_A, mesh_B)
        
#         if connectivity_changed:
#             # 2a. Rebuild collision detection and mortar mesh
#             collision_matrix = self._detect_collisions(mesh_A, mesh_B)
#             mortar_coords = self._construct_mortar_mesh(mesh_A, mesh_B, collision_matrix)
#             interp_weights = self._compute_interpolation_weights(mesh_A, mesh_B, mortar_coords)
            
#             # Cache current state
#             self._cache_current_state(design_vars, collision_matrix)
#         else:
#             # 2b. Only update interpolation weights (linear update)
#             collision_matrix = self.previous_connectivity
#             mortar_coords = self._update_mortar_positions(design_vars)
#             interp_weights = self._update_interpolation_weights(mesh_A, mesh_B, mortar_coords)
        
#         outputs['connectivity_changed'] = connectivity_changed
#         outputs['interpolation_weights'] = interp_weights
#         outputs['mortar_mesh_coords'] = mortar_coords
#         outputs['collision_matrix'] = collision_matrix
    
#     def compute_derivatives(self, inputs, derivatives):
#         # Only provide derivatives for smooth (non-connectivity-changing) updates
#         if not self.connectivity_changed:
#             # Compute derivatives of interpolation weights w.r.t. design variables
#             derivatives['interpolation_weights', 'design_vars'] = self._compute_weight_derivatives()
#             derivatives['mortar_mesh_coords', 'design_vars'] = self._compute_mortar_derivatives()

class BeamModel(csdl.CustomExplicitOperation):
    '''
    initialization inputs: beam axis and xs info

    evaluation inputs: loads (at some subset of points?)
    
    outputs: deflection
    '''
    def __init__(self,beam_axis,xs_info):
        super().__init__()
        self.beam_axis = beam_axis
        self.xs_info = xs_info
        
        ALBATROSS.beam.Beam(beam_axis,xs_info)

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('K',inputs.xy)
    
    def compute(self, inputs, outputs):
        return super().compute(inputs, outputs)
    
    def compute_derivatives(self, inputs, outputs, derivatives):
        return super().compute_derivatives(inputs, outputs, derivatives)
    

