import csdl_alpha as csdl
import ALBATROSS
from scipy.spatial import cKDTree
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
    
    def compute(self, input_vals, output_vals):     
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=input_vals['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=input_vals['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = self.xs.K
        output_vals['A'] = self.xs.A

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=input_vals['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=input_vals['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
            self.xs.compute_xs_stiffness_matrix_sensitivities()
            #TODO: need to restrict to just derivatives on boundary
            if self.boundary_nodes is not None: 
                # derivatives['K', 'xy'] = xs.dKdx_boundary
                derivatives['K', 'xy'] = self.xs.dKdx_boundary.reshape((36,input_vals['xy'].flatten().shape[0]))
            else: 
                derivatives['K', 'xy'] = self.xs.dKdx
            # derivatives['K', 'xy'] = xs.dKdx.reshape((36,xy.flatten().shape[0]))
            # derivatives['K', 'xy'] = xs.dKdx.reshape((xy.flatten().shape[0],36))
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
            self.xs.compute_xs_stiffness_matrix_sensitivities_EB()

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

        # construct output of the model
        output = csdl.VariableGroup()

        #create output for new interior node positions
        xy_interior = self.create_output('xy_interior',(self.interior_nodes.shape[0],2))
        
        #save output
        output.xy_interior = xy_interior
        output.xy_interior.name = 'xy_interior, output'

        return output

    def compute(self, input_vals, output_vals):
        
        # displacement = input_vals['xy']-input_vals['xy_prev']
        displacement = input_vals['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        
        xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                    self.boundary_nodes,
                                                    displacement,
                                                    self.interior_nodes,
                                                    mode='lin_elas',
                                                    plot_result=False,
                                                    step=self.step)
        self.step += 1.0

        output_vals['xy_interior']=xy_interior

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        #returns displacement w.r.t to boundary nodes

        displacement = input_vals['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]

        xy_interior,duhdx = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                        self.boundary_nodes,
                                                        displacement,
                                                        self.interior_nodes,
                                                        plot_result=False,
                                                        get_deriv=True,
                                                        mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
                                                         input_vals['xy'].flatten().shape[0]))

class OversetMeshManager(csdl.CustomExplicitOperation):
    """
    Manages mesh connectivity and interpolation weights for overlapping meshes
    during optimization iterations.
    """
    
    def initialize(self):
        # Design variables that affect mesh positions/shapes
        self.add_input('design_vars', shape=(n_design_vars,))
        
        # Current mesh geometry states
        self.add_input('mesh_A_coords', shape=(n_nodes_A, 2))
        self.add_input('mesh_B_coords', shape=(n_nodes_B, 2))
        
        # Outputs
        self.add_output('connectivity_changed', shape=(1,))  # Boolean flag
        self.add_output('interpolation_weights', shape=(n_interp_weights,))
        self.add_output('mortar_mesh_coords', shape=(n_mortar_nodes, 2))
        self.add_output('collision_matrix', shape=(n_elements_A, n_elements_B))
        
        # Cached states for comparison
        self.previous_connectivity = None
        self.previous_design_vars = None
        self.tolerance = 1e-6  # Connectivity change threshold
    
    def compute(self, inputs, outputs):
        design_vars = inputs['design_vars']
        mesh_A = inputs['mesh_A_coords']
        mesh_B = inputs['mesh_B_coords']
        
        # 1. Check if significant geometry change occurred
        connectivity_changed = self._check_connectivity_change(design_vars, mesh_A, mesh_B)
        
        if connectivity_changed:
            # 2a. Rebuild collision detection and mortar mesh
            collision_matrix = self._detect_collisions(mesh_A, mesh_B)
            mortar_coords = self._construct_mortar_mesh(mesh_A, mesh_B, collision_matrix)
            interp_weights = self._compute_interpolation_weights(mesh_A, mesh_B, mortar_coords)
            
            # Cache current state
            self._cache_current_state(design_vars, collision_matrix)
        else:
            # 2b. Only update interpolation weights (linear update)
            collision_matrix = self.previous_connectivity
            mortar_coords = self._update_mortar_positions(design_vars)
            interp_weights = self._update_interpolation_weights(mesh_A, mesh_B, mortar_coords)
        
        outputs['connectivity_changed'] = connectivity_changed
        outputs['interpolation_weights'] = interp_weights
        outputs['mortar_mesh_coords'] = mortar_coords
        outputs['collision_matrix'] = collision_matrix
    
    def compute_derivatives(self, inputs, derivatives):
        # Only provide derivatives for smooth (non-connectivity-changing) updates
        if not self.connectivity_changed:
            # Compute derivatives of interpolation weights w.r.t. design variables
            derivatives['interpolation_weights', 'design_vars'] = self._compute_weight_derivatives()
            derivatives['mortar_mesh_coords', 'design_vars'] = self._compute_mortar_derivatives()

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
    


def order_boundary_nodes(coords):
    N = len(coords)
    ordered = [0]  # start with first node
    used = set(ordered)

    tree = cKDTree(coords)
    for _ in range(1, N):
        last = coords[ordered[-1]]
        dists, idxs = tree.query(last, k=N)
        next_idx = next(i for i in idxs if i not in used)
        ordered.append(next_idx)
        used.add(next_idx)

    return np.array(ordered)

