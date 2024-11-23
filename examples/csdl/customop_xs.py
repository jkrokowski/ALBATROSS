import csdl_alpha as csdl
import ALBATROSS

# custom cross-sectional model
class CrossSection(csdl.CustomExplicitOperation):

    def __init__(self, domain,xs_analysis_type,material_type,material_name,mech_props):
        super().__init__()

        self.domain = domain
        self.xs_analysis_type = xs_analysis_type
        self.material_type = material_type
        self.material_name =material_name
        self.mech_props = mech_props

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        # self.declare_input('E', inputs.E)
        # self.declare_input('nu', inputs.nu)

        # declare output variables
        if self.xs_analysis_type == 'TS':
            shape = (6,6)
        elif self.xs_analysis_type == 'EB':
            shape = (4,4)
        K = self.create_output('K', shape)

        # declare any derivative parameters
        self.declare_derivative_parameters('K', 'xy', dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.K = K

        return output
    
    def compute(self, input_vals, output_vals):
        # E = input_vals['E']
        # nu = input_vals['nu']
        # density = input_vals['density']
               
        material = ALBATROSS.material.Material(name=self.material_name,
                                           mat_type=self.material_type,
                                           mech_props=self.mech_props,
                                           density=2700)
        
        #update mesh geometry with mesh geometry inputs 
        #TODO: implement mesh deformation subproblem
        self.domain.geometry.x[:,0:2] = inputs.xy.value

        self.xs = ALBATROSS.cross_section.CrossSection(self.domain,[material])
        self.xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = self.xs.K

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        # E = input_vals['E']
        # nu = input_vals['nu']
        xy = input_vals['xy']

        self.xs.compute_xs_stiffness_matrix_sensitivities()
        derivatives['K', 'xy'] = self.xs.dKdx


recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 10
W = .1
H = .1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
xy_coords=domain.geometry.x[:,0:2]

inputs.xy = csdl.Variable(value=xy_coords,shape=xy_coords.shape,name='xy')

# inputs.E = csdl.Variable(value=100.0, name='E')
# inputs.nu = csdl.Variable(value=0.2, name='nu')
# inputs.density =csdl.Variable(value=2700.0, name='density')

crosssection = CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2})

outputs = crosssection.evaluate(inputs)

K = outputs.K

recorder.stop()

print(K.value)

print()

# recorder.active_graph.visualize()