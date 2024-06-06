import csdl_alpha as csdl
import ALBATROSS

# custom paraboloid model
class CrossSection(csdl.CustomExplicitOperation):

    def __init__(self, domain,xs_analysis_type,material_type,material_name):
        super().__init__()

        self.domain = domain
        self.xs_analysis_type = xs_analysis_type
        self.material_type = material_type
        self.material_name =material_name

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        # self.declare_input('N', inputs.N)
        # self.declare_input('W', inputs.W)
        # self.declare_input('H', inputs.H)
        self.declare_input('mesh_coords',inputs.mesh_coords)
        self.declare_input('E', inputs.E)
        self.declare_input('nu', inputs.nu)

        # declare output variables
        if self.xs_analysis_type == 'TS':
            shape = (6,6)
        elif self.xs_analysis_type == 'EB':
            shape = (4,4)
        K = self.create_output('K', shape)

        # declare any derivative parameters
        # self.declare_derivative_parameters('f', 'z', dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.K = K

        # if self.return_g:
        #     g = self.create_output('g', inputs.x.shape)
        #     output.g = g

        return output
    
    def compute(self, input_vals, output_vals):
        # N = input_vals['N']
        # W = input_vals['W']
        # H = input_vals['H']
        E = input_vals['E']
        nu = input_vals['nu']
        # density = input_vals['density']
               
        material = ALBATROSS.material.Material(name=self.material_name,
                                           mat_type=self.material_type,
                                           mech_props={'E':E[0],'nu':nu[0]},
                                           density=2700)
        
        #update mesh geometry with mesh geometry inputs 
        #TODO: implement mesh deformation subproblem
        self.domain.geometry.x[:,0:2] = inputs.mesh_coords.value

        xs = ALBATROSS.cross_section.CrossSection(self.domain,[material])
        xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = xs.K

    # def compute_derivatives(self, input_vals, outputs_vals, derivatives):
    #     x = input_vals['x']
    #     y = input_vals['y']
    #     z = input_vals['z']

    #     derivatives['f', 'x'] = 2*x - self.a + y
    #     derivatives['f', 'y'] = 2*y + x + self.b

    #     if self.return_g:
    #         derivatives['g', 'x'] = z*derivatives['f', 'x']
    #         derivatives['g', 'y'] = z*derivatives['f', 'x']
    #         derivatives['g', 'z'] = outputs_vals['f']


recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 10
W = .1
H = .1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
xy_coords=domain.geometry.x[:,0:2]

inputs.mesh_coords = csdl.Variable(value=xy_coords,shape=xy_coords.shape,name='mesh_coords')
# inputs.N = csdl.Variable(value=20, name='N')
# inputs.W = csdl.Variable(value=0.1, name='W')
# inputs.H = csdl.Variable(value=0.1, name='H')
inputs.E = csdl.Variable(value=100.0, name='E')
inputs.nu = csdl.Variable(value=0.2, name='nu')
# inputs.density =csdl.Variable(value=2700.0, name='density')

crosssection = CrossSection(domain=domain,
                            xs_analysis_type='EB',
                            material_type='ISOTROPIC',
                            material_name='unobtainium')

outputs = crosssection.evaluate(inputs)

K = outputs.K

recorder.stop()
recorder.active_graph.visualize()
print(K.value)