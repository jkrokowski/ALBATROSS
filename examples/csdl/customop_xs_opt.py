import csdl_alpha as csdl
import ALBATROSS
import numpy as np

# custom cross-sectional model
class CrossSection(csdl.CustomExplicitOperation):

    def __init__(self, domain,xs_analysis_type,material_type,material_name,mech_props):
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
        A = self.create_output('A',(1,))

        # declare any derivative parameters
        self.declare_derivative_parameters('K', 'xy', dependent=False)

        # construct output of the model
        output = csdl.VariableGroup()
        output.K = K

        output.A = A

        return output
    
    def compute(self, input_vals, output_vals):
              
        #update mesh geometry with mesh geometry inputs 
        #TODO: implement mesh deformation subproblem
        self.domain.geometry.x[:,0:2] = input_vals['xy']
        # self.domain.geometry.x[:,0:2] = inputs.xy.value

        xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])
        # self.xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = xs.K
        output_vals['A'] = xs.A

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        # xy = 
        self.domain.geometry.x[:,0:2] = input_vals['xy']
        
        xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])
        
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
            xs.compute_xs_stiffness_matrix_sensitivities()
            print(xs.dKdx)
            derivatives['K', 'xy'] = np.ones((36,32))
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
            self.xs.compute_xs_stiffness_matrix_sensitivities_EB()

        # self.xs = ALBATROSS.cross_section.CrossSection(self.domain,[material])
        
        # print('dKdx shape:')
        # print(self.xs.dKdx.shape)
        # print('xy shape:')
        # print(xy.shape)
        # print(self.xs.dKdx)
        # print('-------')
        # print(self.xs.dKdx.reshape((36,xy.flatten().shape[0])))
        
        # derivatives['K', 'xy'] = self.xs.dKdx.reshape((36,xy.flatten().shape[0]))


recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 3
W = .1
H = .1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
xy=domain.geometry.x[:,0:2]
print("shape of xy:")
print(xy.shape)
inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')

xy = inputs.xy

inputs.xy.set_as_design_variable()

crosssection = CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2})

#only call one time
outputs = crosssection.evaluate(inputs)

K = outputs.K
A = outputs.A

with csdl.namespace('Objective'):
    f = -K[4,4] - K[5,5]
    f.add_name('bending_stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = A
    g1.add_name('g1')
    g1.set_as_constraint() # constraint

recorder.stop()

print(K.value)
print(A.value)


sim = csdl.experimental.PySimulator(recorder)

print('current K:      ', sim[K])
print('current dKdx:  ', sim.compute_totals(K,xy)[K,xy], '\n')

sim.check_totals()

# sim[inputs.xy] *=2

# sim.run()

# print('current K:      ', sim[K])

# sim.compute_totals(K,inputs.xy)[K,]

from modopt import CSDLAlphaProblem
from modopt import SLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='bending_stiffness_max',simulator=sim)

optimizer = SLSQP(prob,recording=True)

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

