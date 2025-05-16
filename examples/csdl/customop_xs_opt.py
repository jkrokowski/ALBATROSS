import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities

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

        # print(input_vals['xy'])
        
        xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])
        xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = xs.K
        output_vals['A'] = xs.A
        print(xs.A)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=input_vals['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=input_vals['xy']
        
        xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])
        # xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
            xs.compute_xs_stiffness_matrix_sensitivities()
            # print(xs.dKdx.shape)
            #TODO: need to restrict to just derivatives on boundary
            if self.boundary_nodes is not None: 
                # derivatives['K', 'xy'] = xs.dKdx_boundary
                derivatives['K', 'xy'] = xs.dKdx_boundary.reshape((36,xy.flatten().shape[0]))
            else: 
                derivatives['K', 'xy'] = xs.dKdx
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


    def evaluate(self, inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        # self.declare_input('xy_prev',inputs.xy)

        # construct output of the model
        output = csdl.VariableGroup()

        #create output for new interior node positions
        xy_interior = self.create_output('xy_interior',(self.interior_nodes.shape[0],2))
        
        #save output
        output.xy_interior = xy_interior

        return output

    def compute(self, input_vals, output_vals):
        #update boundary nodes:
        self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        # displacement = input_vals['xy']-input_vals['xy_prev']
        displacement = input_vals['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        
        xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                    self.boundary_nodes,
                                                    displacement,
                                                    self.interior_nodes,
                                                    mode='lin_elas')

        output_vals['xy_interior']=xy_interior

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        # return super().compute_derivatives(inputs, outputs, derivatives)()

        #need to return derivatives of interior mesh node 
        #   displacement w.r.t to boundary nodes

        displacement = input_vals['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]

        xy_interior,duhdx = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                        self.boundary_nodes,
                                                        displacement,
                                                        self.interior_nodes,
                                                        get_deriv=True,
                                                        mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
                                                         xy.flatten().shape[0]))

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 15
# W = .1
# H = .1
# points = [[-W/2,-H/2],[W/2, H/2]]

# domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

radius = 1
num_el = 20 #number of elements through wall thickness

domain = ALBATROSS.mesh.create_circle(radius,num_el,'disk')
all_nodes= locate_entities(domain,0,lambda x: np.ones_like(x[0]))
boundary_nodes = locate_entities_boundary(domain,0,lambda x: np.ones_like(x[0]))
interior_nodes = all_nodes[~np.isin(all_nodes, boundary_nodes)]

xy=domain.geometry.x[boundary_nodes,0:2]
xy_interior = domain.geometry.x[interior_nodes,0:2]

print("shape of xy:")
print(xy.shape)
inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
# inputs.xy_prev = csdl.Variable(value=xy,shape=xy.shape,name='xy_prev')
inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

xy = inputs.xy
xy_interior = inputs.xy_interior

inputs.xy.set_as_design_variable(scaler=10)

# displacement = inputs.xy - inputs.xy_prev

#update interior node locations based on boundary motion 
# (uses elliptic smoothing based on Poisson problem)
# domain.geometry.x[interior_nodes,0:2] = ALBATROSS.mesh.smooth_mesh(domain,
#                                                         boundary_nodes,
#                                                         displacement,
#                                                         interior_nodes)

meshSmoothing = EllipticSmoothing(domain,boundary_nodes,interior_nodes)

outputs_mm = meshSmoothing.evaluate(inputs)

inputs.xy_interior = outputs_mm.xy_interior

# inputs.xy_prev = inputs.xy

crosssection = CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2},
                            boundary_nodes=boundary_nodes,
                            interior_nodes=interior_nodes)

#only call one time
outputs = crosssection.evaluate(inputs)

K = outputs.K
A = outputs.A

with csdl.namespace('Objective'):
    f = -K[5,5]
    f.add_name('axial stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = A
    g1.add_name('g1')
    g1.set_as_constraint(upper=3.2) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

print(K.value)
print(A.value)

sim = csdl.experimental.PySimulator(recorder)

print(inputs.xy.value)

print('current K:      ', sim[K])
# print('dKdx(FD):  ', sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=.0001)[K,xy], '\n')
# dKdx_FD = sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=0.002)[K,xy]
dKdx = sim.compute_totals(K,xy)[K,xy]
# diff=dKdx-dKdx_FD

# print('dKdx(FD):  ', dKdx_FD, '\n')
print('dKdx:  ', dKdx, '\n')
# print('diff:', diff)

# print('norms:')
# print(np.linalg.norm(dKdx_FD))
# print(np.linalg.norm(dKdx))
# print(np.linalg.norm(diff))

# sim.check_totals()

# sim[inputs.xy] *=2

# sim.run()

# print('current K:      ', sim[K])

# sim.compute_totals(K,inputs.xy)[K,]

from modopt import CSDLAlphaProblem
from modopt import SLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='bending_stiffness_max',simulator=sim)

optimizer = SLSQP(prob,recording=True,solver_options={'ftol':1e-8, 'maxiter':20})

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0,step=0.01)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)