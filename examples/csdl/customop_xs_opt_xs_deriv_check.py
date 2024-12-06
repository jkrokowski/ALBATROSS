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
        
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

        # if boundary_nodes is not None:
        #     self.boundary_nodes = boundary_nodes

        # if interior_nodes is not None:
        #     self.interior_nodes = interior_nodes

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        # self.declare_input('xy_interior',inputs.xy_interior)
        
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
            # self.domain.geometry.x[self.interior_nodes,0:2]=self.smooth_mesh(input_vals['xy'])

        # else: 
        #     self.domain.geometry.x[:,0:2]=input_vals['xy']

        # print(input_vals['xy'])
        
        xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])
        # xs.plot_mesh()
        if self.xs_analysis_type == 'TS':
            xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            xs.get_xs_stiffness_matrix_EB()
        output_vals['K'] = xs.K
        output_vals['A'] = xs.A
        print("Cross-sectional Area:",xs.A)

    def compute_derivatives(self, input_vals, outputs_vals, derivatives):
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=input_vals['xy_interior']
            # self.domain.geometry.x[self.interior_nodes,0:2]=self.smooth_mesh(input_vals['xy'])
        
        # else: 
        #     self.domain.geometry.x[:,0:2]=input_vals['xy']
        
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

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 10
W = 1
H = 1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
all_nodes= locate_entities(domain,0,lambda x: np.ones_like(x[0]))
boundary_nodes = locate_entities_boundary(domain,0,lambda x: np.ones_like(x[0]))
interior_nodes = all_nodes[~np.isin(all_nodes, boundary_nodes)]

xy=domain.geometry.x[boundary_nodes,0:2]
xy_interior = domain.geometry.x[interior_nodes,0:2]

print("shape of xy:")
print(xy.shape)
inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')


xy = inputs.xy

inputs.xy.set_as_design_variable(scaler=10)

crosssection = CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2},
                            boundary_nodes=boundary_nodes)

#only call one time
outputs = crosssection.evaluate(inputs)

K = outputs.K
A = outputs.A

with csdl.namespace('Objective'):
    f = -K[5,5]+K[0,0] 
    f.add_name('axial stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = A
    g1.add_name('g1')
    g1.set_as_constraint(upper=0.011) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

print(K.value)
print(A.value)

sim = csdl.experimental.PySimulator(recorder)

# print(inputs.xy.value)

print('current K:      ', sim[K])
# print('dKdx(FD):  ', sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=.0001)[K,xy], '\n')
dKdx_FD = sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=0.0001)[K,xy]
dKdx = sim.compute_totals(K,xy)[K,xy]
diff=dKdx-dKdx_FD

print('dKdx(FD):  ', dKdx_FD, '\n')
print('dKdx:  ', dKdx, '\n')
print('diff:', diff)

print('norms:')
print(np.linalg.norm(dKdx_FD))
print(np.linalg.norm(dKdx))
print(np.linalg.norm(diff))

print('main diagonal entries:')
print('dKdx(FD)[0,0]:  ', dKdx_FD[0,:])
print('dKdx(FD)[1,1]:  ', dKdx_FD[7,:])
print('dKdx(FD)[2,2]:  ', dKdx_FD[14,:])
print('dKdx(FD)[3,3]:  ', dKdx_FD[21,:])
print('dKdx(FD)[4,4]:  ', dKdx_FD[28,:])
print('dKdx(FD)[5,5]:  ', dKdx_FD[35,:])

print()