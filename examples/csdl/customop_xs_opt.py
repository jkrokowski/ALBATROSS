import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities,exterior_facet_indices
import lsdo_function_spaces as lfs
from scipy.spatial import cKDTree

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
        output.xy_interior.name = 'xy_interior, output'

        return output

    def compute(self, input_vals, output_vals):
        
        # displacement = input_vals['xy']-input_vals['xy_prev']
        displacement = input_vals['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        
        # #update boundary nodes:
        # self.domain.geometry.x[self.boundary_nodes,0:2]=input_vals['xy']
        
        xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                    self.boundary_nodes,
                                                    displacement,
                                                    self.interior_nodes,
                                                    mode='lin_elas',
                                                    plot_result=True)

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
                                                        plot_result=True,
                                                        get_deriv=True,
                                                        mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
                                                         xy.flatten().shape[0]))

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 40
W = 1
H = 1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

radius = 1
num_el = 20 #number of elements through wall thickness

# domain = ALBATROSS.mesh.create_circle(radius,num_el,'disk')
all_nodes= locate_entities(domain,0,lambda x: np.ones_like(x[0]))
boundary_nodes = locate_entities_boundary(domain,0,lambda x: np.ones_like(x[0]))
interior_nodes = all_nodes[~np.isin(all_nodes, boundary_nodes)]

#order the boundary using a nearest neighbor search:
ordering = order_boundary_nodes(domain.geometry.x[boundary_nodes,0:2])
ordered_vertices = boundary_nodes[ordering]
ordering_inverse_mapping = np.argsort(ordering)

xy=domain.geometry.x[boundary_nodes,0:2]
xy_interior = domain.geometry.x[interior_nodes,0:2]

print("shape of xy:")
print(xy.shape)
inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
# inputs.xy_prev = csdl.Variable(value=xy,shape=xy.shape,name='xy_prev')
inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

xy = inputs.xy
xy_interior = inputs.xy_interior
ordered_boundary = xy[list(ordering)]
ordered_boundary.name = 'ordered boundary'
# inputs.xy.set_as_design_variable(scaler=40)
# displacement = inputs.xy - inputs.xy_prev

#CONSTRUCT A BOUNDARY B-SPLINE
num_parametric = 30
boundary_spline_space = lfs.BSplineSpace(1,(3,),(num_parametric,))
inputs.parametric_coords = csdl.Variable(value=np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0])]),shape=(boundary_nodes.shape[0],1),name='parametric_coords')
parametric_coords = inputs.parametric_coords
# right_boundary=np.sort(xy.value[np.where(xy.value[:,0]==0.5)],axis=0)
# right_boundary=xy.value[np.where(xy.value[:,0]==0.5)]
# boundary_spline_coeffs = boundary_spline_space.fit(values = right_boundary,parametric_coordinates= np.linspace(0,1,right_boundary.shape[0]))
boundary_spline_coeffs = boundary_spline_space.fit(values = ordered_boundary,parametric_coordinates= parametric_coords.value)
boundary_spline_coeffs.name = 'boundary spline coeffs'
boundary_spline = lfs.Function(boundary_spline_space,boundary_spline_coeffs,name='boundary_spline')
# evaluated_points = boundary_spline.evaluate(parametric_coords,plot=True)

#TODO: increase knot multiplicity or use a composite spline for the boundary
#TODO: fit a closed curve (e.g. duplicated end point)
# knots2 = boundary_spline_space.knots
# boundary_spline_space2 = lfs.BSplineSpace(1,(3,),(num_parametric+6,),knots=np.insert(knots2,[10,10,10,20,20,20],[knots2[10],knots2[10],knots2[10],knots2[20],knots2[20],knots2[20]]))
# boundary_spline_coeffs2 = boundary_spline_space2.fit(values = ordered_coords,parametric_coordinates= np.linspace(0,1,boundary_nodes.shape[0]))
# boundary_spline2 = lfs.Function(boundary_spline_space2,boundary_spline_coeffs2)
# evaluated_points2 = boundary_spline2.evaluate(np.array([(i,) for i in np.linspace(0,1,xy.shape[0])]),plot=True)

#set b-spline coefficients (ctrl points) as the design variables
inputs.coeffs = boundary_spline.coefficients
coeffs = inputs.coeffs
inputs.coeffs.set_as_design_variable(scaler=1)
# inputs.parametric_coords = csdl.Variable(value=parametric_coords,shape=parametric_coords.shape,name='parametric_coords')

#use the boundary spline to update the mesh coordinates:
xy = boundary_spline.evaluate(parametric_coords.value)[list(ordering_inverse_mapping)]

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
K.name = 'stiffness_mat'
A = outputs.A
A.name = 'area'

with csdl.namespace('Objective'):
    f = -K[5,5]+0.1*K[0,0]
    f.add_name('max_bend,min_area')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = A
    g1.add_name('g1')
    g1.set_as_constraint(upper=1.2,lower=0.8) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

print(K.value)
print(A.value)

sim = csdl.experimental.PySimulator(recorder)

print('current K:      ', sim[K])
# print('dKdx(FD):  ', sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=.0001)[K,xy], '\n')
# dKdx_FD = sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=0.002)[K,xy]
dKdx = sim.compute_totals(K,xy)[K,xy]
print('Derivatives w.r.t. b-spline ctrl pts')
dKdcoeffs = sim.compute_totals(K,coeffs)
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