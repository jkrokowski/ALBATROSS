import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities,exterior_facet_indices
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

N = 10
W = 1
H = 1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
domain.name = 'square_mesh_opt'
with XDMFFile(MPI.COMM_WORLD, "output/square_mesh_opt.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

# radius = 1
# num_el = 40 #number of elements through wall thickness
# domain = ALBATROSS.mesh.create_circle(radius,num_el,'disk')
all_nodes= locate_entities(domain,0,lambda x: np.ones_like(x[0]))
boundary_nodes = locate_entities_boundary(domain,0,lambda x: np.ones_like(x[0]))
interior_nodes = all_nodes[~np.isin(all_nodes, boundary_nodes)]

#order the boundary using a nearest neighbor search:
ordering = ALBATROSS.csdl_utils.order_boundary_nodes(domain.geometry.x[boundary_nodes,0:2])
ordered_vertices = boundary_nodes[ordering]
inverse_ordering = np.argsort(ordering)

xy=domain.geometry.x[boundary_nodes,0:2]
xy_interior = domain.geometry.x[interior_nodes,0:2]

inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

xy = inputs.xy
xy_interior = inputs.xy_interior

#CONSTRUCT A BOUNDARY B-SPLINE (with a closed, uniform knot vector)
num_parametric = 16
bspline_degree=3
boundary_spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
parametric_coords = np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0]+1)])
boundary_points = csdl.concatenate([xy[list(ordering)],xy[0:1,:]]) #duplicate the start/endpoint
boundary_spline_coeffs = boundary_spline_space.fit(values = boundary_points,parametric_coordinates= parametric_coords)
coeffs = boundary_spline_coeffs.value

# #TODO: USE A **PERIODIC** B-SPLINE to prevent the corner from being 
# #make a uniform knot vector of length (num_parametric+6)
# num_ctrl_pts = num_parametric+bspline_degree*2+2
# knot_indices = np.arange(0,num_ctrl_pts)
# num_repeated_ctrl_pts = 3
# # knots = (knot_indices)/(num_ctrl_pts-1)
# knots = (knot_indices-bspline_degree)/(num_parametric-1)
# periodic_bspline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,),knots=knots,knot_indices=knot_indices)
# parametric_coords2 = np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0])])
# periodic_boundary_spline_coeffs = periodic_bspline_space.fit(values = xy[list(ordering)],parametric_coordinates= parametric_coords2)

# boundary_spline_coeffs.name = 'boundary spline coeffs'
inputs.coeffs = csdl.Variable(value=coeffs*2)
inputs.coeffs.name = 'boundary spline coeffs'
inputs.coeffs.set_as_design_variable(lower=-2,upper=2,scaler=2)
boundary_spline = lfs.Function(boundary_spline_space,inputs.coeffs,name='boundary_spline')
# evaluated_points = boundary_spline.evaluate(parametric_coords,plot=True)

#TODO: increase knot multiplicity or use a composite spline for the boundary
inputs.xy = boundary_spline.evaluate(parametric_coords)[list(inverse_ordering)]

meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(domain,boundary_nodes,interior_nodes)

outputs_mm = meshSmoothing.evaluate(inputs)

inputs.xy_interior = outputs_mm.xy_interior

crosssection = ALBATROSS.csdl_utils.CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2},
                            boundary_nodes=boundary_nodes,
                            interior_nodes=interior_nodes)

#only evaluate once
outputs = crosssection.evaluate(inputs)
K = outputs.K
K.name = 'stiffness_mat'
A = outputs.A
A.name = 'area'

with csdl.namespace('Objective'):
    f = -K[5,5]
    f.add_name('max_bending_stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = K[0,0]
    g1.add_name('g1')
    g1.set_as_constraint(upper=125,lower=75) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)

# print('current K:      ', sim[K])
# # print('dKdx(FD):  ', sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=.0001)[K,xy], '\n')
# # dKdx_FD = sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=0.002)[K,xy]
# dKdx = sim.compute_totals(K,xy)[K,xy]
# print('Derivatives w.r.t. b-spline ctrl pts')
# dKdcoeffs = sim.compute_totals(K,inputs.coeffs)

from modopt import CSDLAlphaProblem
# from modopt import SLSQP
from modopt import PySLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='bending_stiffness_max',simulator=sim)

# optimizer = SLSQP(prob,recording=True,solver_options={'ftol':1e-8, 'maxiter':20})
optimizer = PySLSQP(prob,recording=True,solver_options={'maxiter':20,'acc':1e-6,'iprint':2})

# Check first derivatives at the initial guess, if needed
optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)