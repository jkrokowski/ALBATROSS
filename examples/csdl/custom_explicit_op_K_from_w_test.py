import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs


N = 2
W = 1.2
H = 1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
domain.name = 'square_mesh_opt'
with XDMFFile(MPI.COMM_WORLD, "output/square_mesh_opt.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

# # radius = 1
# # num_el = 40 #number of elements through wall thickness
# # domain = ALBATROSS.mesh.create_circle(radius,num_el,'disk')

#initialize CrossSection object
material = ALBATROSS.material.Material(name='unobtainium',
                            mat_type='ISOTROPIC',
                            mech_props={'E':100.0,'nu':0.2},
                            density=2700)

xs = ALBATROSS.cross_section.CrossSection(domain,[material])

#get imp
xy=domain.geometry.x[xs.boundary_nodes,0:2]
xy_interior = domain.geometry.x[xs.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

# xy = inputs.xy
# xy_interior = inputs.xy_interior



# #CONSTRUCT A BOUNDARY B-SPLINE (with a closed, uniform knot vector)
# num_parametric = 6
# bspline_degree=3
# boundary_spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
# parametric_coords = np.array([(i,) for i in np.linspace(0,1,xs.boundary_nodes.shape[0]+1)])
# boundary_points = csdl.concatenate([xy[list(xs.boundary_ordering)],xy[0:1,:]]) #duplicate the start/endpoint
# boundary_spline_coeffs = boundary_spline_space.fit(values = boundary_points,parametric_coordinates= parametric_coords)
# coeffs = boundary_spline_coeffs.value

# # #TODO: USE A **PERIODIC** B-SPLINE to prevent the corner from being 
# # #make a uniform knot vector of length (num_parametric+6)
# # num_ctrl_pts = num_parametric+bspline_degree*2+2
# # knot_indices = np.arange(0,num_ctrl_pts)
# # num_repeated_ctrl_pts = 3
# # # knots = (knot_indices)/(num_ctrl_pts-1)
# # knots = (knot_indices-bspline_degree)/(num_parametric-1)
# # periodic_bspline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,),knots=knots,knot_indices=knot_indices)
# # parametric_coords2 = np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0])])
# # periodic_boundary_spline_coeffs = periodic_bspline_spaceordered_nodes.fit(values = xy[list(ordering)],parametric_coordinates= parametric_coords2)


# # boundary_spline_coeffs.name = 'boundary spline coeffs'
# inputs.coeffs = csdl.Variable(value=coeffs)
# inputs.coeffs.name = 'boundary spline coeffs'
# inputs.coeffs.set_as_design_variable(lower=-2,upper=2,scaler=2)
# boundary_spline = lfs.Function(boundary_spline_space,inputs.coeffs,name='boundary_spline')
# # evaluated_points = boundary_spline.evaluate(parametric_coords,plot=True)

# #TODO: increase knot multiplicity or use a composite spline for the boundary

# inputs.xy = boundary_spline.evaluate(parametric_coords)[list(xs.inverse_boundary_ordering)]
# inputs.xy.name = 'xy'
# inputs.xy.set_as_design_variable(lower=-2,upper=2,scaler=2)
# inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

# meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(domain,
#                                                        xs.boundary_nodes,
#                                                        xs.interior_nodes)

# outputs_mm = meshSmoothing.evaluate(inputs)

# inputs_w = outputs_mm
# inputs_w.xy = inputs.xy

# warping_model = ALBATROSS.csdl_utils.WarpingFunctionState(xs=xs,
#                         boundary_nodes=xs.boundary_nodes,
#                         interior_nodes=xs.interior_nodes)

# outputs_w = warping_model.evaluate(inputs_w)

xs._get_warping_functions()

section_model = ALBATROSS.csdl_utils.BeamMatrixFromWarping(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

inputs.w = csdl.Variable(value=np.vstack([xs.warping_functions[i].x.array for i in range(6)]).T)
inputs.lmbda = csdl.Variable(value=np.vstack([xs.lmbdas[i].x.array for i in range(6)]).T)
    
outputs_sec = section_model.evaluate(inputs)

# K = outputs.K
# K.name = 'stiffness_mat'
# A = outputs.A
# A.name = 'area'

# with csdl.namespace('Objective'):
#     f = -K[5,5]
#     f.add_name('max_bending_stiffness')
#     f.set_as_objective()

# with csdl.namespace('Area constraint'):
#     g1 = K[0,0]
#     g1.add_name('g1')
#     g1.set_as_constraint(upper=125,lower=75) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()
# sim.check_totals(outputs_w.w[:10,0],inputs.coeffs)

xy_node0 = inputs.xy.get(csdl.slice[0,:])
# reduced_w = inputs.w.get(csdl.slice[25:30,0])
K00 = outputs_sec.K.get(csdl.slice[0,0])


# K00 = outputs_sec.K[0,0]
K00.name = 'K00'

# xy_node0 = inputs.xy[0,:]
xy_node0.name = 'xy_node0'

dKdx_check = sim.check_totals(outputs_sec.K,inputs.xy,step_size=0.000001,print_results=True)

#TODO: currently have accurate derivatives, except it seems that the partials are a bit scrambled for 
for i in range(36):
    print(i,np.linalg.norm(dKdx_check[outputs_sec.K,inputs.xy]['value'][i]-dKdx_check[outputs_sec.K,inputs.xy]['fd_value'][i]))

dK00dx = sim.compute_totals(K00,reduced_xy)
dK00dx_FD = sim.compute_totals(K00,reduced_xy,use_finite_difference=True,finite_difference_step_size=0.001)
dK00dw = sim.compute_totals(K00,reduced_w)
dK00dw_FD = sim.compute_totals(K00,reduced_w,use_finite_difference=True,finite_difference_step_size=0.001)

# sim.check_totals(outputs_sec.K[0,0],inputs.w[25:30,0])
# sim.check_totals(outputs_wf.w[:10,0],inputs.coeffs,step_size=0.0001,print_results=True)
# sim.check_totals(outputs_mm.xy_interior,inputs.coeffs,step_size=0.0001,print_results=True)


#TODO: need to figure out why check totals doesn't seem to affect FD, but does affect normal totals??
# sim.check_totals(outputs_wf.w[:10,0],inputs.xy,step_size=0.0001,print_results=True) 
dwdx = sim.compute_totals(outputs_wf.w[:10,0],inputs.coeffs)
dwdx_FD = sim.compute_totals(outputs_wf.w[:10,0],inputs.xy,use_finite_difference=True,finite_difference_step_size=0.002)

sim.check_totals(outputs_wf.lmbda,inputs.xy,print_results=True)
# sim.check_totals(outputs_wf.w,inputs.xy,print_results=True)

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
# optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)