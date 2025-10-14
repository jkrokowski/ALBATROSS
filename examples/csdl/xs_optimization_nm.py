import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx import mesh, io
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs

'''
This optimization problem is not well-posed with just the bending stiffness maximization
A potential way to counter this (without applying constraints on the boundary self-intersections)
would be to add a shear stiffness constraint as well as the area constraint?
the shear stiffness constraint prevents the "web" from necking down and self intersecting

UPDATE: the shear stiffness constraint didn't work because element inversion is not handled well by the cross-section model
maybe this needs to be "fixed" by the mesh smoothing?
'''

#=================== mesh construction ==================#
N = 4
offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

H = 1
W = 1
tf = 1/h_to_f
tw = 1/w_to_w

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2
mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
mesh_1.name = 'w'

#================= initialize individual cross-sections ===========#
meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

#================= initialize coupled cross-section ===========#
TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen=1e7)
TXS_nm.plot_meshes()

#get boundary orderings
xy=domain.geometry.x[xs.boundary_nodes,0:2]
xy_interior = domain.geometry.x[xs.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

#create interior node variable
xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')
xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
xy.set_as_design_variable(lower=-1,upper=1,scaler=100)

#=====mesh motion=======#
inputs_mm = csdl.VariableGroup()
inputs_mm.xy = xy
inputs_mm.xy_interior = xy_interior
meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(domain,
                                                       xs.boundary_nodes,
                                                       xs.interior_nodes,
                                                       filename=filename)
outputs_mm = meshSmoothing.evaluate(inputs_mm)

#===== warping function computation =======#
inputs_w = csdl.VariableGroup()
inputs_w.xy = inputs_mm.xy
inputs_w.xy_interior = outputs_mm.xy_interior

warping_model = ALBATROSS.csdl_utils.WarpingFunctionState(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

outputs_w = warping_model.evaluate(inputs_w)

#======= cross-section stiffness matrix ==========#
section_model = ALBATROSS.csdl_utils.BeamMatrixFromWarping(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

inputs_sec = csdl.VariableGroup()
inputs_sec.xy = xy
inputs_sec.xy_interior = outputs_mm.xy_interior
inputs_sec.w = outputs_w.w
inputs_sec.lmbda = outputs_w.lmbda

outputs_sec = section_model.evaluate(inputs_sec)

K = outputs_sec.K
# K.name = 'stiffness_mat'
A = outputs_sec.A
# A.name = 'area'

with csdl.namespace('Objective'):
    f = -K[5,5]
    f.add_name('max_bending_stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = K[0,0]
    g1.add_name('g1')
    g1.set_as_constraint(upper=35,lower=25) # constraint

with csdl.namespace('Shear constraint'):
    g2 = K[2,2]
    g2.add_name('g2')
    g2.set_as_constraint(lower=4) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()

# recorder.visualize_adjacency_matrix()

#uncommment this to check the total derivatives of the pipeline
# sim.check_totals(outputs_sec.K,inputs.coeffs)

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
optimizer = PySLSQP(prob,recording=True,solver_options={'maxiter':100,'acc':1e-6,'iprint':2})

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)