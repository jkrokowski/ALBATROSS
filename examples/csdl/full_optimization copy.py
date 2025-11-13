import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

'''
This optimization problem is not well-posed with just the bending stiffness maximization
A potential way to counter this (without applying constraints on the boundary self-intersections)
would be to add a shear stiffness constraint as well as the area constraint?
the shear stiffness constraint prevents the "web" from necking down and self intersecting

UPDATE: the shear stiffness constraint didn't work because element inversion is not handled well by the cross-section model
maybe this needs to be "fixed" by the mesh smoothing?
'''

N = 10
W = .5
H = .6
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
filename = 'bending_stiffness_maximization'
domain.name = filename
with XDMFFile(MPI.COMM_WORLD, "output/"+filename+".xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

#initialize CrossSection object
material = ALBATROSS.material.Material(name='unobtainium',
                            mat_type='ISOTROPIC',
                            mech_props={'E':100.0,'nu':0.2},
                            density=2700)

xs = ALBATROSS.cross_section.CrossSection(domain,[material])

#get mesh geometry
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

#======= beam model run ==========#
inputs_beam = csdl.VariableGroup()
# inputs_beam = 

beam_model = ALBATROSS.csdl_utils.BeamModel()


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