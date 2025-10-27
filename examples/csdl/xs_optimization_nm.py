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
N = 2
offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

H = 1
W = 1
tf = 1/h_to_f
tw = 1/w_to_w

mesh_A = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
mesh_A.geometry.x[:, :2] -= .5
mesh_A.geometry.x[:, 1] *= tf
mesh_A.geometry.x[:, 0] *= W
mesh_A.geometry.x[:, 1] += H/2 - tf/2
mesh_A.name = 'f'
filename_A = 'nonmatching_flange'
# domain.name = filename
with XDMFFile(MPI.COMM_WORLD, "output/"+filename_A+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_A)

mesh_B = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_B.geometry.x[:, :2] -= .5
mesh_B.geometry.x[:, 0] *= tw
mesh_B.geometry.x[:, 1] *= W
mesh_B.name = 'w'
filename_B = 'nonmatching_web'
with XDMFFile(MPI.COMM_WORLD, "output/"+filename_B+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_B)
    
#================= initialize individual cross-sections ===========#
meshes= [mesh_A,mesh_B]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

#================= initialize coupled cross-section ===========#
TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen=1e7)
TXS_nm.plot_meshes()

filename_C = 'mortar_mesh'
with XDMFFile(MPI.COMM_WORLD, "output/"+filename_C+".xdmf", "w") as xdmf:
    xdmf.write_mesh(TXS_nm.collisions[(0,1)].mortar_mesh.msh)

#get boundary orderings for mesh A
xy_A=mesh_A.geometry.x[XSs[0].boundary_nodes,0:2]
xy_A_interior = mesh_A.geometry.x[XSs[0].interior_nodes,0:2]

#get boundary orderings for mesh B
xy_B=mesh_B.geometry.x[XSs[1].boundary_nodes,0:2]
xy_B_interior = mesh_B.geometry.x[XSs[1].interior_nodes,0:2]

#get boundary orderings for mortar mesh
xy_C= TXS_nm.collisions[(0,1)].mortar_mesh.msh.geometry.x[TXS_nm.collisions[(0,1)].mortar_mesh.boundary_nodes,0:2]
xy_C_interior = TXS_nm.collisions[(0,1)].mortar_mesh.msh.geometry.x[TXS_nm.collisions[(0,1)].mortar_mesh.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

#create csdl variables
xy_A_interior = csdl.Variable(value=xy_A_interior,shape=xy_A_interior.shape,name='xy_interior_A')
xy_A = csdl.Variable(value=xy_A,shape=xy_A.shape,name='xy_A')
# xy_A.set_as_design_variable(lower=-1,upper=1,scaler=100)

xy_B_interior = csdl.Variable(value=xy_B_interior,shape=xy_B_interior.shape,name='xy_interior_B')
xy_B = csdl.Variable(value=xy_B,shape=xy_B.shape,name='xy_B')
# xy_B.set_as_design_variable(lower=-1,upper=1,scaler=100)

xy_C_interior = csdl.Variable(value=xy_C_interior,shape=xy_C_interior.shape,name='xy_interior_C')
xy_C = csdl.Variable(value=xy_C,shape=xy_C.shape,name='xy_C')

#web translation parameter
dx_w = csdl.Variable(value=0.15)
dx_w.set_as_design_variable(lower=-0.45,upper=0.45)


#=====mesh motion=======#
inputs_mm_A = csdl.VariableGroup()
inputs_mm_A.xy = xy_A
inputs_mm_A.xy_interior = xy_A_interior
meshSmoothing_A = ALBATROSS.csdl_utils.EllipticSmoothing(mesh_A,
                                                       XSs[0].boundary_nodes,
                                                       XSs[0].interior_nodes,
                                                       filename=filename_A)
outputs_mm_A = meshSmoothing_A.evaluate(inputs_mm_A)

#massively simplified "mesh motion"
xy_B = xy_B + csdl.expand(csdl.concatenate([dx_w,0]),xy_B.shape,action='j->ij')

inputs_mm_B = csdl.VariableGroup()
inputs_mm_B.xy = xy_B
inputs_mm_B.xy_interior = xy_B_interior
meshSmoothing_B = ALBATROSS.csdl_utils.EllipticSmoothing(mesh_B,
                                                       XSs[1].boundary_nodes,
                                                       XSs[1].interior_nodes,
                                                       filename=filename_B)
outputs_mm_B = meshSmoothing_B.evaluate(inputs_mm_B)



#===== mortar mesh update computation =======#
#mortar mesh is moved identically to the web motion:
xy_C = xy_C + csdl.expand(csdl.concatenate([dx_w,0]),xy_C.shape,action='j->ij')

mortar_mesh = TXS_nm.collisions[(0,1)].mortar_mesh 
filename_C = 'mortar_mesh'

inputs_mm_C = csdl.VariableGroup()
inputs_mm_C.xy = xy_C
inputs_mm_C.xy_interior = xy_C_interior
meshSmoothing_C = ALBATROSS.csdl_utils.EllipticSmoothing(mortar_mesh.msh,
                                                       mortar_mesh.boundary_nodes,
                                                       mortar_mesh.interior_nodes,
                                                       filename=filename_C)
outputs_mm_C = meshSmoothing_C.evaluate(inputs_mm_C)


#======= construct coupled system =========#
#get interpolation matrices
inputs_interp = csdl.VariableGroup()
inputs_interp.xy_A = xy_A
inputs_interp.xy_A_interior = xy_A_interior

nonmatchingdata = ALBATROSS.csdl_utils.NonmatchingInterpolationMatrix(xs=TXS_nm)

outputs_interp = nonmatchingdata.evaluate(inputs_interp)
P_A = outputs_interp.P_A
P_B = outputs_interp.P_B

#get mortar mesh assembled matrices




#===== warping function computation =======#




#we have to construct a single operation for the coupled warping function computation:
inputs_w = csdl.VariableGroup()
inputs_w.xy_A = inputs_mm_A.xy
inputs_w.xy_A_interior = outputs_mm_A.xy_interior
inputs_w.xy_B = inputs_mm_B.xy
inputs_w.xy_B_interior = outputs_mm_B.xy_interior
inputs_w.xy_C = inputs_mm_C.xy
inputs_w.xy_C_interior = outputs_mm_C.xy_interior

warping_model = ALBATROSS.csdl_utils.WarpingFunctionStateCoupled(xs=TXS_nm)

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