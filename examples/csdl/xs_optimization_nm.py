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

m1,n1 = N*h_to_f,N
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
dx_w = csdl.Variable(value=-0.45)
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
#get foreground mesh A values
inputs_A = csdl.VariableGroup()
inputs_A.xy = xy_A
inputs_A.xy_interior = outputs_mm_A.xy_interior

conformal_problem_A = ALBATROSS.csdl_utils.CrossSectionSystemComponents(xs=TXS_nm,
                                                                        mesh_id=0,
                                                                        boundary_nodes=XSs[0].boundary_nodes,
                                                                        interior_nodes=XSs[0].interior_nodes)

outputs_A = conformal_problem_A.evaluate(inputs_A)

K_A = outputs_A.K
C_A = outputs_A.C
F_A = csdl.Variable(value=np.zeros(K_A.shape[0]))
# F_A = outputs_A.F # this is just vector of zeros, does not contain the lagrange multiplier values

#get foreground mesh B values
inputs_B = csdl.VariableGroup()
inputs_B.xy = xy_B
inputs_B.xy_interior = outputs_mm_B.xy_interior

conformal_problem_B = ALBATROSS.csdl_utils.CrossSectionSystemComponents(xs=TXS_nm,
                                                                        mesh_id=1,
                                                                        boundary_nodes=XSs[1].boundary_nodes,
                                                                        interior_nodes=XSs[1].interior_nodes)

outputs_B = conformal_problem_B.evaluate(inputs_B)

K_B = outputs_B.K
C_B = outputs_B.C
F_B = csdl.Variable(value=np.zeros(K_B.shape[0]))
#F_B = outputs_B.F_B

#get interpolation matrices
inputs_interp_A = csdl.VariableGroup()
inputs_interp_A.xy_foreground = xy_A
inputs_interp_A.xy_interior_foreground = outputs_mm_A.xy_interior
inputs_interp_A.xy_mortar = xy_C
inputs_interp_A.xy_interior_mortar = outputs_mm_C.xy_interior

nonmatchingdata_A = ALBATROSS.csdl_utils.NonmatchingInterpolationMatrix(xs=TXS_nm,
                                                                        mesh_id=0,
                                                                        collision=(0,1),
                                                                        foreground_boundary = XSs[0].boundary_nodes,
                                                                        foreground_interior = XSs[0].interior_nodes,
                                                                        mortar_boundary = mortar_mesh.boundary_nodes,
                                                                        mortar_interior = mortar_mesh.interior_nodes)

outputs_interp_A = nonmatchingdata_A.evaluate(inputs_interp_A)

P_A = outputs_interp_A.P

inputs_interp_B = csdl.VariableGroup()
inputs_interp_B.xy_foreground = xy_B
inputs_interp_B.xy_interior_foreground = outputs_mm_B.xy_interior
inputs_interp_B.xy_mortar = xy_C
inputs_interp_B.xy_interior_mortar = outputs_mm_C.xy_interior

nonmatchingdata_B = ALBATROSS.csdl_utils.NonmatchingInterpolationMatrix(xs=TXS_nm,
                                                                        mesh_id=1,
                                                                        collision=(0,1),
                                                                        foreground_boundary = XSs[1].boundary_nodes,
                                                                        foreground_interior = XSs[1].interior_nodes,
                                                                        mortar_boundary = mortar_mesh.boundary_nodes,
                                                                        mortar_interior = mortar_mesh.interior_nodes)

outputs_interp_B = nonmatchingdata_B.evaluate(inputs_interp_B)

P_B = outputs_interp_B.P

#get mortar mesh coupling matrices
inputs_C = csdl.VariableGroup()
inputs_C.xy = xy_C
inputs_C.xy_interior = outputs_mm_C.xy_interior

coupling_terms = ALBATROSS.csdl_utils.CrossSectionCouplingComponents(xs=TXS_nm,
                                                                     collision=(0,1),
                                                                     boundary_nodes=mortar_mesh.boundary_nodes,
                                                                     interior_nodes=mortar_mesh.interior_nodes)

outputs_C = coupling_terms.evaluate(inputs_C)

M_C = outputs_C.MC
S_C = outputs_C.SC
# F_C = outputs_C.F_C

#===== warping function computation =======#
#blocked system:
A00 = K_A + P_A.T() @ (M_C + S_C) @P_A
A01 = -P_A.T() @ (M_C + S_C) @ P_B
A10 = -P_B.T() @ (M_C + S_C) @ P_A
A11 = K_B + P_B.T() @ (M_C + S_C) @ P_B

A22 = csdl.Variable(value=np.zeros((C_A.shape[0],C_A.shape[0])))

A = csdl.blockmat([[A00,A01,C_A.T()],
                      [A10,A11,C_B.T()],
                      [C_A,C_B,A22]])

# warping_solutions = csdl.Variable(shape=(A.shape[0],6))

#compute using the petsc based solve for verification:
TXS_nm._get_warping_functions()

dense_error = 1e-5
warping_solutions = []
for i in range(6):
    F_C = csdl.Variable(value=TXS_nm.XSs[0]._return_rhs_vec(i)) 
    b = csdl.concatenate((F_A,F_B,F_C))
    warping_solution=csdl.solve_linear(A,b)
    warping_solutions.append(warping_solution)

    #check that the csdl linear solve is the same as the petsc solution:
    assert dense_error>np.linalg.norm(np.concatenate([TXS_nm.XSs[0].warping_functions[i].x.array,TXS_nm.XSs[1].warping_functions[i].x.array,TXS_nm.XSs[0].lmbdas[i].x.array])-warping_solution.value)

w_A_list= []
w_B_list = []
lmbda_list = []
for i in range(6):
    w_A_list.append(warping_solutions[i].get(csdl.slice[:A00.shape[0]]))
    assert dense_error>np.linalg.norm(TXS_nm.XSs[0].warping_functions[i].x.array-w_A_list[i].value)

    w_B_list.append(warping_solutions[i].get(csdl.slice[A00.shape[0]:A00.shape[0]+A11.shape[0]]))
    assert dense_error>np.linalg.norm(TXS_nm.XSs[1].warping_functions[i].x.array-w_B_list[i].value)

    lmbda_list.append(warping_solutions[i].get(csdl.slice[A00.shape[0]+A11.shape[0]:]))
    assert dense_error>np.linalg.norm(TXS_nm.XSs[0].lmbdas[i].x.array-lmbda_list[i].value)

w_A = csdl.vstack(w_A_list).T()
w_B = csdl.vstack(w_B_list).T()
lmbdas =  csdl.vstack(lmbda_list).T()

# TXS_nm.get_xs_stiffness_matrix()
# Kcheck = TXS_nm.K 

#======= cross-section stiffness matrix ==========#
inputs_sec = csdl.VariableGroup()
inputs_sec.xy_A = inputs_mm_A.xy
inputs_sec.xy_A_interior = outputs_mm_A.xy_interior
inputs_sec.xy_B = inputs_mm_B.xy
inputs_sec.xy_B_interior = outputs_mm_B.xy_interior
inputs_sec.xy_C = inputs_mm_C.xy
inputs_sec.xy_C_interior = outputs_mm_C.xy_interior
inputs_sec.w_A = w_A
inputs_sec.w_B = w_B
inputs_sec.lmbda = lmbdas

section_model = ALBATROSS.csdl_utils.CoupledBeamMatrixFromWarping(xs=TXS_nm,
                                                                  collision=(0,1))

outputs_sec = section_model.evaluate(inputs_sec)

K = outputs_sec.K
# K.name = 'stiffness_mat'
# A = outputs_sec.A
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