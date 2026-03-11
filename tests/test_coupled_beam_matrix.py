import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx import mesh
from mpi4py import MPI

# ============================================================
# mesh construction (smallest possible for speed)
# ============================================================

N = 1
offset = 1

h_to_f = 6
w_to_w = 6

m1, n1 = N*h_to_f+offset, N
m2, n2 = N, N*w_to_w+offset

H = 1.0
W = 1.0
tf = 1/h_to_f
tw = 1/w_to_w

mesh_A = mesh.create_unit_square(
    MPI.COMM_WORLD, m1, n1,
    cell_type=mesh.CellType.quadrilateral
)
mesh_A.geometry.x[:, :2] -= .5
mesh_A.geometry.x[:,1] *= tf
mesh_A.geometry.x[:,0] *= W
mesh_A.geometry.x[:,1] += H/2 - tf/2

mesh_B = mesh.create_unit_square(
    MPI.COMM_WORLD, m2, n2,
    cell_type=mesh.CellType.quadrilateral
)
mesh_B.geometry.x[:, :2] -= .5
mesh_B.geometry.x[:,0] *= tw
mesh_B.geometry.x[:,1] *= W

# ============================================================
# materials / cross-sections
# ============================================================

unobtainium = ALBATROSS.material.Material(
    name='unobtainium',
    mat_type='ISOTROPIC',
    mech_props={'E':100,'nu':0.2},
    density=2700
)

XSs = [
    ALBATROSS.cross_section.CrossSection(mesh_A,[unobtainium]),
    ALBATROSS.cross_section.CrossSection(mesh_B,[unobtainium])
]

TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(
    XSs,
    pen_u=1e4,
    pen_t=1e4
)

TXS_nm.get_xs_stiffness_matrix()

# ============================================================
# pull coordinate orderings
# ============================================================

xy_A = mesh_A.geometry.x[XSs[0].boundary_nodes,0:2]
xy_Ai = mesh_A.geometry.x[XSs[0].interior_nodes,0:2]

xy_B = mesh_B.geometry.x[XSs[1].boundary_nodes,0:2]
xy_Bi = mesh_B.geometry.x[XSs[1].interior_nodes,0:2]

xy_C = TXS_nm.collisions[(0,1)].mortar_mesh.msh.geometry.x[
    TXS_nm.collisions[(0,1)].mortar_mesh.boundary_nodes,0:2]

xy_Ci = TXS_nm.collisions[(0,1)].mortar_mesh.msh.geometry.x[
    TXS_nm.collisions[(0,1)].mortar_mesh.interior_nodes,0:2]

# ============================================================
# recorder
# ============================================================

recorder = csdl.Recorder(inline=True)
recorder.start()

# ============================================================
# CSDL variables
# ============================================================

xy_A = csdl.Variable(value=xy_A, name='xy_A')
xy_Ai = csdl.Variable(value=xy_Ai, name='xy_Ai')

xy_B = csdl.Variable(value=xy_B, name='xy_B')
xy_Bi = csdl.Variable(value=xy_Bi, name='xy_Bi')

xy_C = csdl.Variable(value=xy_C, name='xy_C')
xy_Ci = csdl.Variable(value=xy_Ci, name='xy_Ci')

# ============================================================
# warping states
# ============================================================

w_A = np.vstack([
    TXS_nm.XSs[0].warping_functions[i].x.array
    for i in range(6)
]).T

w_B = np.vstack([
    TXS_nm.XSs[1].warping_functions[i].x.array
    for i in range(6)
]).T

lmbda = np.vstack([
    TXS_nm.XSs[0].lmbdas[i].x.array
    for i in range(6)
]).T

w_A = csdl.Variable(value=w_A, name='w_A')
w_B = csdl.Variable(value=w_B, name='w_B')
lmbda = csdl.Variable(value=lmbda, name='lmbda')

# ============================================================
# warping slice variable (for clean FD checks)
# ============================================================

start = 0
end = min(6, w_A.value.shape[0])
wf_num = 0

warping_slice_A = csdl.Variable(
    value=w_A.value[start:end,wf_num],
    name='warping_slice'
)

w_A = w_A.set(
    csdl.slice[start:end,wf_num],
    warping_slice_A
)


warping_slice_B = csdl.Variable(
    value=w_B.value[start:end,wf_num],
    name='warping_slice'
)

w_B = w_B.set(
    csdl.slice[start:end,wf_num],
    warping_slice_B
)
# ============================================================
# assemble inputs
# ============================================================

inputs_sec = csdl.VariableGroup()

inputs_sec.xy_A = xy_A
inputs_sec.xy_A_interior = xy_Ai

inputs_sec.xy_B = xy_B
inputs_sec.xy_B_interior = xy_Bi

inputs_sec.xy_C = xy_C
inputs_sec.xy_C_interior = xy_Ci

inputs_sec.w_A = w_A
inputs_sec.w_B = w_B
inputs_sec.lmbda = lmbda

# ============================================================
# choose which derivative block to test
# ============================================================

# options: 'w', 'xA', 'xB', 'xC'
# CHECK_MODE = 'w'
CHECK_MODE = 'xA'
# CHECK_MODE = 'xC'
# CHECK_MODE = 'xC'

section_model = ALBATROSS.csdl_utils.CoupledBeamMatrixFromWarping(
    xs=TXS_nm,
    collision=(0,1),
    check_partials=CHECK_MODE
)

outputs = section_model.evaluate(inputs_sec)
K = outputs.K

# scalarize output for cleaner FD
phi = K[0,0]+ K[4,4] + K[1,5] - K[5,3] +K[5,5]
phi.add_name('phi')

# ============================================================
# simulator
# ============================================================

sim = csdl.experimental.PySimulator(recorder)
# sim.run()

print("phi =", float(phi.value))

# ============================================================
# derivative checks
# ============================================================

if CHECK_MODE == 'w':

    print("\nChecking warping derivatives")

    # sim.check_totals(phi, warping_slice_A, step_size=1e-8)
    sim.check_totals(phi, warping_slice_B, step_size=1e-8)
    print()
    # sim.check_totals(K, inputs_sec.w_A, step_size=1e-6)
    # sim.check_totals(K, inputs_sec.w_B, step_size=1e-6)
    # sim.check_totals(K, inputs_sec.lmbda, step_size=1e-6)

elif CHECK_MODE == 'xA':

    print("\nChecking flange geometry derivatives")

    sim.check_totals(phi, xy_A, step_size=1e-6)
    sim.check_totals(phi, xy_Ai, step_size=1e-6)
    print()
    
elif CHECK_MODE == 'xB':

    print("\nChecking web geometry derivatives")

    sim.check_totals(phi, xy_B, step_size=1e-6)
    sim.check_totals(phi, xy_Bi, step_size=1e-6)

elif CHECK_MODE == 'xC':

    print("\nChecking mortar geometry derivatives")

    sim.check_totals(phi, xy_C, step_size=1e-6)
    sim.check_totals(phi, xy_Ci, step_size=1e-6)

print("\nDone.")