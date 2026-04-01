import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx import mesh
from mpi4py import MPI


# ============================================================
# mesh construction
# ============================================================

N = 3
offset = 1

h_to_f = 10
w_to_w = 10

m1, n1 = N*h_to_f + offset, N
m2, n2 = N, N*w_to_w + offset

H = 1.0
W = 1.0
tf = 1 / h_to_f
tw = 1 / w_to_w

mesh_A = mesh.create_unit_square(
    MPI.COMM_WORLD, m1, n1,
    cell_type=mesh.CellType.quadrilateral
)
mesh_A.geometry.x[:, :2] -= 0.5
mesh_A.geometry.x[:, 1] *= tf
mesh_A.geometry.x[:, 0] *= W
mesh_A.geometry.x[:, 1] += H / 2 - tf / 2

mesh_B = mesh.create_unit_square(
    MPI.COMM_WORLD, m2, n2,
    cell_type=mesh.CellType.quadrilateral
)
mesh_B.geometry.x[:, :2] -= 0.5
mesh_B.geometry.x[:, 0] *= tw
mesh_B.geometry.x[:, 1] *= H
mesh_A.geometry.x[:, 0] += 0.24  

# ============================================================
# materials / cross-sections
# ============================================================

mat = ALBATROSS.material.Material(
    name='unobtainium',
    mat_type='ISOTROPIC',
    mech_props={'E': 100, 'nu': 0.2},
    density=2700
)

XSs = [
    ALBATROSS.cross_section.CrossSection(mesh_A, [mat]),
    ALBATROSS.cross_section.CrossSection(mesh_B, [mat])
]

TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(
    XSs,
    pen_u=1e2,
    pen_t=1e2
)

TXS_nm.get_xs_stiffness_matrix()


# ============================================================
# extract baseline data
# ============================================================

xy_A0 = mesh_A.geometry.x[XSs[0].boundary_nodes, 0:2].copy()
xy_Ai0 = mesh_A.geometry.x[XSs[0].interior_nodes, 0:2].copy()

xy_B0 = mesh_B.geometry.x[XSs[1].boundary_nodes, 0:2].copy()
xy_Bi0 = mesh_B.geometry.x[XSs[1].interior_nodes, 0:2].copy()

w_A0 = np.vstack([
    TXS_nm.XSs[0].warping_functions[i].x.array
    for i in range(6)
]).T.copy()

w_B0 = np.vstack([
    TXS_nm.XSs[1].warping_functions[i].x.array
    for i in range(6)
]).T.copy()

lmbda0 = np.vstack([
    TXS_nm.XSs[0].lmbdas[i].x.array
    for i in range(6)
]).T.copy()


# ============================================================
# settings
# ============================================================

EPS = 1e-4
rng = np.random.default_rng(3)

# OPTIONS:
# 'w_A', 'w_B', 'lmbda', 'xA', 'xB'
CHECK_MODE = 'xB'

# OPTIONS:
# 'fro' for global scalarization
# or one of: (0,0), (3,3), (4,4), (1,5)
PROBE = 'fro'
# PROBE = (0,0)
# PROBE = (4,4)
# PROBE = (5,5)


def normalized_seed(shape):
    s = rng.standard_normal(shape)
    return s / np.linalg.norm(s.ravel())


# fixed perturbation direction
if CHECK_MODE == 'w_A':
    seed_global = normalized_seed(w_A0.shape)
elif CHECK_MODE == 'w_B':
    seed_global = normalized_seed(w_B0.shape)
elif CHECK_MODE == 'lmbda':
    seed_global = normalized_seed(lmbda0.shape)
elif CHECK_MODE == 'xA':
    seed_global = normalized_seed(xy_A0.shape)
elif CHECK_MODE == 'xB':
    seed_global = normalized_seed(xy_B0.shape)
else:
    raise ValueError('Invalid CHECK_MODE')

# fixed scalarization seed
S_global = normalized_seed((6, 6))


# ============================================================
# build graph ONCE
# ============================================================

recorder = csdl.Recorder(inline=True)
recorder.start()

alpha = csdl.Variable(value=np.array([0.0]), name='alpha')

# ---- base variables ----
xy_A = csdl.Variable(value=xy_A0, name='xy_A')
xy_Ai = csdl.Variable(value=xy_Ai0, name='xy_Ai')

xy_B = csdl.Variable(value=xy_B0, name='xy_B')
xy_Bi = csdl.Variable(value=xy_Bi0, name='xy_Bi')

w_A = csdl.Variable(value=w_A0, name='w_A')
w_B = csdl.Variable(value=w_B0, name='w_B')
lmbda = csdl.Variable(value=lmbda0, name='lmbda')

# ---- apply one directional perturbation ----
if CHECK_MODE == 'w_A':
    w_A = w_A + alpha * seed_global
    wrt = alpha
    partial_mode = 'w'

elif CHECK_MODE == 'w_B':
    w_B = w_B + alpha * seed_global
    wrt = alpha
    partial_mode = 'w'

elif CHECK_MODE == 'lmbda':
    lmbda = lmbda + alpha * seed_global
    wrt = alpha
    partial_mode = 'w'

elif CHECK_MODE == 'xA':
    xy_A = xy_A + alpha * seed_global
    wrt = alpha
    partial_mode = 'x'

elif CHECK_MODE == 'xB':
    xy_B = xy_B + alpha * seed_global
    wrt = alpha
    partial_mode = 'x'

# ---- assemble inputs ----
inputs = csdl.VariableGroup()
inputs.xy_A = xy_A
inputs.xy_A_interior = xy_Ai
inputs.xy_B = xy_B
inputs.xy_B_interior = xy_Bi
inputs.w_A = w_A
inputs.w_B = w_B
inputs.lmbda = lmbda

# ---- section op ----
section_model = ALBATROSS.csdl_utils.CoupledBeamMatrixFromWarping(
    xs=TXS_nm,
    collision=(0, 1),
    check_partials=partial_mode,
)

outputs = section_model.evaluate(inputs)
K = outputs.K

# ---- scalar probe ----
if PROBE == 'fro':
    phi = csdl.sum(K * S_global)
    phi.add_name('phi_fro')
elif isinstance(PROBE, tuple) and len(PROBE) == 2:
    i, j = PROBE
    phi = K[i, j]
    phi.add_name(f'phi_K_{i}_{j}')
else:
    raise ValueError('Invalid PROBE')

sim = csdl.experimental.PySimulator(recorder)
sim.run()


# ============================================================
# derivative check using ONE simulator
# ============================================================

adj = sim.compute_totals(ofs=[phi], wrts=[wrt])[(phi, wrt)][0][0]

alpha.value = np.array([+EPS])
sim.run()
phi_p = float(phi.value)

alpha.value = np.array([-EPS])
sim.run()
phi_m = float(phi.value)

fd = (phi_p - phi_m) / (2 * EPS)

alpha.value = np.array([0.0])
sim.run()

print("\n==============================")
print("Single-sim block FD check")
print("==============================")
print(f"Mode        = {CHECK_MODE}")
print(f"Probe       = {PROBE}")
print(f"phi(0)      = {float(phi.value): .12e}")
print(f"adjoint     = {float(adj): .12e}")
print(f"FD          = {fd: .12e}")
print(f"abs error   = {abs(adj - fd): .12e}")
print(f"rel error   = {abs(adj - fd)/max(abs(fd), 1e-14): .12e}")
print("==============================\n")