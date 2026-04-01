import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx import mesh
from mpi4py import MPI


# ============================================================
# mesh + cross section (same as before, minimal)
# ============================================================

N = 3
h_to_f = 10
w_to_w = 25
offset = 0

H = 0.03
W = 0.1

tf = H/h_to_f
tw = W/w_to_w

tw_to_tf = int(tw/tf)

h_to_w = int(H/tw)
w_to_f = int(W/tf)

m1,n1 = N*w_to_f+offset,N
m2,n2 = tw_to_tf*N,N*tw_to_tf*h_to_w+offset

mesh_A = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,
                                 cell_type=mesh.CellType.quadrilateral)
mesh_A.geometry.x[:, :2] -= 0.5
mesh_A.geometry.x[:,1] *= tf
mesh_A.geometry.x[:,0] *= W
mesh_A.geometry.x[:,1] += H/2 - tf/2

mesh_B = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,
                                 cell_type=mesh.CellType.quadrilateral)
mesh_B.geometry.x[:, :2] -= 0.5
mesh_B.geometry.x[:,0] *= tw
mesh_B.geometry.x[:,1] *= H


mat = ALBATROSS.material.Material(
    name='mat',
    mat_type='ISOTROPIC',
    mech_props={'E':70e9,'nu':0.33},
    density=2700
)

XSs = [
    ALBATROSS.cross_section.CrossSection(mesh_A,[mat]),
    ALBATROSS.cross_section.CrossSection(mesh_B,[mat])
]

TXS = ALBATROSS.cross_section.CoupledCrossSection(
    XSs,
    pen_u=1e2,
    pen_t=1e-1
)

TXS.get_xs_stiffness_matrix()


# ============================================================
# extract geometry
# ============================================================

def extract_coords(XS):
    mesh = XS.msh
    b = XS.boundary_nodes
    i = XS.interior_nodes
    return mesh.geometry.x[b,0:2], mesh.geometry.x[i,0:2]

xy_A0, xy_Ai0 = extract_coords(XSs[0])
xy_B0, xy_Bi0 = extract_coords(XSs[1])

mortar = TXS.collisions[(0,1)].mortar_mesh
xy_C0 = mortar.msh.geometry.x[mortar.boundary_nodes,0:2]
xy_Ci0 = mortar.msh.geometry.x[mortar.interior_nodes,0:2]


# ============================================================
# global scalarization seed
# ============================================================

rng = np.random.default_rng(2)
S = rng.standard_normal((6,6))
S /= np.linalg.norm(S)


# ============================================================
# build graph ONCE
# ============================================================

recorder = csdl.Recorder(inline=True)
recorder.start()

x_w = csdl.Variable(value=np.array([0.0]), name='x_w')

xy_A = csdl.Variable(value=xy_A0)
xy_Ai = csdl.Variable(value=xy_Ai0)

xy_B = csdl.Variable(value=xy_B0)
xy_Bi = csdl.Variable(value=xy_Bi0)

xy_C = csdl.Variable(value=xy_C0)
xy_Ci = csdl.Variable(value=xy_Ci0)

# ---- rigid shift ----
shift = csdl.concatenate([x_w, 0.0])

xy_B = xy_B + csdl.expand(shift, xy_B.shape, action='j->ij')
xy_Bi = xy_Bi + csdl.expand(shift, xy_Bi.shape, action='j->ij')

xy_C = xy_C + csdl.expand(shift, xy_C.shape, action='j->ij')
xy_Ci = xy_Ci + csdl.expand(shift, xy_Ci.shape, action='j->ij')

# ---- warping solve ----
inputs_warp = csdl.VariableGroup()
inputs_warp.xy_A = xy_A
inputs_warp.xy_A_interior = xy_Ai
inputs_warp.xy_B = xy_B
inputs_warp.xy_B_interior = xy_Bi
inputs_warp.xy_C = xy_C
inputs_warp.xy_C_interior = xy_Ci

warp_model = ALBATROSS.csdl_utils.NonmatchingWarpingFunctionState(
    xs=TXS,
    collision=(0,1),
)
warp_out = warp_model.evaluate(inputs_warp)

# ---- stiffness matrix ----
inputs_sec = csdl.VariableGroup()
inputs_sec.xy_A = xy_A
inputs_sec.xy_A_interior = xy_Ai
inputs_sec.xy_B = xy_B
inputs_sec.xy_B_interior = xy_Bi
inputs_sec.w_A = warp_out.w_A
inputs_sec.w_B = warp_out.w_B
inputs_sec.lmbda = warp_out.lmbda

sec_model = ALBATROSS.csdl_utils.CoupledBeamMatrixFromWarping(
    xs=TXS,
    collision=(0,1)
)

K = sec_model.evaluate(inputs_sec).K

phi = csdl.sum(K * S)
phi.add_name('phi')


# ============================================================
# simulator
# ============================================================

sim = csdl.experimental.PySimulator(recorder)
sim.run()


# ============================================================
# FD check
# ============================================================

EPS = 1e-7

# adjoint
adj = sim.compute_totals(ofs=[phi], wrts=[x_w])[(phi, x_w)][0][0]

# FD
x_w.value = np.array([+EPS])
sim.run()
phi_p = float(phi.value)

x_w.value = np.array([-EPS])
sim.run()
phi_m = float(phi.value)

fd = (phi_p - phi_m) / (2*EPS)

# reset
x_w.value = np.array([0.0])
sim.run()

print("\n==============================")
print("FD check: x_w → K")
print("==============================")
print(f"phi(0)      = {float(phi.value): .12e}")
print(f"adjoint     = {float(adj): .12e}")
print(f"FD          = {fd: .12e}")
print(f"abs error   = {abs(adj - fd): .12e}")
print(f"rel error   = {abs(adj - fd)/max(abs(fd),1e-14): .12e}")
print("==============================\n")