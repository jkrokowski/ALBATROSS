import csdl_alpha as csdl
import numpy as np
import ALBATROSS
from mpi4py import MPI
from dolfinx import mesh


# ============================================================
# geometry parameters
# ============================================================

dx_w_init = -0.040
EPS = 1e-6

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


# ============================================================
# meshes
# ============================================================

mesh_A = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,
                                 cell_type=mesh.CellType.quadrilateral)
mesh_A.geometry.x[:, :2] -= 0.5
mesh_A.geometry.x[:, 1] *= tf
mesh_A.geometry.x[:, 0] *= W
mesh_A.geometry.x[:, 1] += H/2 - tf/2

mesh_B = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,
                                 cell_type=mesh.CellType.quadrilateral)
mesh_B.geometry.x[:, :2] -= 0.5
mesh_B.geometry.x[:, 0] *= tw
mesh_B.geometry.x[:, 1] *= H


# ============================================================
# materials / cross-section
# ============================================================

mat = ALBATROSS.material.Material(
    name='mat',
    mat_type='ISOTROPIC',
    mech_props={'E':70e9,'nu':0.2},
    density=2700
)

XS_A = ALBATROSS.cross_section.CrossSection(mesh_A,[mat])
XS_B = ALBATROSS.cross_section.CrossSection(mesh_B,[mat])

TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(
    [XS_A, XS_B],
    pen_u=1e2,
    pen_t=1e-1
)

mortar = TXS_nm.collisions[(0,1)].mortar_mesh


# ============================================================
# base coordinates (CONSTANT)
# ============================================================

xy_A0 = mesh_A.geometry.x[XS_A.boundary_nodes,0:2]
xy_Ai0 = mesh_A.geometry.x[XS_A.interior_nodes,0:2]

xy_B0 = mesh_B.geometry.x[XS_B.boundary_nodes,0:2]
xy_Bi0 = mesh_B.geometry.x[XS_B.interior_nodes,0:2]

xy_C0 = mortar.msh.geometry.x[mortar.boundary_nodes,0:2]
xy_Ci0 = mortar.msh.geometry.x[mortar.interior_nodes,0:2]


# ============================================================
# BUILD GRAPH ONCE
# ============================================================

recorder = csdl.Recorder(inline=True)
recorder.start()

dx_w = csdl.Variable(value=np.array([dx_w_init]), name='dx_w')

# --- coordinates ---
xy_A = csdl.Variable(value=xy_A0)
xy_Ai = csdl.Variable(value=xy_Ai0)

xy_B = csdl.Variable(value=xy_B0)
xy_Bi = csdl.Variable(value=xy_Bi0)

xy_C = csdl.Variable(value=xy_C0)
xy_Ci = csdl.Variable(value=xy_Ci0)

# --- rigid shift ---
shift = csdl.concatenate([dx_w, 0.0])

xy_B = xy_B + csdl.expand(shift, xy_B.shape, action='j->ij')
xy_Bi = xy_Bi + csdl.expand(shift, xy_Bi.shape, action='j->ij')

xy_C = xy_C + csdl.expand(shift, xy_C.shape, action='j->ij')
xy_Ci = xy_Ci + csdl.expand(shift, xy_Ci.shape, action='j->ij')

# --- warping solve ---
inputs = csdl.VariableGroup()
inputs.xy_A = xy_A
inputs.xy_A_interior = xy_Ai
inputs.xy_B = xy_B
inputs.xy_B_interior = xy_Bi
inputs.xy_C = xy_C
inputs.xy_C_interior = xy_Ci

warp_op = ALBATROSS.csdl_utils.NonmatchingWarpingFunctionState(
    xs=TXS_nm,
    collision=(0,1)
)

out = warp_op.evaluate(inputs)

wA = out.w_A
wB = out.w_B
lm = out.lmbda

# --- scalar output ---
f = csdl.sum(wA*wA) + csdl.sum(wB*wB) + csdl.sum(lm*lm)
f.add_name('f')


# ============================================================
# simulator
# ============================================================

sim = csdl.experimental.PySimulator(recorder)
# sim.run()

# ============================================================
# sweep setup
# ============================================================

n_samples = 2
dx_vals = np.linspace(-0.048, 0.048, n_samples)

f_vals = []
adj_vals = []
fd_vals = []
abs_err = []
rel_err = []


# ============================================================
# sweep loop
# ============================================================

for i, dx in enumerate(dx_vals):

    # ---------------------------
    # base evaluation
    # ---------------------------
    dx_w.value = np.array([dx])
    sim.run()

    f0 = float(f.value)

    # adjoint at this point
    adj = sim.compute_totals(ofs=[f], wrts=[dx_w])[(f, dx_w)][0][0]

    # ---------------------------
    # finite difference
    # ---------------------------
    dx_w.value = np.array([dx + EPS])
    sim.run()
    f_plus = float(f.value)

    dx_w.value = np.array([dx - EPS])
    sim.run()
    f_minus = float(f.value)

    fd = (f_plus - f_minus) / (2 * EPS)

    # ---------------------------
    # store
    # ---------------------------
    f_vals.append(f0)
    adj_vals.append(adj)
    fd_vals.append(fd)

    err_abs = abs(adj - fd)
    err_rel = err_abs / max(abs(fd), 1e-14)

    abs_err.append(err_abs)
    rel_err.append(err_rel)

    print(f"[{i+1}/{n_samples}] dx={dx: .4f} | f={f0: .3e} | adj={adj: .3e} | fd={fd: .3e} | rel={err_rel: .3e}")


# convert to arrays
f_vals = np.array(f_vals)
adj_vals = np.array(adj_vals)
fd_vals = np.array(fd_vals)
abs_err = np.array(abs_err)
rel_err = np.array(rel_err)

import matplotlib.pyplot as plt

# --------------------------------------------------
# forward solution
# --------------------------------------------------
plt.figure()
plt.plot(dx_vals, f_vals, 'o-')
plt.xlabel('dx_w')
plt.ylabel('f (warping energy)')
plt.title('Forward solution smoothness')
plt.grid()


# --------------------------------------------------
# derivatives comparison
# --------------------------------------------------
plt.figure()
plt.plot(dx_vals, adj_vals, 'o-', label='Adjoint')
plt.plot(dx_vals, fd_vals, 'x--', label='FD')
plt.xlabel('dx_w')
plt.ylabel('df/dx_w')
plt.title('Derivative comparison')
plt.legend()
plt.grid()


# --------------------------------------------------
# error
# --------------------------------------------------
plt.figure()
plt.semilogy(dx_vals, rel_err, 'o-')
plt.xlabel('dx_w')
plt.ylabel('Relative error')
plt.title('Adjoint vs FD error')
plt.grid()


# --------------------------------------------------
# derivative smoothness (second derivative proxy)
# --------------------------------------------------
d_adj = np.gradient(adj_vals, dx_vals)

plt.figure()
plt.plot(dx_vals, d_adj, 'o-')
plt.xlabel('dx_w')
plt.ylabel('d^2f/dx_w^2 (approx)')
plt.title('Derivative smoothness')
plt.grid()

plt.show()

# ============================================================
# derivative check
# ============================================================

# adjoint
adj = sim.compute_totals(ofs=[f], wrts=[dx_w])[(f, dx_w)][0][0]

# FD
dx_w.value = np.array([dx_w_init + EPS])
sim.run()
f_plus = float(f.value)

dx_w.value = np.array([dx_w_init - EPS])
sim.run()
f_minus = float(f.value)

fd = (f_plus - f_minus) / (2*EPS)

# reset
dx_w.value = np.array([dx_w_init])
sim.run()

print("\n==============================")
print("FD check: x_w → warping")
print("==============================")
print(f"f(0)      = {float(f.value): .12e}")
print(f"adjoint   = {float(adj): .12e}")
print(f"FD        = {fd: .12e}")
print(f"abs error = {abs(adj - fd): .12e}")
print(f"rel error = {abs(adj - fd)/max(abs(fd),1e-14): .12e}")
print("==============================\n")