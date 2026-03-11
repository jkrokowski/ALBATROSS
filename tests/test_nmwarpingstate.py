import csdl_alpha as csdl
import numpy as np
import ALBATROSS
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import XDMFFile


# -----------------------------
# geometry parameters
# -----------------------------
dx_w_init = 0.012
eps = 1e-6

N = 4
h_to_f = 10
w_to_w = 10
offset = 1

m1, n1 = N*h_to_f + offset, N
m2, n2 = N, N*w_to_w + offset

H = 0.1
W = 0.1
tf = H/h_to_f
tw = W/w_to_w

directory = 'output/' + str(dx_w_init) + '_N=' + str(N) + '/'

# -----------------------------
# meshes
# -----------------------------
mesh_A = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1, cell_type=mesh.CellType.quadrilateral)
mesh_A.geometry.x[:, :2] -= 0.5
mesh_A.geometry.x[:, 1] *= tf
mesh_A.geometry.x[:, 0] *= W
mesh_A.geometry.x[:, 1] += H/2 - tf/2

mesh_B = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2, cell_type=mesh.CellType.quadrilateral)
mesh_B.geometry.x[:, :2] -= 0.5
mesh_B.geometry.x[:, 0] *= tw
mesh_B.geometry.x[:, 1] *= W


# -----------------------------
# materials
# -----------------------------
mat = ALBATROSS.material.Material(
    name='unobtainium',
    mat_type='ISOTROPIC',
    mech_props={'E':100,'nu':0.33},
    density=2700
)

XS_A = ALBATROSS.cross_section.CrossSection(mesh_A,[mat])
mesh_A.name = 'msh_A_fd'
with XDMFFile(MPI.COMM_WORLD, directory + mesh_A.name + ".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_A)

XS_B = ALBATROSS.cross_section.CrossSection(mesh_B,[mat])
mesh_B.name = 'msh_B_fd'
with XDMFFile(MPI.COMM_WORLD, directory + mesh_B.name + ".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_B)

TXS_nm = ALBATROSS.cross_section.CoupledCrossSection([XS_A,XS_B], pen_u=1e4, pen_t=1e4)
TXS_nm.collisions[(0,1)].mortar_mesh.msh.name = "mortar_msh_C_fd"
with XDMFFile(MPI.COMM_WORLD, directory + TXS_nm.collisions[(0,1)].mortar_mesh.msh.name + ".xdmf", "w") as xdmf:
    xdmf.write_mesh(TXS_nm.collisions[(0,1)].mortar_mesh.msh)
mortar_mesh = TXS_nm.collisions[(0,1)].mortar_mesh


# -----------------------------
# node coordinate sets
# -----------------------------
xy_A0 = mesh_A.geometry.x[XS_A.boundary_nodes,0:2]
xy_Ai0 = mesh_A.geometry.x[XS_A.interior_nodes,0:2]

xy_B0 = mesh_B.geometry.x[XS_B.boundary_nodes,0:2]
xy_Bi0 = mesh_B.geometry.x[XS_B.interior_nodes,0:2]

xy_C0 = mortar_mesh.msh.geometry.x[mortar_mesh.boundary_nodes,0:2]
xy_Ci0 = mortar_mesh.msh.geometry.x[mortar_mesh.interior_nodes,0:2]


# -----------------------------
# function to evaluate f(dx_w)
# -----------------------------
def evaluate_f(dx_val):

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    dx_w = csdl.Variable(value=dx_val,name='dx_w')

    xy_A = csdl.Variable(value=xy_A0)
    xy_Ai = csdl.Variable(value=xy_Ai0)

    xy_B = csdl.Variable(value=xy_B0)
    xy_Bi = csdl.Variable(value=xy_Bi0)

    xy_C = csdl.Variable(value=xy_C0)
    xy_Ci = csdl.Variable(value=xy_Ci0)

    # web translation
    shift_B = csdl.expand(
        csdl.concatenate([dx_w,0.0]),
        xy_B.shape,
        action='j->ij'
    )

    shift_C = csdl.expand(
        csdl.concatenate([dx_w,0.0]),
        xy_C.shape,
        action='j->ij'
    )

    xy_B_moved = xy_B + shift_B
    xy_C_moved = xy_C + shift_C

    # smoothing
    inputs_mm_A = csdl.VariableGroup()
    inputs_mm_A.xy = xy_A
    inputs_mm_A.xy_interior = xy_Ai

    smooth_A = ALBATROSS.csdl_utils.EllipticSmoothing(
        mesh_A,
        XS_A.boundary_nodes,
        XS_A.interior_nodes, 
        filename='msh_A_fd',
        directory=directory
    )

    outA = smooth_A.evaluate(inputs_mm_A)

    inputs_mm_B = csdl.VariableGroup()
    inputs_mm_B.xy = xy_B_moved
    inputs_mm_B.xy_interior = xy_Bi

    smooth_B = ALBATROSS.csdl_utils.EllipticSmoothing(
        mesh_B,
        XS_B.boundary_nodes,
        XS_B.interior_nodes,
        filename='msh_B_fd',
        directory=directory
    )

    outB = smooth_B.evaluate(inputs_mm_B)

    inputs_mm_C = csdl.VariableGroup()
    inputs_mm_C.xy = xy_C_moved
    inputs_mm_C.xy_interior = xy_Ci

    smooth_C = ALBATROSS.csdl_utils.EllipticSmoothing(
        mortar_mesh.msh,
        mortar_mesh.boundary_nodes,
        mortar_mesh.interior_nodes,
        filename='mortar_msh_C_fd',
        directory=directory
    )

    outC = smooth_C.evaluate(inputs_mm_C)

    # implicit warping solve
    inputs_warp = csdl.VariableGroup()

    inputs_warp.xy_A = xy_A
    inputs_warp.xy_A_interior = outA.xy_interior

    inputs_warp.xy_B = xy_B_moved
    inputs_warp.xy_B_interior = outB.xy_interior

    inputs_warp.xy_C = xy_C_moved
    inputs_warp.xy_C_interior = outC.xy_interior

    warp_op = ALBATROSS.csdl_utils.NonmatchingWarpingFunctionState(
        xs=TXS_nm,
        collision=(0,1)
    )

    warp_out = warp_op.evaluate(inputs_warp)

    wA = warp_out.w_A
    wB = warp_out.w_B
    lm = warp_out.lmbda


    # scalar test output
    f = csdl.sum(wA*wA) + csdl.sum(wB*wB) + csdl.sum(lm*lm)
    f.name = 'test_output'

    sim = csdl.experimental.PySimulator(recorder)
    sim.run()

    return f.value, sim, f, dx_w



# -----------------------------
# central difference
# -----------------------------
f0, sim, f, dx = evaluate_f(dx_w_init)

# analytic derivative
totals = sim.compute_totals(ofs=[f], wrts=[dx])
adj_grad = totals[(f,dx)]

# central FD
f_plus,_ ,_,_ = evaluate_f(dx_w_init+eps)
f_minus,_ ,_,_ = evaluate_f(dx_w_init-eps)

fd_grad = (f_plus - f_minus)/(2*eps)

print()
print("Adjoint derivative:", adj_grad)
print("FD derivative     :", fd_grad)
print("Absolute error    :", abs(adj_grad-fd_grad))
print("Relative error    :", abs(adj_grad-fd_grad)/abs(fd_grad))