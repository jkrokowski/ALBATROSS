import numpy as np
from mpi4py import MPI
from dolfinx import mesh, geometry
import ALBATROSS

np.random.seed(2)

# ------------------------------------------------------------
# Build meshes
# ------------------------------------------------------------

mesh_A = mesh.create_unit_square(
    MPI.COMM_WORLD, 6, 2,
    cell_type=mesh.CellType.quadrilateral
)

mesh_A.geometry.x[:, :2] -= 0.5
mesh_A.geometry.x[:,0] *= 1.0
mesh_A.geometry.x[:,1] *= 0.2
mesh_A.geometry.x[:,1] += 0.5 - 0.1


mesh_B = mesh.create_unit_square(
    MPI.COMM_WORLD, 2, 6,
    cell_type=mesh.CellType.quadrilateral
)

mesh_B.geometry.x[:, :2] -= 0.5
mesh_B.geometry.x[:,0] *= 0.2
mesh_B.geometry.x[:,1] *= 1.0


# ------------------------------------------------------------
# Build cross-sections
# ------------------------------------------------------------

mat = ALBATROSS.material.Material(
    name="mat",
    mat_type="ISOTROPIC",
    mech_props={"E":100,"nu":0.2},
    density=1.0
)

XS_A = ALBATROSS.cross_section.CrossSection(mesh_A,[mat])
XS_B = ALBATROSS.cross_section.CrossSection(mesh_B,[mat])

TXS = ALBATROSS.cross_section.CoupledCrossSection(
    [XS_A,XS_B],
    pen_u=1e4,
    pen_t=1e4
)

# ------------------------------------------------------------
# Select scalar subspace
# ------------------------------------------------------------

i = 0
j = 0

V_A, map_A = TXS.XSs[0].V_w.sub(i).sub(j).collapse()
V_C, map_C = TXS.collisions[(0,1)].fxn_space.sub(i).sub(j).collapse()

# ------------------------------------------------------------
# Baseline interpolation matrix
# ------------------------------------------------------------

P_full = TXS._construct_interpolation_operator((0,1),0)
P = ALBATROSS.petsc_utils.convert_petsc_to_numpy(P_full)

P_block = P[np.ix_(map_C,map_A)]

# ------------------------------------------------------------
# Determine rows that are interior to source cells
# ------------------------------------------------------------

x_target = V_C.tabulate_dof_coordinates()
msh0 = V_A.mesh

bb = geometry.bb_tree(msh0,msh0.topology.dim)
cand = geometry.compute_collisions_points(bb,x_target)
cells = geometry.compute_colliding_cells(msh0,cand,x_target)

cell_id = np.array([cells.links(k)[0] for k in range(len(x_target))])

x_ref = np.zeros((len(x_target),2))

for k in range(len(x_target)):
    geom_dofs = msh0.geometry.dofmap[cell_id[k]]
    x_ref[k,:] = msh0.geometry.cmap.pull_back(
        np.array([x_target[k]]),
        msh0.geometry.x[geom_dofs]
    )

tol = 1e-3

interior_rows = np.where(
    (x_ref[:,0] > tol) &
    (x_ref[:,0] < 1-tol) &
    (x_ref[:,1] > tol) &
    (x_ref[:,1] < 1-tol)
)[0]

print("Interior rows:", interior_rows)

rows = interior_rows[:3]

# ------------------------------------------------------------
# random seed restricted to interior rows
# ------------------------------------------------------------

dP = np.zeros_like(P_block)

for r in rows:
    dP[r,:] = np.random.randn(P_block.shape[1])

# ------------------------------------------------------------
# analytic derivative
# ------------------------------------------------------------

dxA_adj, dxC_adj = ALBATROSS.nonmatching_utils.action_of_geom_on_nm_interpolation_matrix(
    V_C,
    V_A,
    dP=dP
)

# ------------------------------------------------------------
# finite difference directional test
# ------------------------------------------------------------

delta_xA = np.random.randn(*dxA_adj.shape)
delta_xC = np.random.randn(*dxC_adj.shape)

eps = 1e-6

xA0 = TXS.XSs[0].msh.geometry.x.copy()
xC0 = TXS.collisions[(0,1)].mortar_mesh.msh.geometry.x.copy()

TXS.XSs[0].msh.geometry.x[:,:2] += eps * delta_xA
TXS.collisions[(0,1)].mortar_mesh.msh.geometry.x[:,:2] += eps * delta_xC

P_plus = ALBATROSS.petsc_utils.convert_petsc_to_numpy(
    TXS._construct_interpolation_operator((0,1),0)
)

P_plus_block = P_plus[np.ix_(map_C,map_A)]

TXS.XSs[0].msh.geometry.x[:] = xA0
TXS.collisions[(0,1)].mortar_mesh.msh.geometry.x[:] = xC0

fd = (P_plus_block - P_block)/eps

lhs = np.sum(dP * fd)

rhs = (
    np.sum(dxA_adj * delta_xA) +
    np.sum(dxC_adj * delta_xC)
)

print("\nAdjoint consistency test")
print("------------------------")
print("FD directional derivative :", lhs)
print("Adjoint prediction        :", rhs)

rel_err = abs(lhs-rhs)/max(abs(lhs),1e-14)

print("Relative error            :", rel_err)