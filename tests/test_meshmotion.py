import numpy as np
from dolfinx import mesh
from mpi4py import MPI
import ALBATROSS

# ----------------------------
# Create a simple mesh
# ----------------------------
Nx = 6
Ny = 6

msh = mesh.create_unit_square(
    MPI.COMM_WORLD,
    Nx,
    Ny,
    cell_type=mesh.CellType.quadrilateral
)

# required for locate_dofs_topological
msh.topology.create_connectivity(0, msh.topology.dim)

# Save original coordinates
coords0 = msh.geometry.x[:, :2].copy()

# ----------------------------
# Define boundary + interior nodes
# ----------------------------
boundary_nodes = np.where(
    (np.isclose(coords0[:,0],0)) |
    (np.isclose(coords0[:,0],1)) |
    (np.isclose(coords0[:,1],0)) |
    (np.isclose(coords0[:,1],1))
)[0]

interior_nodes = np.setdiff1d(np.arange(coords0.shape[0]), boundary_nodes)

print("Total nodes:", coords0.shape[0])
print("Boundary nodes:", len(boundary_nodes))
print("Interior nodes:", len(interior_nodes))

# ----------------------------
# Initialize mesh motion
# ----------------------------
mesh_motion = ALBATROSS.mesh.MeshMotion(
    msh,
    boundary_nodes,
    interior_nodes
)

# ----------------------------
# TEST 1
# Identity test
# ----------------------------
print("\nTEST 1: Identity boundary condition")

boundary_xy = coords0[boundary_nodes]

xy_interior = mesh_motion.smooth_mesh(boundary_xy)

original_interior = coords0[interior_nodes]

max_error = np.max(np.linalg.norm(
    xy_interior - original_interior,
    axis=1
))

print("Max interior error:", max_error)

if max_error < 1e-12:
    print("PASS: interior nodes unchanged")
else:
    print("FAIL: interior nodes moved")

# ----------------------------
# TEST 2
# Small rigid translation
# ----------------------------
print("\nTEST 2: Small rigid translation")

shift = np.array([0.01, 0.0])

boundary_xy_shifted = coords0[boundary_nodes] + shift

xy_interior = mesh_motion.smooth_mesh(boundary_xy_shifted)

expected = coords0[interior_nodes] + shift

error = np.linalg.norm(xy_interior - expected, axis=1)

print("Mean error:", np.mean(error))
print("Max error :", np.max(error))

# ----------------------------
# TEST 3
# Check solver displacement magnitude
# ----------------------------
print("\nTEST 3: Solver displacement magnitude")

uh = mesh_motion.uh.x.array.reshape((-1, msh.geometry.dim))[:, :2]

print("Max displacement in solver:", np.max(np.linalg.norm(uh, axis=1)))