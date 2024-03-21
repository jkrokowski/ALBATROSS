# https://fenicsproject.discourse.group/t/find-cell-tags-from-two-overlapped-meshes-with-different-resolutions/13198/2from mpi4py import MPI
from mpi4py import MPI
import ufl
import numpy as np
from dolfinx import fem, io, mesh, geometry, cpp

def mark_cells(msh, cell_index):
    num_cells = msh.topology.index_map(
        msh.topology.dim).size_local + msh.topology.index_map(
        msh.topology.dim).num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    values = np.full(cells.shape, 0, dtype=np.int32)
    values[cell_index] = np.full(len(cell_index), 1, dtype=np.int32)
    cell_tag = mesh.meshtags(msh, msh.topology.dim, cells, values)
    return cell_tag


mesh_big = mesh.create_unit_square(MPI.COMM_WORLD, 64, 64)
# mesh_big.geometry.x[:, :2] -= 0.51
# mesh_big.geometry.x[:, :2] *= 4
num_big_cells = mesh_big.topology.index_map(mesh_big.topology.dim).size_local + \
    mesh_big.topology.index_map(mesh_big.topology.dim).num_ghosts


mesh_small = mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
# mesh_small.geometry.x[:, :2] -= 0.5
# mesh_small.geometry.x[:, 0] *= 10
# mesh_small.geometry.x[:, 0:1] *=.99

num_small_cells = mesh_small.topology.index_map(mesh_small.topology.dim).size_local + \
    mesh_small.topology.index_map(mesh_small.topology.dim).num_ghosts

bb_tree = geometry.bb_tree(
    mesh_big, mesh_big.topology.dim, np.arange(num_big_cells, dtype=np.int32))
bb_small = geometry.bb_tree(
    mesh_small, mesh_small.topology.dim, np.arange(num_small_cells, dtype=np.int32))
collisions = geometry.compute_collisions_trees(bb_tree, bb_small)


def extract_cell_geometry(input_mesh, cell: int):
    mesh_nodes = cpp.mesh.entities_to_geometry(
        input_mesh._cpp_object, input_mesh.topology.dim, np.array([cell], dtype=np.int32), False)[0]

    return input_mesh.geometry.x[mesh_nodes]


tol = 1e-13
big_cells = []
small_cells = []
for i, (big_cell, small_cell) in enumerate(collisions):
    geom_small = extract_cell_geometry(mesh_small, small_cell)
    geom_big = extract_cell_geometry(mesh_big, big_cell)
    distance = geometry.compute_distance_gjk(geom_big, geom_small)
    if np.linalg.norm(distance) <= tol:
        big_cells.append(big_cell)
        small_cells.append(small_cell)
cell_tags_big = mark_cells(mesh_big, np.asarray(big_cells, dtype=np.int32))
cell_tags_small = mark_cells(
    mesh_small, np.asarray(small_cells, dtype=np.int32))

with io.XDMFFile(MPI.COMM_WORLD, "cell_tags.xdmf", "w") as file:
    file.write_mesh(mesh_big)
    file.write_meshtags(cell_tags_big, mesh_big.geometry)

with io.XDMFFile(MPI.COMM_WORLD, "mesh_small.xdmf", "w") as file:
    file.write_mesh(mesh_small)
    file.write_meshtags(cell_tags_small, mesh_small.geometry)

# Validation with the area of the overlapped region
dxs = ufl.Measure("dx", domain=mesh_big, subdomain_data=cell_tags_big)
f = 1*dxs(1)
area = MPI.COMM_WORLD.allreduce(
    fem.assemble_scalar(fem.form(f)), op=MPI.SUM)
print("Area:", area)