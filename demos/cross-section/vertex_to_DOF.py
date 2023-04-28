import ufl
import dolfinx
from mpi4py import MPI
import numpy as np
import scipy


def f(x):
    return x[0] + 3 * x[1]


N = 4
P = 1
Q = 2
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)

el0 = ufl.VectorElement("Lagrange", mesh.ufl_cell(), P)
el1 = ufl.FiniteElement("Lagrange", mesh.ufl_cell(), Q)
el = ufl.MixedElement([el0, el1])
V = dolfinx.fem.FunctionSpace(mesh, el)
u = dolfinx.fem.Function(V)
V0, V0_to_V = V.sub(0).collapse()
dof_layout = V0.dofmap.dof_layout

num_vertices = mesh.topology.index_map(
    0).size_local + mesh.topology.index_map(0).num_ghosts
vertex_to_par_dof_map = np.zeros(num_vertices, dtype=np.int32)
num_cells = mesh.topology.index_map(
    mesh.topology.dim).size_local + mesh.topology.index_map(
    mesh.topology.dim).num_ghosts
c_to_v = mesh.topology.connectivity(mesh.topology.dim, 0)
for cell in range(num_cells):
    vertices = c_to_v.links(cell)
    dofs = V0.dofmap.cell_dofs(cell)
    for i, vertex in enumerate(vertices):
        vertex_to_par_dof_map[vertex] = dofs[dof_layout.entity_dofs(0, i)]

geometry_indices = dolfinx.cpp.mesh.entities_to_geometry(
    mesh, 0, np.arange(num_vertices, dtype=np.int32), False)
bs = V0.dofmap.bs
vtx_to_dof = np.zeros((num_vertices,bs), dtype=np.int32)
for vertex, geom_index in enumerate(geometry_indices):
    par_dof = vertex_to_par_dof_map[vertex]
    for b in range(bs):
        vtx_to_dof[vertex, b] = V0_to_V[par_dof*bs+b]
vtx_to_dof = np.reshape(vtx_to_dof, (-1,1))