import dolfinx
from mpi4py import MPI
from typing import List
import numpy as np
import pyvista
import ufl

def deform_mesh(V, bcs: List[dolfinx.fem.DirichletBCMetaClass]):
    mesh = V.mesh
    u = dolfinx.fem.Function(V)
    dolfinx.fem.petsc.set_bc(u.vector, bcs)
    deformation_array = u.x.array.reshape((-1, mesh.geometry.dim))
    mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array

def deform_mesh_poisson(V, bcs: List[dolfinx.fem.DirichletBCMetaClass]):
    mesh = V.mesh
    uh = dolfinx.fem.Function(V)
    dolfinx.fem.petsc.set_bc(uh.vector, bcs)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, (0., 0.)), v)*ufl.dx
    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs, uh)
    problem.solve()
    deformation_array = uh.x.array.reshape((-1, mesh.geometry.dim))
    mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array



mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD,  10, 10)

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = mesh.topology.dim
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    plotter.show_axes()
    if not pyvista.OFF_SCREEN:
        plotter.show()


c_el = mesh.ufl_domain().ufl_coordinate_element()
V = dolfinx.fem.FunctionSpace(mesh, c_el)


def moving_boundaries(x):
    return np.isclose(x[1], 1) | np.isclose(x[0], 1)


mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
all_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
top_facets = dolfinx.mesh.locate_entities_boundary(
    mesh, mesh.topology.dim-1, moving_boundaries)

bc_fixed_facets = []
for facet in all_facets:
    if facet not in top_facets:
        bc_fixed_facets.append(facet)

fixed_dofs = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim-1, bc_fixed_facets)
c = dolfinx.fem.Constant(mesh, (0., 0.))
bc_fixed = dolfinx.fem.dirichletbc(c, fixed_dofs, V)


def radial(x):
    return (x[0], x[1])


u_radial = dolfinx.fem.Function(V)
u_radial.interpolate(radial)
top_dofs = dolfinx.fem.locate_dofs_topological(
    V, mesh.topology.dim-1, top_facets)
bc = dolfinx.fem.dirichletbc(u_radial, top_dofs)

bcs = [bc_fixed, bc]
deform_mesh_poisson(V, bcs)


if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = mesh.topology.dim
    topology, cell_types, geometry = dolfinx.plot.create_vtk_mesh(mesh, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    plotter.show_axes()
    if not pyvista.OFF_SCREEN:
        plotter.show()

with dolfinx.io.XDMFFile(mesh.comm, "mesh_deformed.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)