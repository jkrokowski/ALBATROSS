import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import CellType, GhostMode
from ufl import (CellDiameter, FacetNormal, avg, div, dS, dx, grad, inner,
                 jump, pi, sin)

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

msh = mesh.create_rectangle(comm=MPI.COMM_WORLD,
                            points=((0.0, 0.0), (1.0, 1.0)), n=(32, 32),
                            cell_type=CellType.triangle,
                            ghost_mode=GhostMode.shared_facet)
V = fem.FunctionSpace(msh, ("Lagrange", 2))

facets = mesh.locate_entities_boundary(msh, dim=1,
                                       marker=lambda x: np.logical_or.reduce((
                                           np.isclose(x[0], 0.0),
                                           np.isclose(x[0], 1.0),
                                           np.isclose(x[1], 0.0),
                                           np.isclose(x[1], 1.0))))

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

alpha = ScalarType(8.0)
h = CellDiameter(msh)
n = FacetNormal(msh)
h_avg = (h('+') + h('-')) / 2.0

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 4.0 * pi**4 * sin(pi * x[0]) * sin(pi * x[1])

a = inner(div(grad(u)), div(grad(v))) * dx \
    - inner(avg(div(grad(u))), jump(grad(v), n)) * dS \
    - inner(jump(grad(u), n), avg(div(grad(v)))) * dS \
    + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
L = inner(f, v) * dx

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(msh.comm, "out_biharmonic/biharmonic.xdmf", "w") as file:
    V1 = fem.FunctionSpace(msh, ("Lagrange", 1))
    u1 = fem.Function(V1)
    u1.interpolate(uh)
    file.write_mesh(msh)
    file.write_function(u1)

try:
    import pyvista
    cells, types, x = plot.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(cells, types, x)
    grid.point_data["u"] = uh.x.array.real
    grid.set_active_scalars("u")
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    warped = grid.warp_by_scalar()
    plotter.add_mesh(warped)
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh_biharmonic.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
