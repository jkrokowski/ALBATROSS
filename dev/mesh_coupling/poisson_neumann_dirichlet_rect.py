try:
    from dolfinx import fem, mesh, plot, default_scalar_type
except:
    from petsc4py import PETSc
    default_scalar_type = PETSc.ScalarType   
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_rectangle
try:
    from dolfinx.plot import vtk_mesh
except:
    from dolfinx.plot import create_vtk_mesh as vtk_mesh
  

from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad

import numpy as np
import pyvista
w=1
h=0.1
mesh = create_rectangle(MPI.COMM_WORLD,((0,-h/2),(w, h/2)), [100, 10])

V = FunctionSpace(mesh, ("Lagrange", 1))
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx

def u_exact(x):
    print(x[0].shape)
    return np.ones_like(x[0])
    # return 1+x[0] 
def boundary_D(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], w))


dofs_D = locate_dofs_geometrical(V, boundary_D)
u_bc = Function(V)
u_bc.interpolate(u_exact)
bc = dirichletbc(u_bc, dofs_D)

x = SpatialCoordinate(mesh)
g = -4
f = Constant(mesh, default_scalar_type(-6))
L = f * v * dx - g * v * ds

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

V2 = FunctionSpace(mesh, ("Lagrange", 2))
uex = Function(V2)
uex.interpolate(u_exact)
error_L2 = assemble_scalar(form((uh - uex)**2 * dx))
error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2, op=MPI.SUM))

u_vertex_values = uh.x.array
uex_1 = Function(V)
uex_1.interpolate(uex)
u_ex_vertex_values = uex_1.x.array
error_max = np.max(np.abs(u_vertex_values - u_ex_vertex_values))
error_max = MPI.COMM_WORLD.allreduce(error_max, op=MPI.MAX)
print(f"Error_L2 : {error_L2:.2e}")
print(f"Error_max : {error_max:.2e}")

# pyvista.start_xvfb()

pyvista_cells, cell_types, geometry = vtk_mesh(V)
grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geometry)
grid.point_data["u"] = uh.x.array
grid.set_active_scalars("u")

plotter = pyvista.Plotter()
plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
plotter.add_mesh(grid, show_edges=True)
plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("neumann_dirichlet.png")