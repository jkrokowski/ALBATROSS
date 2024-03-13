try:
    from dolfinx import fem, mesh, plot, default_scalar_type
except:
    from petsc4py import PETSc
    default_scalar_type = PETSc.ScalarType    
from dolfinx.fem.petsc import LinearProblem
import numpy
from mpi4py import MPI
from ufl import (Circumradius, FacetNormal, SpatialCoordinate, TrialFunction, TestFunction,
                 div, dx, ds, grad, inner)

#cross-section mesh definition
N = 5 #number of quad elements per side
W = 1 #square height  
H = .1 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
num_el = [10*N,N]

domain = mesh.create_rectangle( MPI.COMM_WORLD,points,num_el, cell_type=mesh.CellType.quadrilateral)
# domain = mesh.create_rectangle( MPI.COMM_WORLD,points,num_el)
V = fem.FunctionSpace(domain, ("Lagrange", 1))

uD = fem.Function(V)
x = SpatialCoordinate(domain)
u_ex =  1 + x[0]**2 + 2 * x[1]**2
uD.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
f = -div(grad(u_ex))

u = TrialFunction(V)
v = TestFunction(V)
n = FacetNormal(domain)
h = 2 * Circumradius(domain)
alpha = fem.Constant(domain, default_scalar_type(10))
a = inner(grad(u), grad(v)) * dx - inner(n, grad(u)) * v * ds
a += - inner(n, grad(v)) * u * ds + alpha / h * inner(u, v) * ds
L = inner(f, v) * dx 
L += - inner(n, grad(v)) * uD * ds + alpha / h * inner(uD, v) * ds

problem = LinearProblem(a, L)
uh = problem.solve()

error_max = domain.comm.allreduce(numpy.max(numpy.abs(uD.x.array-uh.x.array)), op=MPI.MAX)
if domain.comm.rank == 0:
    print(f"Error_max : {error_max:.2e}")

import pyvista
# pyvista.start_xvfb()
try:
    grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
except:
    grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(V))

grid.point_data["u"] = uh.x.array.real
grid.set_active_scalars("u")
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("nitsche.png")