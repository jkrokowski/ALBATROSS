# import scipy
import dolfinx
import numpy as np
import pyvista
import ufl
# import scipy.io
from dolfinx import fem, io, mesh, plot, nls, log
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# from scipy.interpolate import griddata
# from scipy.interpolate import LinearNDInterpolator
# from scipy.io import savemat

# Scaled variable
L = 1
W = 0.1
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

N = 6

import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import mesh, fem, plot, io

# Mesh and function space:
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, W])],
                  [int(L/W)*N,N,N], cell_type=mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

# BCs:
def clamped_boundary(x):
    return np.isclose(x[0], 0)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)
u_D = np.array([0,0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Variational fomulation
T = fem.Constant(domain, ScalarType((0, 0, 0)))
ds = ufl.Measure("ds", domain=domain)
def epsilon(u):
    return ufl.sym(ufl.grad(u)) 
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType((0, 0, -rho*g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

# Solve numerically
problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# Define the exact solution:
I = fem.Constant(domain, (W*(W**3))/12) #second moment of area
E = fem.Constant(domain, (mu*(3*lambda_ + 2*mu)) / (1+mu)) #Youngs mod based on lame params
l = fem.Constant(domain, 1.0) #length of domain, L
f_ex = fem.Constant(domain, ScalarType((0, 0, -rho*g/1029))) #diving force by nb of mesh nodes else it's huge

x = ufl.SpatialCoordinate(domain)
u_exact = (f_ex*x[0]**2 /(24*E*I)) * (x[0]+6*l**2 - 4*l*x[0])

# Interpolate u_exact onto higher dimensional space:
V2 = fem.VectorFunctionSpace(domain, ("CG", 2))
expr = fem.Expression(u_exact, V2.element.interpolation_points())
uex = fem.Function(V2)
uex.interpolate(expr)

# Numerical solution plot
import pyvista
# pyvista.start_xvfb()
# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
   p.show()
else:
   figure_as_array = p.screenshot("deflection.png")


# Exact solution plot
# pyvista.start_xvfb()
# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# Attach vector values to grid and warp grid by vector
grid["uex_1"] = uex.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("uex_1", factor=1.5)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
   p.show()
else:
   figure_as_array = p.screenshot("deflection.png")