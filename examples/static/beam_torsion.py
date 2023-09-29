import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from dolfinx.fem import Function, FunctionSpace, Constant
from ufl import TestFunction, dx, inner


L = 10.0
domain = mesh.create_box(MPI.COMM_WORLD,[[0.0,0.0,0.0],[L,1,1]],[10,5,5], mesh.CellType.hexahedron)
V = fem.VectorFunctionSpace(domain, ("CG", 1))

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

fdim = domain.topology.dim -1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with 2.as_integer_ratio
marked_facets = np.hstack([left_facets,right_facets])


marked_values = np.hstack([np.full_like(left_facets,1), np.full_like(right_facets,2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets],marked_values[sorted_facets])

# boundary condition on the left side: fixed
u_bc = np.array((0,) * domain.geometry.dim, dtype=PETSc.ScalarType)
left_dofs = fem.locate_dofs_topological(V,facet_tag.dim, facet_tag.find(1))
bc_l = fem.dirichletbc(u_bc, left_dofs, V)

# B.C on the right side = a function to interpolate
import numpy as np
import ufl

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx import fem, mesh, plot
from dolfinx.fem import Function, FunctionSpace, Constant
from ufl import TestFunction, dx, inner

from mpi4py import MPI
scale = 0.5
y0 = 0.5
z0 = 0.5

class MyExpression:
    def __init__(self):
        self.theta = np.pi/4

    def eval(self, x):
        return (np.zeros_like(x[0]), # equal to x[0] - x[0],
                y0 + (x[1] - y0) * np.cos(self.theta) - (x[2] - z0) * np.sin(self.theta) - x[1],
                z0 + (x[1] - y0) * np.sin(self.theta) + (x[2] - z0) * np.cos(self.theta) - x[2])

f = MyExpression()    
u_D = Function(V)
u_D.interpolate(f.eval)

right_dofs = fem.locate_dofs_topological(V,facet_tag.dim, facet_tag.find(2))
bc_r = fem.dirichletbc(u_D,right_dofs)
bcs =  [bc_l, bc_r]   

from petsc4py.PETSc import ScalarType

L = 1
W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 500
lambda_ = beta
g = gamma

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType((0, 0, 0)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx #+ ufl.dot(T, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#print(problem)
uh = problem.solve()

# visualization 
import pyvista
# pyvista.set_jupyter_backend("pythreejs")

# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, geometry = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((geometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
   p.show()
else:
   pyvista.start_xvfb()
   figure_as_array = p.screenshot("deflection.png")

uh.vector.destroy()