from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from ufl import FiniteElement,TrialFunction,TestFunctions

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Build function space with Lagrange multiplier
P1 = FiniteElement("Lagrange", domain.ufl_cell(), 1)
R = FiniteElement("Real", domain.ufl_cell(), 0)
print(R)
RFS = FunctionSpace(domain,R)
# RFS1 = FunctionSpace(domain,("Real",0))
W = FunctionSpace(domain, P1 * R)

# Define variational problem
(u, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = (inner(grad(u), grad(v)) + c*v + u*d)*dx
L = f*v*dx + g*v*ds

# Compute solution
w = Function(W)
solve(a == L, w)
(u, c) = w.split()

# Save solution in VTK format
file = File("neumann_poisson.pvd")
file << u