from dolfin import *

# Create mesh
mesh = UnitSquareMesh.create(16,16, CellType.Type.quadrilateral)

P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, P1)

u = TrialFunction(W)
v = TestFunction(W)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)
a = (inner(grad(u), grad(v)) )*dx
L = f*v*dx + g*v*ds

# Compute solution
uh = Function(W)
solve(a == L, uh)

# Save solution in VTK format
file = File("neumann_poisson_test.pvd")
file << uh