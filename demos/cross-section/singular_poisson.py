# Singular Poisson
# ================

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

# Then, we check that dolfin is configured with the backend called
# PETSc, since it provides us with a wide range of methods used by
# :py:class:`KrylovSolver <dolfin.cpp.la.KrylovSolver>`. We set PETSc as
# our backend for linear algebra::

# Test for PETSc
if not has_linear_algebra_backend("PETSc"):
    info("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

parameters["linear_algebra_backend"] = "PETSc"

# Create mesh and define function space
mesh = UnitSquareMesh(20, 20)
V = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree=2)
g = Expression("-sin(5*x[0])", degree=2)

a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Assemble system
A = assemble(a)
b = assemble(L)


# Solution Function
u = Function(V)

# Next, we specify the iterative solver we want to use, in this case a
# :py:class:`PETScKrylovSolver <dolfin.cpp.la.PETScKrylovSolver>` with
# the conjugate gradient (CG) method, and attach the matrix operator to
# the solver. ::

# Create Krylov solver
solver = PETScKrylovSolver("cg")
solver.set_operator(A)

# Create vector that spans the null space and normalize
null_vec = Vector(u.vector())
print(null_vec.get_local())
V.dofmap().set(null_vec, 1.0)
print(null_vec.get_local())
null_vec *= 1.0/null_vec.norm("l2")
print(null_vec.get_local())

# Create null space basis object and attach to PETSc matrix
null_space = VectorSpaceBasis([null_vec])
as_backend_type(A).set_nullspace(null_space)

# Orthogonalization of ``b`` with respect to the null space makes sure
# that it doesn't contain any component in the null space. ::

null_space.orthogonalize(b)

# Finally we are able to solve our linear system ::

solver.solve(u.vector(), b)

print(np.max(u.vector().get_local()))
print(np.min(u.vector().get_local()))

# and plot the solution ::

fig = plt.figure()
plotted = plot(u)
fig.colorbar(plotted)
plt.show()

