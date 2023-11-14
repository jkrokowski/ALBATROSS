import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import fem, mesh, io, plot

# Define temporal parameters
t = 0 # Start time
T = 1.0 # Final time
num_steps = 50     
dt = T / num_steps # time step size

# Define mesh
nx, ny = 50, 50
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([-2, -2]), np.array([2, 2])], 
                               [nx, ny], mesh.CellType.triangle)
V = fem.FunctionSpace(domain, ("CG", 1))

# Create initial condition
def initial_condition(x, a=5):
    return np.exp(-a*(x[0]**2+x[1]**2))
u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

# Create boundary condition
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

xdmf = io.XDMFFile(domain.comm, "diffusion.xdmf", "w")
xdmf.write_mesh(domain)

# Define solution variable, and interpolate initial solution for visualization in Paraview
uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)
xdmf.write_function(uh, t)


import ufl
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))

for i in range(num_steps):
    t += dt

    a = u * v * ufl.dx + dt*ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx 
    L = (u_n + dt * f) * v * ufl.dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = fem.petsc.assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = fem.petsc.create_vector(linear_form)
    fem.petsc.set_bc(b, [bc])
    b.assemble()

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Solve linear problem
    solver.solve(b, uh.vector)

    # Update solution at previous time step (u_n)
    u_n.x.array[:] = uh.x.array

    # Write solution to file
    xdmf.write_function(uh, t)
 
xdmf.close()