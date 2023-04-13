# Singular Poisson
# ================

from mpi4py import MPI
from dolfinx import mesh,plot
from dolfinx.fem import FunctionSpace,Function,form
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import inner,TrialFunction,TestFunction,ds,dx,grad,exp,sin,SpatialCoordinate
from petsc4py import PETSc
import pyvista

# Create mesh and define function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20,20, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))

u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
f = 10*exp(-(((x[0] - 0.5)**2) + (x[1] - 0.5)**2) / 0.02)
g = -sin(5*x[0])

a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Assemble system
A = assemble_matrix(form(a))
A.assemble()
b=create_vector(form(L))
with b.localForm() as b_loc:
            b_loc.set(0)
assemble_vector(b,form(L))

# Solution Function
uh = Function(V)

# Create Krylov solver
solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)

# Create vector that spans the null space
nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
A.setNullSpace(nullspace)

# orthogonalize b with respect to the nullspace ensures that 
# b does not contain any component in the nullspace
nullspace.remove(b)

# Finally we are able to solve our linear system ::
solver.solve(b,uh.vector)

#plotting stuff
tdim = domain.topology.dim
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True,opacity=0.25)
u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
plotter.add_mesh(u_grid.warp_by_scalar("u",factor=.5), show_edges=True)
plotter.view_vector((-0.25,-1,0.5))
if not pyvista.OFF_SCREEN:
    plotter.show()