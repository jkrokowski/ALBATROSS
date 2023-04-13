# Torsion of a closed cross-section
# ================

from mpi4py import MPI
from dolfinx import mesh,plot
from dolfinx.fem import FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import inner,TrialFunction,TestFunction,ds,dx,grad,exp,sin,SpatialCoordinate,FacetNormal
from petsc4py import PETSc
import pyvista
import numpy as np

# Create mesh and define function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20,20, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))

u = TrialFunction(V)
v = TestFunction(V)
#shear centre for symmetric shape
xs=0.5
ys=0.5
x = SpatialCoordinate(domain)
n = FacetNormal(domain)
f = 0.0
g = (x[1]-ys)*n[0] - (x[0]-xs)*n[1] 

a = inner(grad(u), grad(v))*dx
# L = f*v*dx + g*v*ds
L = g*v*ds
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


#======= COMPUTE TORSIONAL CONSTANT =========#
Ixx= assemble_scalar(form(((x[1]-xs)**2)*dx))
Iyy = assemble_scalar(form(((x[0]-ys)**2)*dx))

print(Ixx)
print(Iyy)

#compute derivatives of warping function
W = VectorFunctionSpace(domain, ("CG", 1))
grad_uh = grad(uh)
grad_uh_expr = Expression(grad_uh, W.element.interpolation_points())
grad_u = Function(W)
grad_u.interpolate(grad_uh_expr)

#separate out partial derivatives
dudx_expr = Expression(grad_uh[0], V.element.interpolation_points())
dudx = Function(V)
dudx.interpolate(dudx_expr)

dudy_expr = Expression(grad_uh[1], V.element.interpolation_points())
dudy = Function(V)
dudy.interpolate(dudy_expr)

Kwx = assemble_scalar(form(((x[0]-xs)*dudy)*dx))
Kwy = assemble_scalar(form(((x[1]-ys)*dudx)*dx))

print(Kwx)
print(Kwy)

K = Ixx+Iyy+Kwx-Kwy

print(K)

#======= PLOT WARPING FUNCTION =========#
tdim = domain.topology.dim
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True,opacity=0.25)
u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["u"] = uh.x.array.real
u_grid.set_active_scalars("u")
plotter.add_mesh(u_grid.warp_by_scalar("u",factor=2), show_edges=True)
plotter.view_vector((-0.25,-1,0.5))
if not pyvista.OFF_SCREEN:
    plotter.show()

#======= PLOT GRAD OF WARPING FUNCTION =========#
grad_plotter = pyvista.Plotter()
grad_plotter.add_mesh(grid, style="wireframe", color="k")

values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(grad_u)] = grad_u.x.array.real.reshape((geometry.shape[0], len(grad_u)))

function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["grad_u"] = values
glyphs = function_grid.glyph(orient="grad_u", factor=1)

grad_plotter.add_mesh(glyphs)
grad_plotter.view_xy()
if not pyvista.OFF_SCREEN:
    grad_plotter.show()