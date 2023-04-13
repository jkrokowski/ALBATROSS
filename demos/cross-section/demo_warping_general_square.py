# Torsion of a closed cross-section
# ================

from mpi4py import MPI
from dolfinx import mesh,plot
from dolfinx.fem import FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import inner,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal
from petsc4py import PETSc
import pyvista
import numpy as np

#================Compute auxliary warping function============#
#      (does not require prior knowledge of shear center)

# Create mesh and define function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 20,20, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))

u = TrialFunction(V)
v = TestFunction(V)

#integration measures
dx = Measure("dx",domain=domain)
ds = Measure("ds",domain=domain)
#spatial coordinate and facet normals
x = SpatialCoordinate(domain)
n = FacetNormal(domain)
#construct LHS form
f = 0.0
g = (x[1])*n[0] - (x[0])*n[1] 
a = inner(grad(u), grad(v))*dx
#construct RHS form
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

# Finally we are able to solve our linear system :
solver.solve(b,uh.vector)


#======= COMPUTE TORSIONAL CONSTANT =========#
# This requires computing the shear center and computing the warping 
# function based on the location of that shear center

#area
A = assemble_scalar(form(1.0*dx))
print(A)
#first moment of area / A (gives the centroid)
x_G = assemble_scalar(form(x[0]*dx)) / A
y_G = assemble_scalar(form(x[1]*dx)) / A
print(x_G)
print(y_G)
#second moment of area 
Ixx= assemble_scalar(form(((x[1]-y_G)**2)*dx))
Iyy = assemble_scalar(form(((x[0]-x_G)**2)*dx))
print(Ixx)
print(Iyy)

Ixy = assemble_scalar(form(((x[0]-x_G)*(x[1]-y_G))*dx))
Iwx = assemble_scalar(form(((x[1]-y_G)*uh)*dx))
Iwy = assemble_scalar(form(((x[0]-x_G)*uh)*dx))
print("product and warping")
print(Ixy)
print(Iwx)
print(Iwy)

xs = (Iyy*Iwy-Ixy*Iwx)/(Ixx*Iyy-Ixy**2)
ys = -(Ixx*Iwx-Ixy*Iwy)/(Ixx*Iyy-Ixy**2)
print("shear center:")
print(xs)
print(ys)

w1_expr = Expression(-ys*x[0]+xs*x[1],V.element.interpolation_points())
w1 = Function(V)
w1.interpolate(w1_expr)

w = Function(V)
w.x.array[:] = uh.x.array[:] + w1.x.array[:]

#compute derivatives of warping function
W = VectorFunctionSpace(domain, ("CG", 1))
grad_uh = grad(w)
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

Kwx = assemble_scalar(form(((x[0]-x_G)*dudy)*dx))
Kwy = assemble_scalar(form(((x[1]-y_G)*dudx)*dx))

print(Kwx)
print(Kwy)

K = Ixx+Iyy+Kwx-Kwy
print("Torsional Constant:")
print(K)

#======= PLOT WARPING FUNCTION =========#
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = uh.x.array.real
# u_grid.set_active_scalars("u")
# plotter.add_mesh(u_grid.warp_by_scalar("u",factor=2), show_edges=True)
# plotter.view_vector((-0.25,-1,0.5))
# if not pyvista.OFF_SCREEN:
#     plotter.show()

# #plot warping
tdim = domain.topology.dim
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True,opacity=0.25)
u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
u_grid.point_data["w"] = w.x.array.real
u_grid.set_active_scalars("w")
plotter.add_mesh(u_grid.warp_by_scalar("w",factor=2), show_edges=True)
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