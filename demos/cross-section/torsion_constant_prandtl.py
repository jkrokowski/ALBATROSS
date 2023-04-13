from dolfinx import mesh,nls
from dolfinx.fem import (FunctionSpace,Constant,Function,form,dirichletbc,locate_dofs_geometrical
                         ,assemble_scalar,VectorFunctionSpace,Expression)
from dolfinx.fem.petsc import LinearProblem,create_vector,assemble_matrix,NonlinearProblem
from dolfinx.plot import create_vtk_mesh
from mpi4py import MPI
from ufl import Measure,TrialFunction,TestFunction,inner,grad,nabla_grad,dx,ds,SpatialCoordinate,as_matrix,as_vector,dot
from petsc4py.PETSc import ScalarType,KSP,NullSpace
import numpy as np
from scipy.linalg import null_space
import pyvista
from dolfinx import plot
'''
Computing the torsional constant using a Prandtl Stress function of a square

'''

# Create mesh
# rectangle2to1 = mesh.create
square = mesh.create_unit_square(MPI.COMM_WORLD, 20,20, mesh.CellType.quadrilateral)
domain = square
V = FunctionSpace(domain, ("CG", 1))

phi = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
#G and Theta are virutal loads that can be anything within numerical reason
# their purpose is to perturb the solution away from a "boring" LaPlace equation
# with the boundary equal to 0 (which would just be 0 everywhere)
G = 10
Theta = 0.1
f = Constant(domain, ScalarType(2*G*Theta))
a = (inner(grad(phi), grad(v)) )*dx
L = f*v*dx 

def left(x):
    return np.isclose(x[0], 0)
def right(x):
    return np.isclose(x[0],1)
def top(x):
    return np.isclose(x[1],0)
def bot(x):
    return np.isclose(x[1],1)

# fdim = domain.topology.dim - 1
# boundary_facets = mesh.locate_entities_boundary(domain, fdim, left)

# u_D = np.array([0], dtype=ScalarType)
bc1 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, left),V)
bc2 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, right),V)
bc3 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, top),V)
bc4 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, bot),V)

# Compute solution
phih = Function(V)
problem = LinearProblem(a,L,bcs=[bc1,bc2,bc3,bc4])
phih = problem.solve()

It = (2/(G*Theta))*assemble_scalar(form(phih*dx))
print("torsional constant:")
print(It)

#TODO:compute warping function from the prandtl stress function:


W = VectorFunctionSpace(domain, ("CG", 1))

grad_phih = grad(phih)
print(grad_phih.ufl_shape)
grad_phi_expr = Expression(grad_phih, W.element.interpolation_points())
grad_phi = Function(W)
grad_phi.interpolate(grad_phi_expr)

print(phih.x.array.shape)
print(grad_phi.x.array.shape)
print(grad_phi.ufl_shape)

#plot grad phi (stress function)
topology, cell_types, geometry = create_vtk_mesh(W)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(grad_phi)] = grad_phi.x.array.real.reshape((geometry.shape[0], len(grad_phi)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["grad_phi"] = values
glyphs = function_grid.glyph(orient="grad_phi", factor=.25)

# Create a pyvista-grid for the mesh
grid = pyvista.UnstructuredGrid(*create_vtk_mesh(domain, domain.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")



#compute derivative of warping from stress function    
xs=0.5
ys=0.5
grad_psi_ex = dot((1/(G))*as_matrix([[0,1],[-1,0]]),grad_phih) + as_vector([Theta*(x[1]-ys),-Theta*(x[0]-xs)])
grad_psi_expr = Expression(grad_psi_ex, W.element.interpolation_points())
grad_psi = Function(W)
grad_psi.interpolate(grad_psi_expr)

#plot grad psi (derivative of warping)
topology, cell_types, geometry = create_vtk_mesh(W)
values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
values[:, :len(grad_phi)] = grad_psi.x.array.real.reshape((geometry.shape[0], len(grad_psi)))

# Create a point cloud of glyphs
function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
function_grid["grad_psi"] = values
glyphs = function_grid.glyph(orient="grad_psi", factor=5)

# Create a pyvista-grid for the mesh
grid = pyvista.UnstructuredGrid(*create_vtk_mesh(domain, domain.topology.dim))

# Create plotter
plotter = pyvista.Plotter()
plotter.add_mesh(grid, style="wireframe", color="k")
plotter.add_mesh(glyphs)
plotter.view_xy()
if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    fig_as_array = plotter.screenshot("glyphs.png")

# psi = Function(V)
# dpsi = TestFunction(V)
# print('---------')
# print(grad(psi).ufl_shape)
# print(grad_psi.ufl_shape)
# print(grad(dpsi).ufl_shape)
# print((grad(psi)-grad_psi).ufl_shape)
# print(inner(grad(psi)-grad_psi,grad(dpsi)).ufl_shape)
# # A = inner(grad(psi)-grad_psi,grad(dpsi))*dx
# # f = Constant(domain, ScalarType(0.0))
# # l = f*dpsi*dx 
# F = inner(grad(psi)-grad_psi,grad(dpsi))*dx +f*dpsi*dx

# def center(x):
#     return np.isclose(x[0], 0)

# bc = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, center),V)

# # Compute solution
# psih = Function(V)

# # problem = LinearProblem(A,l,bcs=[bc])
# # psih = problem.solve()

# problem = NonlinearProblem(F,psih,bcs=[bc])
# solver = nls.petsc.NewtonSolver(domain.comm, problem)

# # Set Newton solver options
# # solver.atol = 1e-3
# # solver.rtol = 1e-3
# solver.max_it = 100
# solver.report = True
# # solver.convergence_criterion = "incremental"

# num_its, converged = solver.solve(psih)

# print(psih.x.array)








#=============================
#compute torsional constant:
#=============================
# dx = Measure("dx",domain=domain)
# A = assemble_scalar(form(1.0*dx))
# x_G = assemble_scalar(form(x[1]*dx))/A
# y_G = assemble_scalar(form(x[0]*dx))/A

# print(x_G)
# print(y_G)

# Ix= assemble_scalar(form(((x[1]-y_G)**2)*dx))
# Iy = assemble_scalar(form(((x[0]-x_G)**2)*dx))

# print(Ix)
# print(Iy)e
# print(Kwx)
# print(Kwy)

#=============================
#plotting stuff
#=============================
import pyvista
from dolfinx import plot
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     figure = plotter.screenshot("fundamentals_mesh.png")

# #2d plot
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["phi"] = phih.x.array
# u_grid.set_active_scalars("phi")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True)
# u_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     u_plotter.show()


# #grad phi plot
# # Create plotter and pyvista grid
# p = pyvista.Plotter()
# topology, cell_types, geometry = plot.create_vtk_mesh(V)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# # Attach vector values to grid and warp grid by vector
# print(geometry.shape)

# grid["grad_phi"] = np.concatenate((grad_phi.x.array.reshape((geometry.shape[0], 2)),np.zeros((geometry.shape[0],1))),axis=1)
# actor_0 = p.add_mesh(grid, style="wireframe", color="k")
# warped = grid.warp_by_vector("grad_phi", factor=1)
# actor_1 = p.add_mesh(warped, show_edges=True)
# p.show_axes()
# if not pyvista.OFF_SCREEN:
#    p.show()
# else:
#    figure_as_array = p.screenshot("deflection.png")



# # 3d Plot of warping function
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# grid.point_data["phi"] = phih.x.array
# warped = grid.warp_by_scalar("phi",factor=10)
# # u_grid.set_active_scalars("phi")
# warped_plotter = pyvista.Plotter()
# warped_plotter.add_mesh(warped, show_edges=True,show_scalar_bar=True,scalars='phi')
# warped_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     warped_plotter.show()


