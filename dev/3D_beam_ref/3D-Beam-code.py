#Reference 3D Solution for comparison with the beam analysis approach

# Scaled variable
Lx = 20
W = 1
H = 1

E = 10e6
nu = 0.2
rho = 2.7e-3
mu = E/(2*(1+nu))
lambda_ = (E*nu)/((1+nu)*(1-2*nu))
g = 9.81
N=20
Nx = N*Lx
print("number of dofs: %i" % (N*N*Nx))

import numpy as np
import ufl

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

from dolfinx import mesh, fem, plot, io

import time
t0 = time.time()
domain = mesh.create_box(MPI.COMM_WORLD, [np.array([0,-W/2,-H/2]), np.array([Lx, W/2, H/2])],
                  [Nx,N,N], cell_type=mesh.CellType.hexahedron)
t1 = time.time()
V = fem.VectorFunctionSpace(domain, ("CG", 1))

def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0,0,0], dtype=ScalarType)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

T = fem.Constant(domain, ScalarType((0, 0, 0)))

ds = ufl.Measure("ds", domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u)) # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)
def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2*mu*epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType((0, 0, -rho*g)))
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

t2 = time.time()
print("Total time:")
print(t2-t0)
print("Mesh construction:")
print(t1-t0)
print("3D analysis:")
print(t2-t1)
import pyvista
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
# Create plotter and pyvista grid
p = pyvista.Plotter()
topology, cell_types, ggeometry = plot.create_vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, ggeometry)

# Attach vector values to grid and warp grid by vector
grid["u"] = uh.x.array.reshape((ggeometry.shape[0], 3))
actor_0 = p.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1)
actor_1 = p.add_mesh(warped, show_edges=True)
p.show_axes()
if not pyvista.OFF_SCREEN:
   p.show()
else:
   figure_as_array = p.screenshot("deflection.png")

print("displacement at neutral axis:")
tol = 0.001 # Avoid hitting the outside of the domain
points = np.array([[Lx-tol,0+tol,0+tol]]).T
u_values = []
tol = 0.001 # Avoid hitting the outside of the domain
# y = np.linspace(-1 + tol, 1 - tol, 101)
# points = np.zeros((3, 101))
# points[1] = y
u_values = []
p_values = []
from dolfinx import geometry
bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
cells = []
points_on_proc = []
# Find cells whose bounding-box collide with the the points
cell_candidates = geometry.compute_collisions(bb_tree, points.T)
# Choose one of the cells that contains the point
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
for i, point in enumerate(points.T):
    if len(colliding_cells.links(i))>0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])
points_on_proc = np.array(points_on_proc, dtype=np.float64)
u_values = uh.eval(points_on_proc, cells)
print(u_values)
if True:
    s = sigma(uh) -1./3*ufl.tr(sigma(uh))*ufl.Identity(len(uh))
    von_Mises = ufl.sqrt(3./2*ufl.inner(s, s))

    V_von_mises = fem.FunctionSpace(domain, ("DG", 0))
    stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
    stresses = fem.Function(V_von_mises)
    stresses.interpolate(stress_expr)

    warped.cell_data["VonMises"] = stresses.vector.array
    warped.set_active_scalars("VonMises")
    p = pyvista.Plotter()
    p.add_mesh(warped)
    p.show_axes()
    if not pyvista.OFF_SCREEN:
        p.show()
    else:
        stress_figure = p.screenshot(f"stresses.png")