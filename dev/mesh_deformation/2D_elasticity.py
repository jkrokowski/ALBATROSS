from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_topological,dirichletbc,VectorFunctionSpace,Function,TensorFunctionSpace
from ufl import tr,sym,grad,Constant,Identity,TrialFunction,TestFunction,inner,dx
from dolfinx import geometry,mesh,plot
from mpi4py import MPI
import numpy as np
import pyvista
from petsc4py.PETSc import ScalarType

L = 25
H = 1.
Nx = 250
Ny = 10
delta = 0.05
domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[L, H]]),[Nx,Ny], cell_type=mesh.CellType.quadrilateral)
# domain2 = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[L-H-delta,0+delta],[L-delta, L+delta]]),[Ny,Nx], cell_type=mesh.CellType.quadrilateral)
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
if True:
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)

    # topology, cell_types, geometry = plot.create_vtk_mesh(domain2, tdim)
    # grid2 = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # plotter.add_mesh(grid2, show_edges=True,opacity=0.25)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()

def eps(v):
    return sym(grad(v))

# E = Constant(domain,1e5)
# nu = Constant(domain,0.3)
E = 1e5
nu = 0.3
model = "plane_stress"

mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
if model == "plane_stress":
    lmbda = 2*mu*lmbda/(lmbda+2*mu)

def sigma(v):
    return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)

rho_g = 1e-3
f = Constant(domain,ScalarType((0, -rho_g)))

V = VectorFunctionSpace(domain, ("CG", 1))
du = TrialFunction(V)
u_ = TestFunction(V)
a = inner(sigma(du), eps(u_))*dx
# l = inner(f, u_)*dx
l = -rho_g*u_[1]*dx

def clamped_boundary(x):
    return np.isclose(x[0], 0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0,0], dtype=ScalarType)
bc = dirichletbc(u_D, locate_dofs_topological(V, fdim, boundary_facets), V)

# u = Function(V, name="Displacement")
problem = LinearProblem(a, l, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

# print("Maximal deflection:", -uh(L,H/2.)[1])
# print("Beam theory deflection:", float(3*rho_g*L**4/2/E/H**3))

if True:
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style='wireframe',color='k')

    values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
    values[:, :len(uh)] = uh.x.array.real.reshape((geometry.shape[0], len(uh)))
    grid['u'] = values
    # grid["u"] = uh.x.array.reshape((geometry.shape[0], 2))
    
    # grid.set_active_vectors('u')
    # geom = pyvista.Arrow()
    # glyphs = grid.glyph(orient="vectors", scale="u", factor=0.003, geom=geom)
    
    warped = grid.warp_by_vector("u", factor=1000)
    plotter.add_mesh(warped, show_edges=True)
    # plotter.add_mesh(glyphs, show_edges=True)

    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()


# Vsig = TensorFunctionSpace(mesh, "DG", degree=0)
# sig = Function(Vsig, name="Stress")
# sig.assign(project(sigma(u), Vsig))
# print("Stress at (0,H):", sig(0, H))

# # Fields can be exported in a suitable format for vizualisation using Paraview.
# # VTK-based extensions (.pvd,.vtu) are not suited for multiple fields and parallel
# # writing/reading. Prefered output format is now .xdmf::

# file_results = XDMFFile("elasticity_results.xdmf")
# file_results.parameters["flush_output"] = True
# file_results.parameters["functions_share_mesh"] = True
# file_results.write(u, 0.)
# file_results.write(sig, 0.)
