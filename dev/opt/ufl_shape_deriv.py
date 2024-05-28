from dolfinx import mesh,fem,cpp,plot
from mpi4py import MPI
from ufl import SpatialCoordinate,dx,Argument,derivative
import numpy as np
import pyvista
W = 1 
H = 2
N = 4
num_el = [N,N]
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
mesh_2D = mesh.create_rectangle(MPI.COMM_WORLD,points,num_el, cell_type=mesh.CellType.quadrilateral)

X = SpatialCoordinate(mesh_2D)

VX = fem.VectorFunctionSpace(mesh_2D,("CG",1))

VT = fem.FunctionSpace(mesh_2D,("CG",1))
T = fem.Function(VT)
T.x.array[:] = 1.
area = T*dx
# UFL arguments need unique indices within a form
args = area.arguments()
n = max(a.number() for a in args) if args else -1
du = Argument(VX, n+1)
dAdX = derivative(area, X, du)
print("Derivatives of area w.r.t. spatial coordinates")
dAdX_vec = fem.petsc.assemble_vector(fem.form(dAdX)).array
print(dAdX_vec.reshape(-1,2))
fdim = 0
num_facets_owned_by_proc = mesh_2D.topology.index_map(fdim).size_local
geometry_entities = cpp.mesh.entities_to_geometry(mesh_2D, fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)
points = mesh_2D.geometry.x
print('Node id, Coords')
for e, entity in enumerate(geometry_entities):
    print(e, points[entity])

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
plotter = pyvista.Plotter()
#plot mesh
tdim = mesh_2D.topology.dim
topology, cell_types, geom = plot.create_vtk_mesh(mesh_2D, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geom)

sensitivity = np.concatenate([dAdX_vec.reshape(-1,2),np.zeros((dAdX_vec.reshape(-1,2).shape[0],1))],axis=1)

grid.point_data["sensitivity"] = sensitivity
warped = grid.warp_by_vector("sensitivity")
plotter.add_mesh(warped,show_edges=True,opacity=0.5)
# plotter.add_mesh(grid,show_edges=True,opacity=1)
plotter.view_isometric()
plotter.show_bounds()
plotter.add_axes()
if not pyvista.OFF_SCREEN:
    plotter.show()