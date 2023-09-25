import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI
import pyvista
from dolfinx import plot

folder = 'output/'
gmsh.initialize()

# model and mesh parameters
gdim = 3
tdim = 1
R = 10.0
num_el = 30
#construct line in 3D space
gmsh.model.add("Beam")
gmsh.model.setCurrent("Beam")
p1 = gmsh.model.geo.addPoint(0,0,0)
p2 = gmsh.model.geo.addPoint(R, 0, 0)
line1 = gmsh.model.geo.addLine(p1,p2)
gmsh.model.geo.mesh.setTransfiniteCurve(line1, int(num_el + 1))

# Synchronize OpenCascade representation with gmsh model
gmsh.model.geo.synchronize()

# add physical marker
gmsh.model.add_physical_group(tdim,[line1])

# #adjust mesh size parameters
# gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01*R)
# gmsh.option.setNumber('Mesh.MeshSizeMax', 0.1*R)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)

gmsh.write(folder+"beam_mesh.msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, folder+f"beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

#read in xdmf mesh from generation process
fileName = folder+"beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain = xdmf.read_mesh(name="beam_mesh")

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,style='points')
    plotter.view_isometric()
    if not pyvista.OFF_SCREEN:
        plotter.show()