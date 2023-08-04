import gmsh
# import math
import sys
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI
import meshio

gmsh.initialize()

gmsh.model.add("square_2iso")

xcName = "square_2iso"
#mesh parameters
gdim=2
tdim=2

# Copied from `t1.py'...
lc = 1e-2

# When the surface has only 3 or 4 points on its boundary the list of corners
# can be omitted in the `setTransfiniteSurface()' call:
gmsh.model.geo.addPoint(0, 0, 0, 1.0, 1)
gmsh.model.geo.addPoint(0.05, 0, 0, 1.0, 2)
gmsh.model.geo.addPoint(0.05, 0.1, 0, 1.0, 3)
gmsh.model.geo.addPoint(0, 0.1, 0, 1.0, 4)
gmsh.model.geo.addPoint(-0.05, 0, 0, 1.0, 5)
gmsh.model.geo.addPoint(-0.05, 0.1, 0, 1.0, 6)
gmsh.model.geo.addLine(1, 2, 1)
gmsh.model.geo.addLine(2, 3, 2)
gmsh.model.geo.addLine(3, 4, 3)
gmsh.model.geo.addLine(4, 1, 4)
gmsh.model.geo.addLine(1,5,5)
gmsh.model.geo.addLine(5,6,6)
gmsh.model.geo.addLine(6,4,7)

cl1 = gmsh.model.geo.addCurveLoop([1, 2, 3, 4], -1)
cl2 = gmsh.model.geo.addCurveLoop([5,6,7,4], -1)
right_plane =gmsh.model.geo.addPlaneSurface([cl1], -1)
left_plane =gmsh.model.geo.addPlaneSurface([cl2], -1)

num_el = 10
#center line
gmsh.model.geo.mesh.setTransfiniteCurve(4, int(num_el + 1))
#right surface lines meshing constraints
gmsh.model.geo.mesh.setTransfiniteCurve(1, int(num_el/2 +1))
gmsh.model.geo.mesh.setTransfiniteCurve(2, int(num_el + 1))
gmsh.model.geo.mesh.setTransfiniteCurve(3, int(num_el/2 +1))   
#left surface lines meshing constraints
gmsh.model.geo.mesh.setTransfiniteCurve(5, int(num_el/2 +1))
gmsh.model.geo.mesh.setTransfiniteCurve(6, int(num_el + 1))
gmsh.model.geo.mesh.setTransfiniteCurve(7, int(num_el/2 +1))   
#surface meshing constraints
gmsh.model.geo.mesh.setTransfiniteSurface(right_plane)
gmsh.model.geo.mesh.setTransfiniteSurface(left_plane)

gmsh.model.geo.mesh.setRecombine(2, right_plane)
gmsh.model.add_physical_group(tdim,[right_plane],0,"right")

gmsh.model.geo.mesh.setRecombine(2, left_plane)
gmsh.model.add_physical_group(tdim,[left_plane],1,"left")

gmsh.model.geo.synchronize()

# # Finally we apply an elliptic smoother to the grid to have a more regular
# # mesh:
# gmsh.option.setNumber("Mesh.Smoothing", 100)

gmsh.model.mesh.generate(2)
gmsh.write(xcName+".msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

gmsh.finalize()


def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

if MPI.COMM_WORLD.rank == 0:
    # Read in mesh
    msh = meshio.read(xcName + ".msh")

    # Create and save one file for the mesh, and one file for the facets 
    mesh = create_mesh(msh, "quad", prune_z=True)
    print(mesh.cells)
    print(mesh.cell_data)
    meshio.write(f"output/"+xcName+".xdmf", mesh)
