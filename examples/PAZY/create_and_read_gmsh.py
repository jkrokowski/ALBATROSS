from dolfinx.io import XDMFFile
from dolfinx import plot,mesh
from mpi4py import MPI
import pyvista
import os,sys
import numpy as np

import gmsh
import meshio

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)
xsName = "square_2iso_quads"
meshName_msh = xsName + ".msh"
meshfile = os.path.join(dirpath,meshName_msh)
#==================================
#====== ALLL THE GMSH MESH CREATION
#==================================

gmsh.initialize()

gmsh.model.add("square_2iso")

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
gmsh.write(meshfile)

# Launch the GUI to see the results:
# if '-nopopup' not in sys.argv:
#     gmsh.fltk.run()

gmsh.finalize()



#utility function for saving cell data
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

if MPI.COMM_WORLD.rank == 0:
    # Read in mesh
    msh = meshio.read(meshfile)

    # Create and save one file for the mesh, and one file for the facets 
    msh_out = create_mesh(msh, "quad", prune_z=True)
    print(msh_out.cells)
    print(msh_out.cell_data)
    meshName_xdmf = xsName + ".msh"
    meshfile_xdmf = os.path.join(dirpath,meshName_xdmf)

    meshio.write(meshfile_xdmf,msh_out)

# xsName = "beam_crosssection_rib_221_quad"
xsName = "square_2iso_quads"

fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
print(filePath)

with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain,name="Grid")

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#plot mesh:
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xz()
p.show()