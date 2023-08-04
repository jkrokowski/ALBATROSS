# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 6
#
#  Transfinite meshes, deleting entities
#
# ------------------------------------------------------------------------------

import gmsh
# import math
import sys
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI

gmsh.initialize()

gmsh.model.add("t6")

xcName = "tfs_test"
#mesh parameters
gdim=2
tdim=2

# Copied from `t1.py'...
lc = 1e-2

# When the surface has only 3 or 4 points on its boundary the list of corners
# can be omitted in the `setTransfiniteSurface()' call:
gmsh.model.geo.addPoint(0, 0, 0, 1.0, 7)
gmsh.model.geo.addPoint(0.05, 0, 0, 1.0, 8)
gmsh.model.geo.addPoint(0.05, 0.1, 0, 1.0, 9)
gmsh.model.geo.addPoint(0, 0.1, 0, 1.0, 10)
gmsh.model.geo.addLine(7, 8, 11)
gmsh.model.geo.addLine(8, 9, 12)
gmsh.model.geo.addLine(9, 10, 13)
gmsh.model.geo.addLine(10, 7, 14)
gmsh.model.geo.addCurveLoop([11, 12, 13, 14], 15)
gmsh.model.geo.addPlaneSurface([15], 16)
num_el = 10
gmsh.model.geo.mesh.setTransfiniteCurve(11, int(num_el/2 +1))
gmsh.model.geo.mesh.setTransfiniteCurve(12, int(num_el + 1))
gmsh.model.geo.mesh.setTransfiniteCurve(13, int(num_el/2 +1))   
gmsh.model.geo.mesh.setTransfiniteCurve(14, int(num_el + 1))
gmsh.model.geo.mesh.setTransfiniteSurface(16)

# The way triangles are generated can be controlled by specifying "Left",
# "Right" or "Alternate" in `setTransfiniteSurface()' command. Try e.g.
#
# gmsh.model.geo.mesh.setTransfiniteSurface(15, "Alternate")

gmsh.model.geo.mesh.setRecombine(2, 16)
gmsh.model.add_physical_group(tdim,[16],0,"right")

gmsh.model.geo.synchronize()

# # Finally we apply an elliptic smoother to the grid to have a more regular
# # mesh:
# gmsh.option.setNumber("Mesh.Smoothing", 100)

gmsh.model.mesh.generate(2)
gmsh.write("t6.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()


#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

gmsh.finalize()