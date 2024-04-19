import gmsh
import math
import os
import sys

gmsh.initialize()

gmsh.model.add("ribXS")

# Load a STEP file (using `importShapes' instead of `merge' allows to directly
# retrieve the tags of the highest dimensional imported entities):
path = os.path.dirname(os.path.abspath(__file__))
v = gmsh.model.occ.importShapes(os.path.join(path, 'Rib_cross_section.step'))

gmsh.fltk.run()