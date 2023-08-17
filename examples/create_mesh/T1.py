'''
T1 - Three Cell Hollow Box Validation Example Mesh from BECAS manual
'''

import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI

#mesh parameters
gdim=2
tdim=2

#cross section properties
xcName = "three_cell_hollow_box"
t = 0.01
width = 1.0
height = 0.25
width1 = .2375
width2 = 0.485
blcx = -0.5
blcy = -0.125
tlcy = 0.125

gmsh.initialize()
gmsh.model.add(xcName)
gmsh.model.setCurrent(xcName)

#corner locations,number of elements on each edge,
#for example: [-0.5,-0.5,-0.49,0.49],[2,2]
#   or:         [-0.49,-.5,-(0.49-0.2375),-.49],[10,2]]
box_nodes = [[blcx,blcy,blcx+t,blcy+t],
             [blcx+t,blcy,blcx+t+width1,blcy+t],
             [blcx+t+width1,blcy,blcx+2*t+width1,blcy+t],
             [blcx+2*t+width1,blcy,blcx+2*t+2*width1,blcy+t],
             [blcx+2*t+2*width1,blcy,blcx+3*t+2*width1,blcy+t],
             [blcx+3*t+2*width1,blcy,width/2-t,blcy+t],
             [width/2-t,blcy,width/2,blcy+t],
             [blcx,tlcy,blcx+t,tlcy-t],
             [blcx+t,tlcy,blcx+t+width1,tlcy-t],
             [blcx+t+width1,tlcy,blcx+2*t+width1,tlcy-t],
             [blcx+2*t+width1,tlcy,blcx+2*t+2*width1,tlcy-t],
             [blcx+2*t+2*width1,tlcy,blcx+3*t+2*width1,tlcy-t],
             [blcx+3*t+2*width1,tlcy,width/2-t,tlcy-t],
             [width/2-t,tlcy,width/2,tlcy-t],
             [blcx,blcy+t,blcx+t,tlcy-t],
             [blcx+t+width1,blcy,blcx+2*t+width1,tlcy-t],
             [blcx+2*t+2*width1,blcy,blcx+3*t+2*width1,tlcy-t],
             [width/2-t,blcy,width/2,tlcy-t]]
box_el_num = [[2,2],
              [10,2],
              [2,2],
              [10,2],
              [2,2],
              [10,2],
              [2,2],
              [2,2],
              [10,2],
              [2,2],
              [10,2],
              [2,2],
              [10,2],
              [2,2],
              [2,10],
              [2,10],
              [2,10],
              [2,10]]
print(len(box_nodes))
tags = list(range(len(box_nodes)))
for box,el_num in zip(box_nodes,box_el_num):
    print(box)
    x1,y1,x2,y2 = box
    print(box)
    num_el_x, num_el_y = el_num

    p1 = gmsh.model.geo.addPoint(x1,y1,0)
    p2 = gmsh.model.geo.addPoint(x1,y2,0)
    p3 = gmsh.model.geo.addPoint(x2,y2,0)
    p4 = gmsh.model.geo.addPoint(x2,y1,0)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    gmsh.model.geo.mesh.setTransfiniteCurve(l1, int(num_el_y + 1))
    gmsh.model.geo.mesh.setTransfiniteCurve(l2, int(num_el_x + 1))
    gmsh.model.geo.mesh.setTransfiniteCurve(l3, int(num_el_y + 1))
    gmsh.model.geo.mesh.setTransfiniteCurve(l4, int(num_el_x + 1))

    cl1 = gmsh.model.geo.addCurveLoop([p1, p2, p3, p4])
    rect =gmsh.model.geo.addPlaneSurface([cl1])

    gmsh.model.geo.mesh.setTransfiniteSurface(rect)
    gmsh.model.geo.mesh.setRecombine(2, rect)

gmsh.model.add_physical_group(tdim,tags,1,"rect")
gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(2)
gmsh.write(xcName+".msh")
gmsh.fltk.run()

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
# cell_markers.name = f"{msh.name}_cells"
# facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"output/{xcName}.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()