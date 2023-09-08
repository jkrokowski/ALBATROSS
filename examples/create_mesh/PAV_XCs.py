# ------------------------------------------------------------------------------
#
#  Gmsh Python tutorial 20
#
#  STEP import and manipulation, geometry partitioning
#
# ------------------------------------------------------------------------------

# The OpenCASCADE CAD kernel allows to import STEP files and to modify them. In
# this tutorial we will load a STEP geometry and partition it into slices.

import gmsh
import math
import os
import sys
import time
import numpy as np

gmsh.initialize()

gmsh.model.add("t20")

# Load a STEP file (using `importShapes' instead of `merge' allows to directly
# retrieve the tags of the highest dimensional imported entities):
path = os.path.dirname(os.path.abspath(__file__))
# gmsh.option.setString('Geometry.OCCTargetUnit', 'MM')
v = gmsh.model.occ.importShapes(os.path.join(path, os.pardir, '../../Aurora_PAV_shell_surfaces_08042023.igs'))
gmsh.model.occ.synchronize()
# gmsh.fltk.run()

#gather tags
all_tags = gmsh.model.getEntities(-1)
surfs_tags_vec = gmsh.model.getEntities(2)
surfs_tags = [x[1] for x in surfs_tags_vec]

#add physical group of all surfaces
surfs = gmsh.model.addPhysicalGroup(2,surfs_tags)

#scale model to meters
gmsh.model.occ.dilate(all_tags,0,0,0,0.001,0.001,0.001)
gmsh.model.occ.synchronize()

#find bounding box
xmin,ymin,zmin,xmax,ymax,zmax = gmsh.model.get_bounding_box(-1,-1)
dx = xmax-xmin
dy = ymax-ymin
dz = zmax-zmin
# print(dx,dy,dz)

plane = gmsh.model.occ.addRectangle(xmin-dx,ymin,zmin-dz,3*dx,3*dz)
gmsh.model.occ.rotate([(2,plane)],xmin,ymin,zmin,1,0,0,math.pi/2)
gmsh.model.occ.translate([(2,plane)],0,1,-.25)
gmsh.model.occ.synchronize()

def getRelevantTags(cutplane_tag,model_tags):
    #fxn finds surfaces that have bounding boxes that contain cutplane y-coord
    pts = np.zeros((len(model_tags),6))
    for i,tag in enumerate(model_tags):
        pts[i,:] = gmsh.model.occ.getBoundingBox(2,tag)
        print(pts[i,:])
    _,planemin,_,_,planemax,_ = gmsh.model.occ.get_bounding_box(2,cutplane_tag)
    print(planemin,planemax)
    print(model_tags)
    for i in range(pts.shape[0]):
        print(i+1,pts[i,[1,4]])
        print(pts[i,1] <= planemin,pts[i,4]>=planemax)
    return [i+1 for i,pt in enumerate(pts[:,[1,4]]) if (pt[0]<=planemin and pt[1]>= planemax) ]
    
relevant_tags = getRelevantTags(plane,surfs_tags)
print(relevant_tags)
object_tags = list(zip(np.ones_like(relevant_tags)*2,relevant_tags))
# print(object_tags)
t1 = time.time()
intersections = []
for object_tag in object_tags:
    intersection = gmsh.model.occ.fragment([object_tag],[(2,plane)],removeObject=False,removeTool=False)
    gmsh.model.occ.synchronize()
    # print(intersection)
    intersections.append(intersection[0])
t2 = time.time()
print(t2-t1)

print(intersections)

#isolate curve segments from fragmentations
planebb = gmsh.model.occ.get_bounding_box(2,plane)
(xmin,ymin,zmin,xmax,ymax,zmax) = planebb
eps = np.abs(.001*np.average((dx,dy,dz)))
print(eps)
curves = gmsh.model.occ.getEntitiesInBoundingBox(xmin+eps,ymin-eps,zmin+eps,xmax-eps,ymax+eps,zmax-eps,1)
print(curves)

thick = 0.00127

#need to be able to handle interior and exterior separately
curve_offset0 = gmsh.model.occ.copy([curves[0]])
gmsh.model.occ.translate(curve_offset0,0,0,-thick)

curve_offset1 = gmsh.model.occ.copy([curves[1]])
gmsh.model.occ.translate(curve_offset1,thick,0,0)

curve_offset2 = gmsh.model.occ.copy([curves[2]])
gmsh.model.occ.translate(curve_offset2,0,0,thick)

curve_offset3 = gmsh.model.occ.copy([curves[3]])
gmsh.model.occ.translate(curve_offset3,-thick,0,0)


# # option 1 SLOW
# t1 = time.time()
# intersection = gmsh.model.occ.fragment([(2,plane)],surfs_tags_vec)
# # intersection = gmsh.model.occ.fragment(surfs_tags_vec,[(2,plane)])
# t2 = time.time()
# #takes about 2 min :(
# print(t2-t1)

# #option 2: define bounding box based on cutplane and then fragment
# pts = gmsh.model.occ.get_bounding_box(2,plane)
# print(pts)

# relevant_surfs = gmsh.model.occ.getEntitiesInBoundingBox(*pts,dim=2)
# print(relevant_surfs)


#option 3: intersect

# intersection2 = gmsh.model.occ.intersect(surfs_tags_vec,[(2,plane)])
# t2 = time.time()
# print(t2-t1)
# intersection3 = gmsh.model.occ.intersect([(2,plane)],surfs_tags_vec)
# t3 =time.time()
# print(t3-t2)

# print(surfs_tags_vec[0])
# print(surfs_tags_vec[1:])
# fuse = gmsh.model.occ.fuse([surfs_tags_vec[0]],surfs_tags_vec[1:])
# print(fuse)
# intersection2 = gmsh.model.occ.intersect(fuse[0],[(2,plane)])
gmsh.model.occ.synchronize()

# gmsh.model.

# print(gmsh.model.occ.getMatrixOfInertia(1,1))
gmsh.fltk.run()

# print(gmsh.model.getPhysicalGroupsForEntity(2,gmsh.model.getEntities(2)))