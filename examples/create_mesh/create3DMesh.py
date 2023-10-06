import gmsh
import math
import os
import sys
import numpy as np
import meshio

path = os.path.dirname(os.path.abspath(__file__))
mesh = meshio.read(os.path.join(path, os.pardir, '../../mesh2d_0.xdmf'))
mesh.write('mesh2d_0.gmsh',file_format='gmsh')

gmsh.initialize()
# gmsh.model.add('simple_model')
gmsh.open(os.path.join(path, os.pardir, '../../mesh2d_0.gmsh'))
print(gmsh.model.getDimension())

print(gmsh.model.setCurrent('mesh2d_0'))
# print(gmsh.model.getType(2,0))
# angle = gmsh.onelab.getNumber('Parameters/Angle for surface detection')[0]
# forceParametrizablePatches = gmsh.onelab.getNumber('Parameters/Create surfaces guaranteed to be parametrizable')[0]
# includeBoundary = True
# curveAngle = 180

# gmsh.model.mesh.classify_surfaces(angle * math.pi / 180., includeBoundary,
#                                      forceParametrizablePatches,
#                                      curveAngle * math.pi / 180.
# )
# gmsh.merge(os.path.join(path, os.pardir, '../../mesh2d_0.gmsh'))
# surf = gmsh.model.addDiscreteEntity(2)
# gmsh.model.mesh.classifySurfaces(angle=1)
# gmsh.model.mesh.createEdges()
# gmsh.model.mesh.addElements()
# gmsh.model.mesh.addElementsByType(1,2,[],
# gmsh.model.mesh.reclassifyNodes()
# geom = gmsh.model.mesh.createGeometry()
gmsh.model.mesh.createTopology()
# s = gmsh.model.getEntities(2)
# l = gmsh.model.geo.addSurfaceLoop([e[1] for e in s])
# print(gmsh.model.getEntities(1))
# gmsh.model.geo.addCurveLoop((1,1))
# print(gmsh.model.getEntities())
# print(gmsh.model.mesh.getNodes())
meshnodes = gmsh.model.mesh.getNodes()
nodeTags = meshnodes[0]
nodeCoords = meshnodes[1].reshape([len(nodeTags),3])

eleTypes,eleTags,eleNodTags = gmsh.model.mesh.getElements(1,1)
elements = eleTags[0]
elementNodeTags = eleNodTags[0].reshape((len(elements),2))
boundary_pts = list(set(eleNodTags[0]))
print(boundary_pts)
model_pts = []
for node in boundary_pts:
    idx = int(np.where(boundary_pts == node)[0][0])
    pt_tag = gmsh.model.geo.addPoint(nodeCoords[idx,0],nodeCoords[idx,1],nodeCoords[idx,2])
    model_pts.append([node,pt_tag])
print(elements)
print(elementNodeTags)
print(model_pts)
for nodes in elementNodeTags:
    print(nodes)
    # gmsh.model.geo.addLine()
# for i in range(len(eleTags[0])):
#     gmsh.model.geo.addLine()
    
gmsh.model.geo.synchronize()
print(gmsh.model.getEntities())
# gmsh.model.mesh.setSize([[2,0]],size=0.001)
# gmsh.option.setNumber('Mesh.MeshSizeMin', 0.005)
# gmsh.option.setNumber('Mesh.MeshSizeMin', 0.005)
# gmsh.model.mesh.generate(1)
# boundary_tags = gmsh.model.getBoundary([[2,0]])
# print(boundary_tags)
# print(gmsh.model.getType(1,1))

gmsh.fltk.run()
