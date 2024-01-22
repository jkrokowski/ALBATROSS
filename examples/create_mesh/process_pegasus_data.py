#create gmsh readable xcs from PEGASUS wing data
import scipy.io
import os
path = os.getcwd()
print(path)
mat = scipy.io.loadmat(path+'/PEGASUSwingboxdata/cs_tip_data.mat')

elems = mat['vabs_2d_mesh_elems']
nodes = mat['vabs_2d_mesh_nodes']
print('Number of nodes:')
print(len(nodes))
print('Number of Elements:')
print(len(elems))

import gmsh

gmsh.initialize()

xcname = 'PEGASUS_root'
gmsh.model.add(xcname)
gmsh.model.set_current(xcname)
for node in nodes[0:20000]:
    gmsh.model.occ.add_point(node[0],node[1],0)

for elem in elems[:,0:3]:
    print(elem)
    gmsh.model.occ.add_line(elem[0],elem[1])
    gmsh.model.occ.add_line(elem[1],elem[2])
    gmsh.model.occ.add_line(elem[0],elem[2])
gmsh.model.occ.synchronize()

gmsh.fltk.run()
