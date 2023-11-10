#create gmsh readable xcs from PEGASUS wing data
import scipy.io
import os
import meshio
    
import pyvista 
import numpy as np
from dolfinx.plot import create_vtk_mesh

from FROOT_BAT.utils import mat_to_mesh

path = os.getcwd()

msh_list = []
K_list = []
centroid_list = []
for i in range(11):
    filename = path+'/PAV_xcs/PavCs'+str(i+1)+'.mat'

    other_data = ['cs_K','cs_centroid']

    msh,data = mat_to_mesh(filename,other_data)

    msh_list.append(msh)
    K_list.append(data[0])
    centroid_list.append((data[1][0,1],data[1][0,2],data[1][0,0]))


# plotter = pyvista.Plotter()
# for i,msh in enumerate(msh_list):
#     num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
#     topology, cell_types, x = create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

#     grid = pyvista.UnstructuredGrid(topology, cell_types, x)
#     grid = grid.translate((centroid_list[i][0],centroid_list[i][1],centroid_list[i][2]))
#     actor = plotter.add_mesh(grid,show_edges=True)
# plotter.show_axes()

# plotter.show()

 


