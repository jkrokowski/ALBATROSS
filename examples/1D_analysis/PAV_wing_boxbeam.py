#create gmsh readable xcs from PEGASUS wing data
import scipy.io
import os
import meshio
    
import pyvista 
import numpy as np
from dolfinx.plot import create_vtk_mesh

from FROOT_BAT.utils import mat_to_mesh,beam_interval_mesh_3D
from FROOT_BAT.beam_model import BeamModel
path = os.getcwd()

node_x = np.array([[0.        ],
       [0.43664183],
       [0.85817859],
       [1.27971536],
       [1.70125212],
       [2.12278888],
       [2.54432565],
       [2.96586241],
       [3.38739917],
       [3.80893593],
       [4.24557775]])

axial_pos_ne = list(np.ones((node_x.shape[0]-1)))
beam_el_num = 10
axial_ne = list(beam_el_num*np.ones((node_x.shape[0]-1)))

meshname_axial_pos = 'PAV_axial_postion_mesh'
meshname_axial = 'PAV_axial_mesh'

#mesh for locating beam cross-sections along beam axis
pts = np.concatenate([node_x,np.zeros((len(node_x),2))],axis=1)
axial_pos_mesh = beam_interval_mesh_3D(pts,axial_pos_ne,meshname_axial_pos)

#mesh used for 1D analysis
axial_mesh = beam_interval_mesh_3D(pts,axial_ne,meshname_axial)
if True:
    plotter = pyvista.Plotter()
    #plot xcs
    # for i,msh in enumerate(msh_list):
    #     num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
    #     topology, cell_types, x = create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

    #     grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    #     # grid = grid.translate((centroid_list[i][0],centroid_list[i][1],centroid_list[i][2]))
    #     actor = plotter.add_mesh(grid,show_edges=True)

    #plot axial meshes:
    def plot_mesh(msh,style='o'):
        num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
        topology, cell_types, x = create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        # grid = grid.translate((centroid_list[i][0],centroid_list[i][1],centroid_list[i][2]))
        if style=='o':
            actor = plotter.add_mesh(grid,show_edges=True,style='points',color='k',render_points_as_spheres=True)
        elif style=='x':
            actor = plotter.add_mesh(grid,show_edges=True,style='points',color='r',point_size=10)
        else:
            actor = plotter.add_mesh(grid,show_edges=True,style='wireframe',color='k')

    plot_mesh(axial_mesh)
    plot_mesh(axial_pos_mesh,style='x')
    plotter.show_axes()

    plotter.show()

#define orienation of primary cross-section axis
orientations = np.tile([0,1,0],len(node_x))
#collect relevant beam model properties
xc_info = [K_list,axial_pos_mesh,orientations]
#initialize beam model
PAV_wing = BeamModel(axial_mesh,xc_info,xc_type='analytical',shape='box')

#gather loading data
pts = PAV_wing.axial_pos_mesh.geometry.x[1:,:]
fz_lbf= np.array([[325.1311971,
                 318.2693761,
                 292.5067646,
                 281.8637954,
                 274.6239472,
                 264.3659094,
                 254.0378545,
                 255.3444864,
                 225.7910602,
                 178.5249412]]).T

lbf_to_N = 4.4482216
fz = lbf_to_N*fz_lbf

f = np.concatenate([np.zeros((fz.shape[0],2)),fz.reshape(((fz.shape[0],1)))],axis=1)
# f = np.concatenate([np.zeros((fz.shape[0],1)),fz,np.zeros((fz.shape[0],1))],axis=1)

PAV_wing.add_clamped_point((0,0,0))

PAV_wing.add_point_load(f,pts)

PAV_wing.solve()
m_to_in = 39.37007874

print(np.max(PAV_wing.uh.sub(0).collapse().x.array)*m_to_in)
PAV_wing.plot_axial_displacement(warp_factor=10)

print()

 


