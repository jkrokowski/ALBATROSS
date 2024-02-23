#create gmsh readable xcs from PEGASUS wing data
import scipy.io
import os
import meshio
    
import pyvista 
import numpy as np
from dolfinx.plot import create_vtk_mesh

from ALBATROSS.utils import mat_to_mesh,beam_interval_mesh_3D
from ALBATROSS.beam_model import BeamModel
path = os.getcwd()

#unit conversions
lbf_to_N = 4.4482216
in_to_m = 0.0254 
m_to_in = 39.37007874
m_to_ft = 3.28084

#beam input data
num_xc = 11
# node_x = np.array([0.         ,0.43664183,0.85817859,1.27971536,1.70125212,2.12278888,2.54432565,2.96586241,3.38739917,3.80893593,4.24557775])
node_x = np.array([0, 4.26719858e-01, 8.74003116e-01, 1.28012847e+00,1.70684846e+00, 2.13356845e+00, 2.56028844e+00, 2.98700842e+00,3.41373953e+00, 3.84045955e+00, 4.26720000e+00])
node_y = np.array([ 0.0000000e+00, -1.9808000e-04, -8.0169820e-02, -1.3107788e-01,-1.7653760e-01, -2.2199733e-01, -2.6745705e-01, -3.1291677e-01,-3.5472852e-01, -3.8419568e-01, -4.1366405e-01])
node_z = np.zeros_like(node_y)

width = in_to_m*np.array([38.22183283, 38.20994261, 33.78853197, 30.92259183, 28.36566531,25.80873883, 23.25181241, 20.69488606, 18.3431398 , 16.68571462,15.02832841])
height = in_to_m*np.array([12.29249872, 12.27959289, 10.78952744,  9.93768319,  9.11595932,8.29423545,  7.47251157,  6.6507877 ,  5.89497505,  5.36233072, 4.82965527])
t_cap =  0.05 *in_to_m * np.ones(num_xc)
t_web =  0.05 *in_to_m * np.ones(num_xc)

axial_pos_ne = list(np.ones((num_xc-1)))
beam_el_num = 100
axial_ne = list(beam_el_num*np.ones((num_xc-1)))

meshname_axial_pos = 'PAV_axial_postion_mesh'
meshname_axial = 'PAV_axial_mesh'

#mesh for locating beam cross-sections along beam axis
pts = np.concatenate([node_x.reshape((num_xc,1)),node_y.reshape((num_xc,1)),node_z.reshape((num_xc,1))],axis=1)
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
    plotter.view_xy()

    plotter.show()

#define orienation of primary cross-section axis
orientations = np.tile([0,1,0],len(node_x))
#collect relevant beam model properties
xs_params = []
for i in range(num_xc):
    params={'shape': 'box',
            'h': height[i],
            'w': width[i],
            't_h': t_cap[i],
            't_w': t_web[i],
            'E':7.31E10,
            'nu':0.40576923076923066}
            # 'nu':0.333}
    xs_params.append(params)
xs_info = [xs_params,axial_pos_mesh,orientations]
#initialize beam model
PAV_wing = BeamModel(axial_mesh,xs_info,xs_type='analytical')
for i,xc in enumerate(PAV_wing.xss):
    print('stiffness matrix '+str(i))
    print(np.diag(xc.K))
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


fz = lbf_to_N*fz_lbf

f = np.concatenate([np.zeros((fz.shape[0],2)),fz.reshape(((fz.shape[0],1)))],axis=1)
# f = np.concatenate([np.zeros((fz.shape[0],1)),fz,np.zeros((fz.shape[0],1))],axis=1)
# PAV_wing.plot_xc_orientations()
PAV_wing.add_clamped_point((0,0,0))

PAV_wing.add_point_load(f,pts)

PAV_wing.solve()
print("tip deflection:")
print(np.max(PAV_wing.uh.sub(0).collapse().x.array)*m_to_in)
print("spanwise data:")
spanwise_disp=(PAV_wing.uh.sub(0).collapse().x.array*m_to_in).reshape(((num_xc-1)*beam_el_num+1,3))
print(spanwise_disp)
PAV_wing.plot_axial_displacement(warp_factor=10)

print()

 


