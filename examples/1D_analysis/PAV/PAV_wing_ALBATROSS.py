#create gmsh readable xcs from PEGASUS wing data
import os
    
import pyvista 
import numpy as np
from dolfinx.plot import create_vtk_mesh

from ALBATROSS.utils import mat_to_mesh,beam_interval_mesh_3D
from ALBATROSS.beam_model import BeamModel
path = os.getcwd()
import time
t0=time.time()
msh_list = []
K_list = []
centroid_list = []
idx = [0,1,2,3,4,5] # used to reorder from VABS to FROOT_BAT convention
num_xc = 11
for i in range(11):
    filename = path+'/PAV_xcs/PavCsData1/PavCs'+str(i+1)+'.mat'

    other_data = ['cs_K','cs_centroid']

    msh,data = mat_to_mesh(filename,other_data,plot_xs=False)

    msh_list.append(msh)

axial_pos_ne = list(np.ones((len(msh_list)-1)))
beam_el_num = 10
axial_ne = list(beam_el_num*np.ones((len(msh_list)-1)))

meshname_axial_pos = 'PAV_axial_postion_mesh'
meshname_axial = 'PAV_axial_mesh'

#mesh for locating beam cross-sections along beam axis
# pts = np.concatenate([np.array(centroid_list)[:,[0]],np.zeros((len(centroid_list),2))],axis=1)
node_x = np.array([0, 4.26719858e-01, 8.74003116e-01, 1.28012847e+00,1.70684846e+00, 2.13356845e+00, 2.56028844e+00, 2.98700842e+00,3.41373953e+00, 3.84045955e+00, 4.26720000e+00])
# node_y = -1*np.array([ 0.0000000e+00, -1.9808000e-04, -8.0169820e-02, -1.3107788e-01,-1.7653760e-01, -2.2199733e-01, -2.6745705e-01, -3.1291677e-01,-3.5472852e-01, -3.8419568e-01, -4.1366405e-01])
node_y = np.array([0.        , 0.06100983, 0.11915066, 0.17758845, 0.23607181,0.29455586, 0.3530391 , 0.41152375, 0.47003158, 0.52863541,0.58934847])
node_z = np.zeros_like(node_y)
pts = np.concatenate([node_x.reshape((num_xc,1)),node_y.reshape((num_xc,1)),node_z.reshape((num_xc,1))],axis=1)

axial_pos_mesh = beam_interval_mesh_3D(pts,axial_pos_ne,meshname_axial_pos)

#mesh used for 1D analysis
axial_mesh = beam_interval_mesh_3D(pts,axial_ne,meshname_axial)
if False:
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
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':7.31E10,'nu':0.40576923076923066} ,
                        'DENSITY':2.7e3}
                        }
mats2D = [mats for i in range(len(msh_list))]

#define orienation of primary cross-section axis
orientations = np.tile([0,1,0],len(msh_list))
#collect relevant beam model properties
# xc_info = [K_list,axial_pos_mesh,orientations]
xc_info = [msh_list,mats2D,axial_pos_mesh,orientations]
#initialize beam model
PAV_wing = BeamModel(axial_mesh,xc_info)
t1 =time.time()
print('time to anlayze beam xs: %f' % (t1-t0))
PAV_wing.plot_xs_orientations()
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
print((PAV_wing.uh.sub(0).collapse().x.array)*m_to_in)
PAV_wing.plot_axial_displacement(warp_factor=10)

PAV_wing.recover_displacement(plot_xss=True)

PAV_wing.plot_xs_disp_3D()

print()

 


