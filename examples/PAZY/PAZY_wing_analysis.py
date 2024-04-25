#example of cross-sectional analysis of an multimaterial isotropic square:

from mpi4py import MPI
from dolfinx import mesh
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh
import ufl

import ALBATROSS

import os
import sys

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

#################################################################
################# IMPORT CROSS-SECTIONAL MESHES #################
#################################################################

#read in meshes
ribXSname= "beam_crosssection_rib_221_quad"
mainXSname = "beam_crosssection_2_95_quad"

def import_cross_section_mesh(xsName):
    fileName =  xsName + ".xdmf"
    filePath=os.path.join(dirpath,fileName)
    with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
        #read in mesh and convert to a topological AND geometrically 2D mesh
        in_mesh = xdmf.read_mesh(name="Grid")
        shape = in_mesh.ufl_cell().cellname()
        degree = 1
        cell = ufl.Cell(shape)
        adj_list =in_mesh.topology.connectivity(2,0)
        cells = adj_list.array.reshape((adj_list.num_nodes,adj_list.links(0).shape[0]))
        chord = .1 #m
        #need to be conscious of which axis we are using
        x_points = in_mesh.geometry.x[:,0] - .25*chord #adjust x location of mesh
        y_points = in_mesh.geometry.x[:,2]
        points = np.stack([x_points,y_points],axis=1)
        domain= mesh.create_mesh(MPI.COMM_WORLD,
                            cells,
                            points,
                            ufl.Mesh(ufl.VectorElement("Lagrange",cell,degree)) )
        #read celltags
        ct = xdmf.read_meshtags(in_mesh, name="Grid")   

    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
    
    # #adjust to make tags start at 0
    # ct.values[:]=ct.values-np.min(ct.values)

    #TOTAL HACK here, but just making sure 0 == Nylon, 1 == AL
    ct.values[ct.values == 37] = 0
    ct.values[ct.values == 38] = 1
    ct.values[ct.values == 13] = 1
    ct.values[ct.values == 14] = 0
        
    return domain,ct

ribXSmesh,ribXSct = import_cross_section_mesh(ribXSname)
mainXSmesh,mainXSct = import_cross_section_mesh(mainXSname)

#plot mesh:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'

for domain,ct in zip([ribXSmesh,mainXSmesh],[ribXSct,mainXSct]):
    p = pyvista.Plotter(window_size=[800, 800])
    num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
    topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
    grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    grid.cell_data["Marker"] = ct.values
    p.add_mesh(grid, show_edges=True)
    p.show_grid()
    # p.view_xy()
    p.view_xy()
    p.show_axes()
    p.show()

aluminum7075 = ALBATROSS.material.Material(name='Aluminium7075',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':71e9,'nu':0.33},
                                           density=2795,
                                           celltag=1)
nylon_pa12 = ALBATROSS.material.Material(name='NylonPA12',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':1.7e9,'nu':0.394},
                                           density=930,
                                           celltag=0)

#collect materials in list
mats = [aluminum7075,nylon_pa12]

#initialize cross-section object
ribXS = ALBATROSS.cross_section.CrossSection(ribXSmesh,mats,celltags=ribXSct)

#compute the stiffness matrix
ribXS.getXSStiffnessMatrix()

#initialize cross-section object
mainXS =  ALBATROSS.cross_section.CrossSection(mainXSmesh,mats,celltags=mainXSct)

#compute the stiffness matrix
mainXS.getXSStiffnessMatrix()

# xs_list = [ribXS,mainXS]
xs_list = [mainXS,ribXS]
# xs_list = [mainXS]
# xs_list = [ribXS]

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

deg="7"
pressure="1825"
conservation = "conservative"
aero_forces_filename = "beam_forces_"+conservation+"_"+deg+"deg_pitch_"+pressure+"pa_dyn_pres_5x55vlmgrid_116beamnodes.npy"
aero_forces_file=os.path.join(dirpath,'vlm_forces_for_beam',aero_forces_filename)
# aero_forces = np.load(aero_forces_filename,allow_pickle=True)

# approximate airspeed
rho_air = 1.225 #kg/m^3 at STP
pres = float(pressure)
v_air = np.sqrt(2*pres/rho_air)
print('approximate airspeed:',v_air)
aero_force_ndarray = np.load(aero_forces_file,allow_pickle=True)
aero_force_dict = aero_force_ndarray[()] #extract dict from 0d numpy object array 
aero_forces = aero_force_dict['nodal_forces']
nodal_coordinates_from_aero = aero_force_dict['node_coordinates']

#stated wingspan:
L = .550 
#wing span offsets from CAD
start = np.array([0,12,28.5])*1e-3
middle = np.tile([5,33.25],12)*1e-3
end = np.array([5,27.5,27])*1e-3
# adjust to make sure wingspan matches other simulations
tip_correction =np.sum([np.sum(i) for i in [start,middle,end]])-L
end[-1] -=  tip_correction

#enter the y coordinates of the wingspan
axial_coords_offsets = np.concatenate((start,middle,end))
axial_coords_y = np.cumsum(axial_coords_offsets)
x_offset = 0.025 #meters (along span)
axial_coords_x = np.tile(x_offset,axial_coords_y.shape[0])
axial_coords_z = np.zeros_like(axial_coords_y)
nodal_coordinates = np.stack([axial_coords_x,axial_coords_y,axial_coords_z],axis=1)
root_pt = nodal_coordinates[0,:]
tip_pt = nodal_coordinates[-1,:]

#check that specificied wingspan is the same as the stated wingspan
assert(np.isclose(L,np.linalg.norm(root_pt-tip_pt)))

#1D mesh for locating beam cross-sections along beam axis
# meshname_axial_pos = 'axial_postion_mesh'
num_segments = nodal_coordinates.shape[0]-1 # number of xs's used
# num_ele_pos = np.ones((num_segments))
# axial_pos_mesh = ALBATROSS.utils.beam_interval_mesh_3D(nodal_coordinates,num_ele_pos,meshname_axial_pos)

# #plot axial position mesh
# p = pyvista.Plotter(window_size=[800, 800])
# domain = axial_pos_mesh
# num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
# topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
# grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# p.add_mesh(grid, style='points',color='b',show_edges=True)
# p.show_grid()
# # p.view_xy()
# p.view_isometric()
# p.show_axes()
# p.show()

#1D mesh used for 1D analysis
meshname = 'PAZY'
# ne_1D = nodal_coordinates.shape[0] #number of elements for 1D mesh
mesh_size = L/100 #approximate minimum mesh element size
num_ele = np.ceil(axial_coords_offsets[1:]/mesh_size).astype('int')
# axial_mesh = ALBATROSS.utils.beam_interval_mesh_3D(nodal_coordinates,num_ele,meshname_axial)

beam_axis = ALBATROSS.axial.BeamAxis(nodal_coordinates,num_ele,meshname)


#output for aero force generation
with open(os.path.join(dirpath,'segment_locations.npy'), 'wb') as f:
    np.save(f,nodal_coordinates)
with open(os.path.join(dirpath,'nodal_coords.npy'), 'wb') as f:
    np.save(f,beam_axis.axial_mesh.geometry.x)

#define orientation of each xs with a vector
orientations = np.tile([-1,0,0],num_segments)

#construct an adjacency list to map xs's to segments
rib_sect = [1,1]
main_sect = [0,0]
num_rib_main_sects = int(start.shape[0] / 2) + int(middle.shape[0] / 2) + int(end.shape[0] / 2)
xs_adjacency_list = np.concatenate([np.tile([rib_sect,main_sect],(num_rib_main_sects,1)),[rib_sect]])
# xs_adjacency_list = np.tile(main_sect,(29,1))
#collect all xs information
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using 1D mesh and definition of xs's
PAZYWing = ALBATROSS.beam.Beam(beam_axis,xs_info)

# #show the orientation of each xs and the interpolated orientation along the beam
# PAZYWing.plot_xs_orientations()

#apply force at free end in the negative z direction
PAZYWing.add_point_load(aero_forces[1:,:],nodal_coordinates_from_aero[1:,:])
# PAZYWing.add_point_load([10,0,1000],[tip_pt])

#applied fixed bc to first endpoint
PAZYWing.add_clamped_point(root_pt)

#solve the linear problem
PAZYWing.solve()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

# from matplotlib import pyplot as plt
# xs_props = PAZYWing.k.vector.array
# row = 0
# EA = xs_props[[i*36 +row*6+row for i in range(axial_mesh.geometry.x.shape[0]-1)]]
# row = 3
# GJ = xs_props[[i*36 +row*6+row for i in range(axial_mesh.geometry.x.shape[0]-1)]]
# row = 4
# EI_flap = xs_props[[i*36 +row*6+row for i in range(axial_mesh.geometry.x.shape[0]-1)]]
# row = 5
# EI_lag = xs_props[[i*36 +row*6+row for i in range(axial_mesh.geometry.x.shape[0]-1)]]
# # print(xs_prop)
# # print(xs_prop.shape)
# print(axial_mesh.geometry.x.shape)

# fig,ax=plt.subplots()
# # ax.scatter(axial_mesh.geometry.x[:-1,1],EA)
# # ax.scatter(axial_mesh.geometry.x[:-1,1],GJ)
# ax.scatter(axial_mesh.geometry.x[:-1,1],EI_flap)
# # ax.scatter(axial_mesh.geometry.x[:-1,1],EI_lag)

# ax.set(xlabel='spanwise', ylabel='axial stiffness')
# ax.grid()
# plt.show()

#shows plot of 1D displacement solution (recovery doesn't need be executed)
PAZYWing.plot_axial_displacement(warp_factor=1)

#recovers the 3D displacement field over each xs
PAZYWing.recover_displacement(plot_xss=True)

#plots both 1D and 2D solutions together
PAZYWing.plot_xs_disp_3D()

#shows plot of stress over cross-section 
PAZYWing.recover_stress() # currently only axial component sigma11 plotted

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
# print('Max Deflection for point load (EB analytical analytical solution)')
# E = mats['Unobtainium']['MECH_PROPS']['E']
# I = W*H**3/12
# print( (-F*L**3)/(3*E*I) )

print('Max vertical deflection of centroid:')
print(PAZYWing.get_local_disp([tip_pt])[0][2])