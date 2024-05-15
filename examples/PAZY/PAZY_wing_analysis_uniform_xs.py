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
    p.view_isometric()
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
ribXS.get_xs_stiffness_matrix()

#initialize cross-section object
mainXS =  ALBATROSS.cross_section.CrossSection(mainXSmesh,mats,celltags=mainXSct)

#compute the stiffness matrix
mainXS.get_xs_stiffness_matrix()

# xs_list = [ribXS,mainXS]
xs_list = [mainXS]

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

L = .550 

aero_forces_filename = "beam_forces_conservative_7deg_pitch_1825pa_dyn_pres_5x55vlmgrid_50beamnodes.npy"
aero_forces_file=os.path.join(dirpath,aero_forces_filename)
# aero_forces = np.load(aero_forces_filename,allow_pickle=True)

# approximate airspeed
rho_air = 1.225 #kg/m^3 at STP
v_air = np.sqrt(2*1825/rho_air)
print(v_air)
aero_force_ndarray = np.load(aero_forces_file,allow_pickle=True)
aero_force_dict = aero_force_ndarray[()] #extract dict from 0d numpy object array 
aero_forces = aero_force_dict['nodal_forces']
nodal_coordinates = aero_force_dict['node_coordinates']

#beam endpoint locations
p1 = nodal_coordinates[0,:]
p2 = nodal_coordinates[-1,:]

#wing span
L = p2[1]-p1[1]

#1D mesh for locating beam cross-sections along beam axis
meshname_axial_pos = 'axial_postion_mesh'
num_segments = 1 # number of xs's used
axial_pos_mesh = ALBATROSS.mesh.beam_interval_mesh_3D([p1,p2],[num_segments],meshname_axial_pos)

# #1D mesh used for 1D analysis
# meshname_axial = 'axial_mesh'
ne_1D = nodal_coordinates.shape[0] #number of elements for 1D mesh
num_elements = np.tile([1],ne_1D)
# axial_mesh = ALBATROSS.mesh.beam_interval_mesh_3D(nodal_coordinates,np.ones_like(nodal_coordinates[:-1,0]),meshname_axial)

#define orientation of each xs with a vector
orientations = np.tile([-1,0,0],num_segments)

#construct an adjacency list to map xs's to segments
#   list of lists where each entry is a list of the xs index used in each segment 
#   (either 1 or 2 xs's per segment typically)
xs_adjacency_list = np.array([[0,0]])

#collect all xs information
xs_info = [xs_list,axial_pos_mesh,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using 1D mesh and definition of xs's
PAZYWing = ALBATROSS.beam.Beam(axial_mesh,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
PAZYWing.plot_xs_orientations()

#apply force at free end in the negative z direction
PAZYWing.add_point_load(aero_forces[1:,:],nodal_coordinates[1:,:])

#applied fixed bc to first endpoint
PAZYWing.add_clamped_point(p1)

#solve the linear problem
PAZYWing.solve()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

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
print(PAZYWing.get_local_disp([p2])[0][2])