'''
Uniform rectangular prismatic cantilevered beam with a single tip load
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script demonstrates:
    -axial (1D) analysis
    -cross-section (2D) analysis
    -(1D) <--> (2D) connection functionality 
'''
import numpy as np
import ALBATROSS
from dolfinx import mesh
from mpi4py import MPI


#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 6 #number of quad elements per side on xc mesh
W = 0.1 #xs width
H = 0.1 #xs height
A = W*H #xs area
L = 2.0 
# section_type = 'L'

#define tip load magnitude 
F = 1000.0
loading = 'z'

#beam endpoint locations
p1 = (0,0,0)
p2 = (L,0,0)

offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

tf = H/h_to_f
tw = W/w_to_w

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2
mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
# if section_type == 'L':
#     mesh_1.geometry.x[:,0] += -0.45
mesh_1.name = 'w'

#================= initialize individual cross-sections ===========#
meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':70e9,'nu':0.33},
                                           density=2700)

XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

#================= initialize coupled cross-section ===========#
TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen_u=1e4,pen_t=1e4)
TXS_nm.plot_meshes()

#identify meshes:
mesh_A = TXS_nm.XSs[0].msh
mesh_B = TXS_nm.XSs[1].msh

TXS_nm.get_xs_stiffness_matrix()

xs_list = [TXS_nm]

#create a beam axis
meshname = 'txs_nm_beam'
nodal_points = [p1,p2]
# number of segments of the beams that use different cross-sections
num_segments = len(nodal_points)-1 
num_ele = [10] #number of subdivisions for each beam segment
beam_axis = ALBATROSS.axial.BeamAxis(nodal_points,num_ele,meshname)

#define orientation of each xs with a vector
orientations = np.tile([0,1,0],num_segments+1)

#collect all xs information
xs_adjacency_list = [[0,0]] #this is the trivial connectivity for a uniform beam 
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using beam axis and definition of xs's
CantileverBeam = ALBATROSS.beam.Beam(beam_axis,xs_info,segment_type='LINEAR')

#show the orientation of each xs and the interpolated orientation along the beam
CantileverBeam.plot_xs_orientations()

#applied fixed bc to first endpoint
CantileverBeam.add_clamped_point(p1)

#apply force at free end in the negative z direction
if loading == 'z':
    CantileverBeam.add_point_load([(0,0,-F)],[p2])
if loading == 'y':
    CantileverBeam.add_point_load([(0,0,-F)],[p2])

#solve the linear problem
CantileverBeam.solve()

#compute beam mass:
CantileverBeam.get_mass()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

#shows plot of 1D displacement solution (recovery doesn't need be executed)
CantileverBeam.plot_axial_displacement(warp_factor=10)

# #recovers the 3D displacement field over each xs
# CantileverBeam.recover_displacement()

# #shows plot of stress over cross-section 
# CantileverBeam.recover_stress()

# #plots both 1D and 2D solutions together
# CantileverBeam.plot_xs_disp_3D()


print('ALBATROSS computed value:')
print(CantileverBeam.get_local_disp([p2])[0][2])
print('------')

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Tip Deflection for point load')
print('EB analytical solution:')
E=unobtainium.E
if loading == 'z':
    I1_O = ( (tw*H**3)/12 + 
      ( (((W-tw)*tf**3)/12) 
       + ((tf*(W-tw)))*((H-tf)/2)**2) )
    I1 = ( ((tw*H**3)/12 + ( (tw*H)* TXS_nm.zavg**2) ) +
      ( (((W-tw)*tf**3)/12) + ((tf*(W-tw)))*((H-tf)/2 -TXS_nm.zavg)**2) )
    print('deflection')
    print( (-F*L**3)/(3*E*I1) )
if loading == 'y':
    print('deflection')
    I2 = ((tf*W**3)/12) + ((H-tf)*tw**3)/12
    print( (-F*L**3)/(3*E*I2) )


print('ALBATROSS computed value:')
print(CantileverBeam.get_local_disp([p2])[0][2])
print('------')

print('Maximum Stress for point load (at root of beam)')
print('EB analytical solution:')
M = -F*L #maximum moment
print( (-H/2)* (M) / I  )
print('ALBATROSS computed value:')
print( CantileverBeam.get_max_stress() )
