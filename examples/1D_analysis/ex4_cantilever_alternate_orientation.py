'''
Uniform rectangular prismatic cantilevered beam with a single tip load
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script is nearly identical to example 1, but demonstrates that the 
1D beam axis need not be aligned with a cartesian direction, yet we can
still produce the same answer:
'''
import numpy as np

import ALBATROSS

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 20 #number of quad elements per side on xc mesh
W = .1 #xs width
H = .1 #xs height
A = W*H #xs area
L = 20 

#define tip load magnitude 
F = .01 

#beam endpoint locations
p1 = (0,0,0)
p2 = (np.sqrt(2)*L/2,np.sqrt(2)*L/2,0)

#create cross-sectional mesh
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
squareXSmesh = ALBATROSS.mesh.create_rectangle(points,[N,N])

#initialize material object
unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10e6,'nu':0.2},
                                           density=2700)

#initialize and run cross-sectional analysis
squareXS = ALBATROSS.cross_section.CrossSection(squareXSmesh,[unobtainium])
squareXS.get_xs_stiffness_matrix()
xs_list = [squareXS]

#create a beam axis
meshname = 'ex_1'
nodal_points = [p1,p2]
# number of segments of the beams that use different cross-sections
num_segments = len(nodal_points)-1 
num_ele = [100] #number of subdivisions for each beam segment
beam_axis = ALBATROSS.axial.BeamAxis(nodal_points,num_ele,meshname)

#define orientation of each xs with a vector
#note we don't even have to specify a unit vector
orientations = np.tile([- np.sqrt(2),np.sqrt(2),0],num_segments)

#collect all xs information
xs_adjacency_list = [[0,0]] #this is the trivial connectivity for a uniform beam
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using beam axis and definition of xs's
CantileverBeam = ALBATROSS.beam.Beam(beam_axis,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
CantileverBeam.plot_xs_orientations()

#applied fixed bc to first endpoint
CantileverBeam.add_clamped_point(p1)

#apply force at free end in the negative z direction
CantileverBeam.add_point_load([(0,0,-F)],[p2])

#solve the linear problem
CantileverBeam.solve()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################
#shows plot of 1D displacement solution (recovery doesn't need be executed)
CantileverBeam.plot_axial_displacement(warp_factor=10)

#recovers the 3D displacement field over each xs
CantileverBeam.recover_displacement()

#shows plot of stress over cross-section 
CantileverBeam.recover_stress()

#plots both 1D and 2D solutions together
CantileverBeam.plot_xs_disp_3D()

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Tip Deflection for point load')
print('EB analytical solution:')
E=unobtainium.E
I = W*H**3/12
print( (-F*L**3)/(3*E*I) )

print('ALBATROSS computed value:')
print(CantileverBeam.get_local_disp([p2])[0][2])
print('------')

print('Maximum Stress for point load (at root of beam)')
print('EB analytical solution:')
M = -F*L #maximum moment
print( (-H/2)* (M) / I  )
print('ALBATROSS computed value:')
print( CantileverBeam.get_max_stress() )
