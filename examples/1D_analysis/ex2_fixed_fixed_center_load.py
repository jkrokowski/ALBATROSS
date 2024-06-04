'''
Uniform rectangular prismatic beam, single center load, fixed on both ends
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script demonstrates:
    -axial (1D) analysis
    -cross-section (2D) analysis
    -(1D) <--> (2D) connection functionality 
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

#beam endpoints and midpoint locations
p1 = (0,0,0)
midpoint = (L/2,0,0)
p2 = (L,0,0)

#create cross-sectional mesh
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
squareXSmesh = ALBATROSS.mesh.create_rectangle(points,[N,N])

# initialize material object
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
orientations = np.tile([0,1,0],num_segments)

#collect all xs information
xs_adjacency_list = [[0,0]] #this is the trivial connectivity for a uniform beam
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using 1D mesh and definition of xs's
FixedFixedBeam = ALBATROSS.beam.Beam(beam_axis,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
FixedFixedBeam.plot_xs_orientations()

#applied fixed bc to endpoints
FixedFixedBeam.add_clamped_point(p1)
FixedFixedBeam.add_clamped_point(p2)

#apply force at free end in the negative z direction
FixedFixedBeam.add_point_load([(0,0,-F)],[midpoint])

#solve the linear problem
FixedFixedBeam.solve()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

#shows plot of 1D displacement solution (recovery doesn't need be executed)
FixedFixedBeam.plot_axial_displacement(warp_factor=10)

#plots both 1D and 2D solutions together
FixedFixedBeam.recover_displacement(plot_xss=True)

#computes stress over cross-sections and plots 
FixedFixedBeam.recover_stress()

#plots both 1D and 2D solutions together
FixedFixedBeam.plot_xs_disp_3D()

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Deflection for point load')
print('EB analytical solution:')
E=unobtainium.E
I = W*H**3/12
print( (-F*(L**3))/(192*E*I) )

print('ALBATROSS computed value:')
print(FixedFixedBeam.get_local_disp([midpoint])[0][2])
print('------')

print('Maximum Stress for point load (at root of beam)')
print('EB analytical solution:')
M = (-F*L) / 8 #maximum moment
print( (-H/2)* (M) / I  )
print('ALBATROSS computed value:')
print( FixedFixedBeam.get_max_stress() )