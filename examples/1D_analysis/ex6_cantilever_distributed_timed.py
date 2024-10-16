'''
Uniform rectangular prismatic cantilevered beam with a single distributed load
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script demonstrates:
    -axial (1D) analysis
    -cross-section (2D) analysis
    -(1D) <--> (2D) connection functionality 
'''
import numpy as np

import ALBATROSS 
import time

t0 = time.time()
#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 15 #number of quad elements per side on xc mesh
W = .1 #xs width
H = .1 #xs height
A = W*H #xs area
L = 2 

#define distributed load magnitude (gravitational force)
g = 5000*9.81

#beam endpoint locations
p1 = (0,0,0)
p2 = (L,0,0)

#create cross-sectional mesh
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
squareXSmesh = ALBATROSS.mesh.create_rectangle(points,[N,N])
t1 = time.time()
#initialize material object
unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10e6,'nu':0.2},
                                           density=2.7e-3)

#initialize and run cross-sectional analysis
squareXS = ALBATROSS.cross_section.CrossSection(squareXSmesh,[unobtainium])
squareXS.get_xs_stiffness_matrix()
xs_list = [squareXS]
t2 = time.time()

#create a beam axis
meshname = 'ex_1'
nodal_points = [p1,p2]
# number of segments of the beams that use different cross-sections
num_segments = len(nodal_points)-1 
num_ele = [1000] #number of subdivisions for each beam segment
beam_axis = ALBATROSS.axial.BeamAxis(nodal_points,num_ele,meshname)
t3 = time.time()

#define orientation of each xs with a vector
orientations = np.tile([0,1,0],num_segments)

#collect all xs information
xs_adjacency_list = [[0,0]] #this is the trivial connectivity for a uniform beam
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using beam axis and definition of xs's
CantileverBeam = ALBATROSS.beam.Beam(beam_axis,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
# CantileverBeam.plot_xs_orientations()
t4 = time.time()

#applied fixed bc to first endpoint
CantileverBeam.add_clamped_point(p1)

#add distributed load
CantileverBeam.add_dist_load((0,0,-g))
t5 = time.time()
#solve the linear problem
CantileverBeam.solve()
t6 = time.time()
#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

#shows plot of 1D displacement solution (recovery doesn't need be executed)
# CantileverBeam.plot_axial_displacement(warp_factor=10)

#recovers the 3D displacement field over each xs
CantileverBeam.recover_displacement()

#shows plot of stress over cross-section 
CantileverBeam.recover_stress()

t7 = time.time()

#plots both 1D and 2D solutions together
CantileverBeam.plot_xs_disp_3D()
#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Deflection for point load (EB analytical analytical solution)')
E=unobtainium.E
rho = unobtainium.density
I = W*H**3/12
q = rho*A*g
print( (-q*L**4)/(8*E*I) )

print('Max vertical deflection of centroid:')
print(CantileverBeam.get_local_disp([p2])[0][2])

print("Total time:")
print(t7-t0)
print("2D Mesh construction:")
print(t1-t0)
print("2D analysis:")
print(t2-t1)
print("1D Mesh construction:")
print(t3-t2)
print("1D+2D connection:")
print(t4-t3)
print("1D loads and BC application:")
print(t5-t4)
print("1D analysis:")
print(t6-t5)
print("Displacement and Stress recovery:")
print(t7-t6)