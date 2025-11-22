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

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 2 #num quad elements through thickness
L = 20 
h_to_f = 10
w_to_w = 10

H = 1
W = 1
tf = H/h_to_f
tw = W/w_to_w

A = tf*W + tw*(H-tf)

#define tip load magnitude and direction
F = .1 
loading = 'z'

#beam endpoint locations
p1 = (0,0,0)
p2 = (L,0,0)

#create cross-sectional mesh
meshname_section = 'T_section'
msh = ALBATROSS.mesh.create_T_section([H,W,tf,tw],[N,N],meshname_section)

#initialize material object
unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10e6,'nu':0.2},
                                           density=2700)

#initialize and run cross-sectional analysis
xs = ALBATROSS.cross_section.CrossSection(msh,[unobtainium])
xs.plot_mesh()
xs.get_xs_stiffness_matrix()
xs_list = [xs]

#create a beam axis
meshname = 't-section'
nodal_points = [p1,p2]
# number of segments of the beams that use different cross-sections
num_segments = len(nodal_points)-1 
num_ele = [10] #number of subdivisions for each beam segment
beam_axis = ALBATROSS.axial.BeamAxis(nodal_points,num_ele,meshname)

#define orientation of each xs with a vector
orientations = np.tile([0,1,0],num_segments)

#collect all xs information
xs_adjacency_list = [[0]] #this is the trivial connectivity for a uniform beam 
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
if loading == 'z':
    CantileverBeam.add_point_load([(0,0,-F)],[p2])
if loading == 'y':
    CantileverBeam.add_point_load([(0,-F,0)],[p2])

#solve the linear problem
CantileverBeam.solve()

#compute beam mass:
CantileverBeam.get_mass()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

#shows plot of 1D displacement solution (recovery doesn't need be executed)
CantileverBeam.plot_axial_displacement(warp_factor=1e3)

#recovers the 3D displacement field over each xs
CantileverBeam.recover_displacement()

#shows plot of stress over cross-section 
CantileverBeam.recover_stress()

#plots both 1D and 2D solutions together
CantileverBeam.plot_xs_disp_3D()

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
# however, we could also compute the shear based deflection as F*L/kappa*G*A
# additionally, for bending about the asymmetric portion of this T-section, we have to properly 
# account for the fact that even though our model can arbitrarily select the beam axis location 
# in the cross-seciton  
print('Max Tip Deflection for point load')
print('EB analytical solution:')
E=unobtainium.E
if loading == 'z':
    I1_O = ( (tw*H**3)/12 + 
      ( (((W-tw)*tf**3)/12) 
       + ((tf*(W-tw)))*((H-tf)/2)**2) )
    I1 = ( ((tw*H**3)/12 + ( (tw*H)* xs.zavg**2) ) +
      ( (((W-tw)*tf**3)/12) + ((tf*(W-tw)))*((H-tf)/2 -xs.zavg)**2) )
    print('deflection')
    print( (-F*L**3)/(3*E*I1) )
if loading == 'y':
    print('deflection')
    I2 = ((tf*W**3)/12) + ((H-tf)*tw**3)/12
    print( (-F*L**3)/(3*E*I2) )

print('ALBATROSS computed value:')
if loading == 'z':
    print(CantileverBeam.get_local_disp([p2])[0][2])
if loading == 'y':
    print(CantileverBeam.get_local_disp([p2])[0][1])
print('------')

print('Maximum Stress for point load (at root of beam)')
print('EB analytical solution:')
M = -F*L #maximum moment
if loading == 'y':
    print((-W/2)* (M) / I2 )
if loading == 'z':
    print('compressive stress (bottom of web):')
    print(-(-H/2-xs.zavg)* (M) / I1)
    print('Tensile stress (bottom of web):')
    print(-(H/2-xs.zavg)* (M) / I1  )
print('ALBATROSS computed maximum magnitude stress:')
print( CantileverBeam.get_max_stress() )
print()