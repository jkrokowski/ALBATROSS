'''
Uniform rectangular prismatic beam, single center load, fixed on both ends
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script demonstrates:
    -axial (1D) analysis
    -cross-section (2D) analysis
    -(1D) <--> (2D) connection functionality 
'''
import ALBATROSS
import numpy as np

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

mesh2d_0 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
meshes2D = 2*[mesh2d_0] #duplicate the mesh for endpoint

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':10e6,'nu':0.2} ,
                        'DENSITY':2.7e-3}
                        }

#1D mesh for locating beam cross-sections along beam axis
meshname_axial_pos = 'axial_postion_mesh'
ne_2D = len(meshes2D)-1 # number of xs's used
axial_pos_mesh = utils.beam_interval_mesh_3D([p1,p2],[ne_2D],meshname_axial_pos)

#1D mesh used for 1D analysis
meshname_axial = 'axial_mesh'
ne_1D = 100 #number of elements for 1D mesh
axial_mesh = utils.beam_interval_mesh_3D([p1,p2],[ne_1D],meshname_axial)

#define orientation of each xs with a vector
orientations = np.tile([0,1,0],num_segments)

#collect all xs information
xs_info = [meshes2D,mats2D,axial_pos_mesh,orientations]

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

#plots both 1D and 2D solutions together
FixedFixedBeam.plot_xs_disp_3D()

#computes stress over cross-sections and plots 
FixedFixedBeam.recover_stress() # currently plots only axial component sigma11

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Deflection for point load (EB analytical analytical solution)')
E = mats['Unobtainium']['MECH_PROPS']['E']
rho = mats['Unobtainium']['DENSITY']
I = H**4/12
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