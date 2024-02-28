'''
Uniform rectangular prismatic cantilevered beam with a single distributed load
Cross-sectional properties computed with ALBATROSS cross_section module
----
This script demonstrates:
    -axial (1D) analysis
    -cross-section (2D) analysis
    -(1D) <--> (2D) connection functionality 
'''
from mpi4py import MPI
import numpy as np
from dolfinx import mesh

from ALBATROSS import utils
from ALBATROSS.beam_model import BeamModel

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 10 #number of quad elements per side on xc mesh
W = .1 #xs width
H = .1 #xs height
A = W*H #xs area
L = 20 

#define distributed load magnitude (gravitational force)
g = 9.81

#beam endpoint locations
p1 = (0,0,0)
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
orientations = np.tile([0,1,0],len(meshes2D))

#define material for beam
mats2D = [mats for i in range(len(meshes2D))]

#collect all xs information
xs_info = [meshes2D,mats2D,axial_pos_mesh,orientations]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs, & SOLVE ############
#################################################################

#initialize beam object using 1D mesh and definition of xs's
CantileverBeam = BeamModel(axial_mesh,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
CantileverBeam.plot_xs_orientations()

#applied fixed bc to first endpoint
CantileverBeam.add_clamped_point(p1)

#add distributed load
CantileverBeam.add_dist_load((0,0,-g))

#solve the linear problem
CantileverBeam.solve()

#################################################################
######### POSTPROCESSING, TESTING & VISUALIZATION ############
#################################################################

#shows plot of 1D displacement solution (recovery doesn't need be executed)
CantileverBeam.plot_axial_displacement(warp_factor=10)

#recovers the 3D displacement field over each xs
CantileverBeam.recover_displacement(plot_xss=True)

#plots both 1D and 2D solutions together
CantileverBeam.plot_xs_disp_3D()

#shows plot of stress over cross-section 
CantileverBeam.recover_stress() # currently only axial component sigma11 plotted

#compare with an analytical EB bending solution 
# for this relatively slender beam, this should be nearly identical to the timoshenko solution)
print('Max Deflection for point load (EB analytical analytical solution)')
E = mats['Unobtainium']['MECH_PROPS']['E']
rho = mats['Unobtainium']['DENSITY']
I = H**4/12
q = rho*A*g
print( (-q*L**4)/(8*E*I) )

print('Max vertical deflection of centroid:')
print(CantileverBeam.get_local_disp([p2])[0][2])