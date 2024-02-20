from dolfinx.io import XDMFFile

from mpi4py import MPI
import numpy as np
from dolfinx import mesh,plot,fem
import pyvista

from ALBATROSS import beam_model,cross_section,utils,axial
from ALBATROSS.beam_model import BeamModel

from ufl import as_vector,as_matrix,sin,cos
#################################################################
########### CONSTRUCT MESH FOR LOCATING BEAM XCs ################
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 10
W = .1
H = .1
# W1 = 0.75*W
# H1 = 0.5*H
W1= W
H1= H

L = 20
mesh2d_0 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
# with XDMFFile(mesh2d_0.comm, 'mesh2d_0', "w") as file:
#         file.write_mesh(mesh2d_0)
mesh2d_1 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W1, H1]]),[N,N], cell_type=mesh.CellType.quadrilateral)

meshes2D = [mesh2d_0,mesh2d_1]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':10e6,'nu':0.2} ,
                        'DENSITY':2.7e-3}
                        }

#define spanwise locations of XCs with a 1D mesh
p1 = (0,0,0)
p2 = (L,0,0)
ne_2D = len(meshes2D)-1
ne_1D = 100

meshname_axial_pos = 'axial_postion_mesh'
meshname_axial = 'axial_mesh'

#mesh for locating beam cross-sections along beam axis
axial_pos_mesh = utils.beam_interval_mesh_3D([p1,p2],[ne_2D],meshname_axial_pos)

#mesh used for 1D analysis
axial_mesh = utils.beam_interval_mesh_3D([p1,p2],[ne_1D],meshname_axial)

#note orientation vector does not have to be a unit fector
# orientations = np.tile([- np.sqrt(2),np.sqrt(2),0],len(meshes2D))
# orientations = np.tile([np.sqrt(2),-np.sqrt(2),0],len(meshes2D))
orientations = np.tile([0,1,0],len(meshes2D))
# orientations = np.tile([-1,0,0],len(meshes2D))
# orientations = np.array([0,1,0,0,0,1])
mats2D = [mats for i in range(len(meshes2D))]

xc_info = [meshes2D,mats2D,axial_pos_mesh,orientations]

ExampleBeam = BeamModel(axial_mesh,xc_info)

ExampleBeam.plot_xs_orientations()
rho = 2.7e-3
g = 9.81
A = 0.01

#add loads
# ExampleBeam.add_dist_load((0,0,-g))
#SUBLTY here:
'''
the entire system must be assembled so that the rhs can be modfied

likely what this will do under the hood is cache the loads and 
the points to be applied at the time of the solve as the assembly 
of the system should not happen prior to the solution of the system
for most users.'''
F = .01
ExampleBeam.add_point_load([(0,-F,-F)],[p2])

#add boundary conditions
ExampleBeam.add_clamped_point(p1)
# ExampleBeam.add_clamped_point(p2)

#solve 
ExampleBeam.solve()

ExampleBeam.plot_axial_displacement(warp_factor=10)

ExampleBeam.recover_displacement(plot_xss=True)

# ExampleBeam.plot_xc_disp_3D()

ExampleBeam.recover_stress()
print("xc centroid local displacements and rotations")
print(ExampleBeam.get_local_disp([p1,p2]))
# print("xc centroid global displacments and rotations")
# print(ExampleBeam.get_global_disp([p1,p2]))

print('Max Deflection for point load (EB analytical)')
E = mats['Unobtainium']['MECH_PROPS']['E']
rho = mats['Unobtainium']['DENSITY']
I = H**4/12
print( (-F*L**3)/(3*E*I) )

#TODO:
# ExampleBeam.get_max_stress()
# ExampleBeam.plot_stress_over_xc()
