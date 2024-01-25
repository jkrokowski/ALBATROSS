from dolfinx.io import XDMFFile

from mpi4py import MPI
import numpy as np
from dolfinx import mesh,plot,fem
import pyvista
import ALBATROSS

from ufl import as_vector,as_matrix,sin,cos
#################################################################
########### CONSTRUCT MESH FOR LOCATING BEAM XSs ################
#################################################################

#create mesh
N = 10
W = .1
H = .1
t1 = 0.01
t2 = 0.01
t3 = 0.01
t4 = 0.01

points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [t1,t2,t3,t4]
num_el = 4*[3]
domain = ALBATROSS.utils.create_2D_box(points,thicknesses,num_el,'box_xs')

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    plotter.show_axes()
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#commented out b/c we want to construct a series of meshes
# #analyze cross section
# boxXS = ALBATROSS.cross_section.CrossSection(domain,mats)
# boxXS.getXSStiffnessMatrix()

# #output stiffess matrix
# print(boxXS.K)


# model and mesh parameters
gdim = 3
tdim = 1 

# #create or read in series of 2D meshes
# N = 10
# W = .1
# H = .1

L = 1
meshes2D= 2*[domain]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':10e6,'nu':0.2} ,
                        'DENSITY':2.7e-3}
                        }

#define spanwise locations of XSs with a 1D mesh
p1 = (0,0,0)
p2 = (L,0,0)
ne_2D = len(meshes2D)-1
ne_1D = 10

meshname_axial_pos = 'axial_postion_mesh'
meshname_axial = 'axial_mesh'

#mesh for locating beam cross-sections along beam axis
axial_pos_mesh = ALBATROSS.utils.beam_interval_mesh_3D([p1,p2],[ne_2D],meshname_axial_pos)

#mesh used for 1D analysis
axial_mesh = ALBATROSS.utils.beam_interval_mesh_3D([p1,p2],[ne_1D],meshname_axial)

#note orientation vector does not have to be a unit vector
# orientations = np.tile([-np.sqrt(2),np.sqrt(2),0],len(meshes2D))
# orientations = np.tile([np.sqrt(2),-np.sqrt(2),0],len(meshes2D))
orientations = np.tile([0,1,0],len(meshes2D))
# orientations = np.tile([-1,0,0],len(meshes2D))
# orientations = np.array([0,1,0,0,0,1])
mats2D = [mats for i in range(len(meshes2D))]

xs_info = [meshes2D,mats2D,axial_pos_mesh,orientations]

ExampleBeam = ALBATROSS.beam_model.BeamModel(axial_mesh,xs_info)

#check that 
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
F = 2
ExampleBeam.add_point_load([(0,-F,-F)],[p2])

#add boundary conditions
ExampleBeam.add_clamped_point(p1)
# ExampleBeam.add_clamped_point(p2)

#solve 
ExampleBeam.solve()

#visualize the solution:
ExampleBeam.plot_axial_displacement(warp_factor=5)

ExampleBeam.recover_displacement(plot_xss=True)

ExampleBeam.plot_xs_disp_3D()

ExampleBeam.recover_stress()

print("xs centroid local displacements and rotations")
print(ExampleBeam.get_local_disp([p1,p2]))

