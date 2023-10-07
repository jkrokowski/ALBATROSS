#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

from dolfinx.io import XDMFFile

from mpi4py import MPI
import numpy as np
from dolfinx import mesh,plot,fem
import pyvista

from FROOT_BAT import cross_section,beam_model,utils

from ufl import as_vector,as_matrix,sin,cos
#################################################################
########### CONSTRUCT MESH FOR LOCATING BEAM XCs ################
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 3
W = .1
H = .1
W1 = 0.75*W
H1 = 0.5*H
mesh2d_0 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
with XDMFFile(mesh2d_0.comm, 'mesh2d_0', "w") as file:
        file.write_mesh(mesh2d_0)
mesh2d_1 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W1, H1]]),[N,N], cell_type=mesh.CellType.quadrilateral)

meshes2D = [mesh2d_0,mesh2d_1]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':10000,'nu':0.2} }
                        }

#define spanwise locations of XCs with a 1D mesh
p1 = (0,0,0)
p2 = (5,0,0)
ne_2D = len(meshes2D)-1
ne_1D = 10

meshname1D_2D = 'square_tapered_beam_1D_2D'
meshname1D_1D = 'square_tapered_beam_1D_1D'

#mesh for locating beam cross-sections along beam axis
mesh1D_2D = utils.beam_interval_mesh_3D([p1,p2],[ne_2D],meshname1D_2D)

#mesh used for 1D analysis
mesh1D_1D = utils.beam_interval_mesh_3D([p1,p2],[ne_1D],meshname1D_1D)

#get fxn for XC properties at any point along beam axis
mats2D = [mats for i in range(len(meshes2D))]
xcdata=[meshes2D,mats2D]
xcinfo = cross_section.get_xc_info([mesh1D_2D,xcdata],mesh1D_1D)
#intialize 1D analysis model
square_tapered_beam = beam_model.LinearTimoshenko(mesh1D_1D,xcinfo)
square_tapered_beam.elastic_energy()

#API for adding loads
# rho = Constant(mesh1D_1D,2.7e-3)
# g = Constant(mesh1D_1D,9.81)
rho = 2.7e-3
g = 9.81
A = 0.01

square_tapered_beam.add_body_force((0,0,-A*rho*g))
#TODO: add point load
#  square_tapered_beam.addPointLoad()
#  square_tapered_beam.addPointMoment()
#  square_tapered_beam.addDistributedLoad((0,0,-rho*g),'where')

#API for adding loads
square_tapered_beam.add_clamped_point(p1)
# square_tapered_beam.addClampedPointTopo(0)
# TODO: add pinned,rotation and trans restrictions
# square_tapered_beam.addPinnedPoint()
# square_tapered_beam.restrictTranslation()
# square_tapered_beam.restrictRotation()

#find displacement solution for beam axis
square_tapered_beam.solve()

# [u_values,theta_values] =square_tapered_beam.get_global_disp(p2)
[u_values,theta_values] =square_tapered_beam.get_local_disp(p2)

print(u_values)
print(theta_values)

#get displacement at tip
V = fem.VectorFunctionSpace(mesh2d_1,('CG',1),dim=3)
xcdisp = fem.Function(V)

# print(int(xcdisp.x.array.shape[0]/3))
numdofs = int(xcdisp.x.array.shape[0]/3)
# numdofs = mesh2d_1.topology.index_map(mesh2d_1.topology.dim).size_local
# print(numdofs)

# print(xcdisp.x.array)
xc = W1/2 
yc = H1/2
def apply_rotation(x):
    alpha = theta_values[1]
    beta = theta_values[2]
    gamma = theta_values[0]
    Rx = np.array([[1,         0,         0],
                    [0,cos(alpha),-sin(alpha)],
                    [0,sin(alpha),cos(alpha)]])
    # rotation about Y-axis
    Ry = np.array([[cos(beta), 0,sin(beta)],
                    [0,         1,        0],
                    [-sin(beta),0,cos(beta)]])
    #rotation about Z-axis
    Rz = np.array([[cos(gamma),-sin(gamma),0],
                    [sin(gamma),cos(gamma), 0],
                    [0,         0,          1]])
    
    # #3D rotation matrix
    R = Rz@Ry@Rx
    vec = R@np.array([x[0]-xc,x[1]-yc,x[2]])    
    return 20*np.array([vec[0]-x[0]+xc,vec[1]-x[1]+yc,vec[2]-x[2]])
xcdisp.interpolate(apply_rotation)
#need beam axis x and y location as output along with area and warping fxns

xcdisp.vector.array += np.tile(np.array([u_values[1],u_values[2],u_values[0]]),numdofs)
print(xcdisp.x.array)

xcdisp.vector.destroy()

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
tdim = mesh2d_1.topology.dim
topology, cell_types, geom = plot.create_vtk_mesh(mesh2d_1, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geom).rotate_z(90).rotate_y(90)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# grid.rotate_z(90).rotate_y(90)
# plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# have to be careful about how displacement data is populated into grid before or after rotations for visualization
grid.point_data["u"] = xcdisp.x.array.reshape((geom.shape[0],3))

warped = grid.warp_by_vector("u", factor=1)
actor_1 = plotter.add_mesh(warped, show_edges=True)
plotter.show_axes()
# if add_nodes==True:
#     plotter.add_mesh(grid, style='points')
plotter.view_isometric()
if not pyvista.OFF_SCREEN:
    plotter.show()

# print(xcdisp.x.array)
v = square_tapered_beam.uh.sub(0)
v.name= "Displacement"
# File('beam-disp.pvd') << v
#save rotations
theta = square_tapered_beam.uh.sub(1)
theta.name ="Rotation"

with XDMFFile(MPI.COMM_WORLD, "output/output.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh1D_1D)
    xdmf.write_function(v)
    xdmf.write_function(theta)

# #TODO: get displacement field for full 3D beam structure
# square_tapered_beam.get3DModel()
# square_tapered_beam.plot3DModel()
# 
# THIS SHOULD COME FROM THE DISP SOLN IN XC ANALYSIS
# NEED TO SAVE OUT UBAR,HAT,TILDE,BREVE FOR 3DDISP RECONSTRUCTION
# square_tapered_beam.get3DDisp()
# square_tapered_beam.plot3DDisp()

# #TODO: get stress field for full 3D beam structure
# square_tapered_beam.get3DStress()
# square_tapered_beam.plot3DStress()

#visualize with pyvista:
if True:
    tdim = mesh1D_1D.topology.dim
    topology, cell_types, geom = plot.create_vtk_mesh(mesh1D_1D,tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
    plotter = pyvista.Plotter()

    grid.point_data["u"] = v.collapse().x.array.reshape((geom.shape[0],3))
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=5)
    actor_1 = plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plot.screenshot("beam_mesh.png")