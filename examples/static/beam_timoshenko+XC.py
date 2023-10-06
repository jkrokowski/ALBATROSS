#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

from dolfinx.io import XDMFFile

from mpi4py import MPI
import numpy as np
from dolfinx import mesh,plot
import pyvista

from FROOT_BAT import geometry,cross_section,beam_model

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
mesh2d_0 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
with XDMFFile(mesh2d_0.comm, 'mesh2d_0', "w") as file:
        file.write_mesh(mesh2d_0)
mesh2d_1 = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[0.75*W, 0.5*H]]),[N,N], cell_type=mesh.CellType.quadrilateral)

meshes2D = [mesh2d_0,mesh2d_1]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':10000,'nu':0.2} }
                        }

#define spanwise locations of XCs with a 1D mesh
p1 = (0,0,0)
p2 = (5,0,0)
ne_2D = len(meshes2D)-1
ne_1D = 10000

meshname1D_2D = 'square_tapered_beam_1D_2D'
meshname1D_1D = 'square_tapered_beam_1D_1D'

#mesh for locating beam cross-sections along beam axis
mesh1D_2D = geometry.beamIntervalMesh3D([p1,p2],[ne_2D],meshname1D_2D)

#mesh used for 1D analysis
mesh1D_1D = geometry.beamIntervalMesh3D([p1,p2],[ne_1D],meshname1D_1D)

#get fxn for XC properties at any point along beam axis
mats2D = [mats for i in range(len(meshes2D))]
xcdata=[meshes2D,mats2D]
xcinfo = cross_section.defineXCsFor1D([mesh1D_2D,xcdata],mesh1D_1D)
#intialize 1D analysis model
square_tapered_beam = beam_model.LinearTimoshenko(mesh1D_1D,xcinfo)
square_tapered_beam.elasticEnergy()

#API for adding loads
# rho = Constant(mesh1D_1D,2.7e-3)
# g = Constant(mesh1D_1D,9.81)
rho = 2.7e-3
g = 9.81
A = 0.01

square_tapered_beam.addBodyForce((0,0,-A*rho*g))
#TODO: add point load
#  square_tapered_beam.addPointLoad()
#  square_tapered_beam.addPointMoment()
#  square_tapered_beam.addDistributedLoad((0,0,-rho*g),'where')

#API for adding loads
square_tapered_beam.addClampedPoint(p1)
# square_tapered_beam.addClampedPointTopo(0)
# TODO: add pinned,rotation and trans restrictions
# square_tapered_beam.addPinnedPoint()
# square_tapered_beam.restrictTranslation()
# square_tapered_beam.restrictRotation()

#find displacement solution for beam axis
square_tapered_beam.solve()

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
    topology, cell_types, geom = plot.create_vtk_mesh(mesh1D_1D,tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
    plotter = pyvista.Plotter()

    grid.point_data["u"] = v.collapse().x.array.reshape((geom.shape[0],3))
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plot.screenshot("beam_mesh.png")
