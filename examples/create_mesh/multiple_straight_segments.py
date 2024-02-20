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

points = [(0,0,0),(1,2,3),(6,5,4),(7,8,9),(15,20,25)]
num_els = [2,3,4,5]
meshname= 'my_mesh'

domain = ALBATROSS.utils.beam_interval_mesh_3D(points,num_els,meshname)

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, style="wireframe",color='b')
    plotter.add_mesh(grid, style='points',color='r')

    plotter.view_isometric()
#     plotter.show_axes()
    plotter.show_grid()
#     plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
