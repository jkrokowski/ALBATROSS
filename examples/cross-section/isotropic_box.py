import os
# # print(os.environ)
os.environ['SCIPY_USE_PROPACK'] = "1"

import ALBATROSS

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile

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
num_el = 4*[4]
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

#analyze cross section
boxXS = ALBATROSS.cross_section.CrossSection(domain,mats)
boxXS.getXSStiffnessMatrix()

#output stiffess matrix
print(boxXS.K)