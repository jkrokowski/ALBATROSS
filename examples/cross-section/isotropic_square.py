import os
# # print(os.environ)
os.environ['SCIPY_USE_PROPACK'] = "1"
from FROOT_BAT import cross_section

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile

#create mesh
N = 5
W = .1
H = .1

domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':7.31E10,'nu':0.40576923076923066} ,
                        'DENSITY':2.7e3}
        }
#analyze cross section
squareXC = cross_section.CrossSection(domain,mats)
squareXC.getXCStiffnessMatrix()

#output stiffess matrix
print(squareXC.K)

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
    if not pyvista.OFF_SCREEN:
        plotter.show()







