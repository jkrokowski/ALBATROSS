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
N = 40
W = .1
H = .1

# domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[-W/2,-H/2],[W/2, H/2]]),[N,N], cell_type=mesh.CellType.quadrilateral)

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }
#analyze cross section
squareXS = ALBATROSS.cross_section.CrossSection(domain,mats)
squareXS.getXSStiffnessMatrix()

#output stiffess matrix
print(squareXS.K)

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
    if not pyvista.OFF_SCREEN:
        plotter.show()







