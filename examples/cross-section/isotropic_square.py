from FROOT_BAT import cross_section

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile

# Create 2d mesh and define function space
N = 3
W = .1
H = .1

# domain = mesh.create_unit_square(MPI.COMM_WORLD,N,N, mesh.CellType.quadrilateral)
domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)

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

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100,'nu':0.2} }
                        }

squareXC = cross_section.CrossSection(domain,mats)
squareXC.getXCStiffnessMatrix()
print(squareXC.K)







