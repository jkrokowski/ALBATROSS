#example of cross-sectional analysis of an multimaterial isotropic square:

from mpi4py import MPI
from dolfinx.mesh import meshtags
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh

import ALBATROSS

import os
import sys

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

#read in mesh
xsName = "square_2iso_quads"
fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,"..","..","mesh",fileName)
with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#GET material markers
tdim = 2
#right
right_marker=0
right_facets = ct.find(right_marker)
right_mt = meshtags(domain, tdim, right_facets, right_marker)
#left
left_marker=1
left_facets = ct.find(left_marker)
left_mt = meshtags(domain, tdim, left_facets, left_marker)

#plot mesh:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Material"] = ct.values
sargs = dict(
        title_font_size=20,
        label_font_size=16,
        shadow=True,
        n_labels=2,
        italic=True,
        fmt="%1.0f",
        font_family="arial",
        # width=2,
        # color = [1,1,1]
    )
annotations = {1:"adamantium",0:"unobtainium"}
p.add_mesh(grid, show_edges=True,scalar_bar_args=sargs,annotations=annotations)
p.view_xy()
p.show()

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=10000,
                                           celltag=0)
adamantium = ALBATROSS.material.Material(name='adamantium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10,'nu':0.2},
                                           density=5000,
                                           celltag=1)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium,adamantium],celltags=ct)

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()

np.set_printoptions(precision=10,suppress=True)

#output flexibility matrix
print('Flexibility matrix:')
print(squareXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)
