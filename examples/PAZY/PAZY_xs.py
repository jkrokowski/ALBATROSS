#example of cross-sectional analysis of an multimaterial isotropic square:

from mpi4py import MPI
from dolfinx.mesh import meshtags,create_mesh
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh
import ufl

import ALBATROSS

import os,sys
import numpy as np

import meshio

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

#read in mesh
# xsName = "beam_crosssection_rib_221_quad"
# xsName = "square_2iso_quads"
xsName = "beam_crosssection_2_95_quad"
fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    in_mesh = xdmf.read_mesh(name="Grid")
    gdim = 2
    shape = in_mesh.ufl_cell().cellname()
    degree = 1
    cell = ufl.Cell(shape)
    adj_list =in_mesh.topology.connectivity(2,0)
    cells = adj_list.array.reshape((adj_list.num_nodes,adj_list.links(0).shape[0]))
    # points = in_mesh.geometry.x[:,:2]
    points = np.stack([in_mesh.geometry.x[:,0],in_mesh.geometry.x[:,2]],axis=1)
    domain= create_mesh(MPI.COMM_WORLD,
                        cells,
                        points,
                        ufl.Mesh(ufl.VectorElement("Lagrange",cell,degree)) )

    ct = xdmf.read_meshtags(in_mesh, name="Grid")   

# domain = create_mesh(MPI.COMM_WORLD,cells)
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
print(domain.topology.dim)
print(domain.geometry.dim)
# #these mesh coords need to be in xy coords, not xz
# def xz_to_xy(domain):
#     return np.stack([domain.geometry.x[:,0],domain.geometry.x[:,1],np.zeros_like(domain.geometry.x[:,1])],axis=1)

# domain.geometry.x[:,:] = xz_to_xy(domain)

# domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

ct.values[:]=ct.values-np.min(ct.values)

# print(domain.topology.dim)

#GET material markers
tdim = 2
# #right
# right_marker=0
# right_facets = ct.find(right_marker)
# right_mt = meshtags(domain, tdim, right_facets, right_marker)
# #left
# left_marker=1
# left_facets = ct.find(left_marker)
# left_mt = meshtags(domain, tdim, left_facets, left_marker)

#plot mesh:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=10000,
                                           celltag=1)
adamantium = ALBATROSS.material.Material(name='adamantium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10,'nu':0.2},
                                           density=5000,
                                           celltag=0)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium,adamantium],celltags=ct)

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.getXSStiffnessMatrix()

np.set_printoptions(precision=5,suppress=True)

#output flexibility matrix
print('Flexibility matrix:')
print(squareXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)
