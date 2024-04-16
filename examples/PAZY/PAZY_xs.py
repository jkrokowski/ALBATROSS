#example of cross-sectional analysis of an multimaterial isotropic square:
from mpi4py import MPI
from dolfinx.mesh import meshtags,create_mesh
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh
import ufl

import ALBATROSS

import os
import sys

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

#read in mesh
xsName = "beam_crosssection_rib_221_quad"
# xsName = "square_2iso_quads"
# xsName = "beam_crosssection_2_95_quad"

fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #read in mesh and convert to a topological AND geometrically 2D mesh
    in_mesh = xdmf.read_mesh(name="Grid")
    gdim = 2
    shape = in_mesh.ufl_cell().cellname()
    degree = 1
    cell = ufl.Cell(shape)
    adj_list =in_mesh.topology.connectivity(2,0)
    cells = adj_list.array.reshape((adj_list.num_nodes,adj_list.links(0).shape[0]))
    chord = .1 #m
    x_points = in_mesh.geometry.x[:,0] - .25*chord #adjust x location of mesh
    print(np.max(x_points))
    print(np.min(x_points))
    y_points = in_mesh.geometry.x[:,2]
    print(np.max(y_points))
    print(np.min(y_points)) 
    points = np.stack([x_points,y_points],axis=1)
    domain= create_mesh(MPI.COMM_WORLD,
                        cells,
                        points,
                        ufl.Mesh(ufl.VectorElement("Lagrange",cell,degree)) )
    #read celltags
    ct = xdmf.read_meshtags(in_mesh, name="Grid")   

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
# print(domain.topology.dim)
# print(domain.geometry.dim)

ct.values[:]=ct.values-np.min(ct.values)

#plot mesh:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.show_grid()
# marker = pyvista.create_axes_marker(ambient=0.1,tip_length=.05)
# p.add_actor(marker)
# p.add_axes_at_origin()
p.view_xy()
p.show()

aluminum7075 = ALBATROSS.material.Material(name='Aluminium7075',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':71e9,'nu':0.33},
                                           density=2795,
                                           celltag=0)
nylon_pa12 = ALBATROSS.material.Material(name='NylonPA12',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':1.7e9,'nu':0.394},
                                           density=930,
                                           celltag=1)
mats = [aluminum7075,nylon_pa12]
#initialize cross-section object
ribXS = ALBATROSS.cross_section.CrossSection(domain,mats,celltags=ct)

#show me what you got
ribXS.plot_mesh()

#compute the stiffness matrix
ribXS.getXSStiffnessMatrix()

np.set_printoptions(precision=5,suppress=True)

#output flexibility matrix
print('Flexibility matrix:')
print(ribXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(ribXS.K)
