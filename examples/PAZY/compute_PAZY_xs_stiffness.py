#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS
from dolfinx.io import XDMFFile
from dolfinx import plot,mesh
from mpi4py import MPI
import pyvista

import os,sys
import numpy as np

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

# xsName = "beam_crosssection_rib_221_quad"
xsName = "beam_crosssection_2_95_quad"
# xsName = "square_2iso_quads"

fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
print(filePath)

with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain,name="Grid")

#these mesh coords need to be in xy coords, not xz
def xz_to_xy(domain):
    return np.stack([domain.geometry.x[:,0],domain.geometry.x[:,2],np.zeros_like(domain.geometry.x[:,1])],axis=1)

domain.geometry.x[:,:] = xz_to_xy(domain)

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

ct.values[:]=ct.values-np.min(ct.values)

#should we add material tag to 
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
# mats = [aluminum7075]

#initialize cross-section object
ribXS = ALBATROSS.cross_section.CrossSection(domain,mats,celltags=ct)
# ribXS = ALBATROSS.cross_section.CrossSection(domain,mats)

#show me what you got
ribXS.plot_mesh()

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()


#compute the stiffness matrix
ribXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(ribXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(ribXS.K)

# print("Analytical axial stiffness (EA):")
# E = mats['Unobtainium']['MECH_PROPS']['E']
# A = W*H
# print(E*A)
# print("Computed Axial Stiffness:")
# print(squareXS.K[0,0])

# print("Analytical Bending stiffness (EI):")
# I = (W*H**3)/12
# print(E*I)
# print("Computed bending stiffness:")
# print(squareXS.K[4,4])








