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

xsName = "beam_crosssection_rib_221_quad"
# xsName = "square_2iso_quads"

fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
print(filePath)

with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain,name="Grid")

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#should we add material tag to 
aluminum7075 = ALBATROSS.material.Material(name='Aluminium7075',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':71e9,'nu':0.33},
                                           density=2795)
nylon_pa12 = ALBATROSS.material.Material(name='Aluminium7075',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':1.7e9,'nu':0.394},
                                           density=930)
mats = [aluminum7075,nylon_pa12]
#initialize cross-seciton object
ribXS = ALBATROSS.cross_section.CrossSection(domain,mats,celltags=ct)

#show me what you got
ribXS.plot_mesh()

#compute the stiffness matrix
ribXS.getXSStiffnessMatrix()

# np.set_printoptions(precision=3)

# #output flexibility matrix
# print('Flexibility matrix:')
# print(squareXS.S)

# #output stiffness matrix
# print('Stiffness matrix:')
# print(squareXS.K)

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








