#create gmsh readable xcs from PEGASUS wing data
import scipy.io
import os
path = os.getcwd()
print(path)
mat = scipy.io.loadmat(path+'/PAV_xcs/PavCs11.mat')

elems = mat['vabs_2d_mesh_elements']
nodes = mat['vabs_2d_mesh_nodes']
print('Number of nodes:')
print(len(nodes))
print('Number of Elements:')
print(len(elems))
elems -=1

import meshio

cells = {'triangle':elems[:,0:3]}
meshio.write_points_cells('file.xdmf',nodes,cells,file_format='xdmf')

from dolfinx.io import XDMFFile
from mpi4py import MPI

with XDMFFile(MPI.COMM_WORLD, 'file.xdmf', "r") as xdmf:
    domain = xdmf.read_mesh(name='Grid')
    # cells = 
from dolfinx.fem import FunctionSpace

domain.topology.create_connectivity(domain.topology.dim-1, 0)

# S = FunctionSpace(mesh,('CG',1))

import pyvista 
import numpy as np
from dolfinx.plot import create_vtk_mesh
plotter = pyvista.Plotter()
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))

grid = pyvista.UnstructuredGrid(topology, cell_types, x)
plotter.add_mesh(grid,show_edges=True)
plotter.show()

from FROOT_BAT.cross_section import CrossSection
mats = {'Aluminum':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':7.31e10,'nu':0.33} ,
                        'DENSITY':2768}
                        }

XC = CrossSection(domain,mats)
import time
t0=time.time()
XC.getXCStiffnessMatrix()
t1=time.time()
#get vabs data:
K_vabs = mat['cs_K']
S_vabs = mat['cs_F']
print("time to compute stiffness of xc:")
print(t1-t0)
float_formatter= '{:.4e}'.format
np.set_printoptions(formatter={'float_kind':float_formatter})
print("stiffness matrix from FROOTBAT:")
print(XC.K)
print("stiffness matrix from VABS:")
print(K_vabs)
print("abs difference between VABS and FROOTBAT")
print(XC.K-K_vabs)
print('percentage difference between VABS and FROOT_BAT')
float_formatter= '{:.1f}'.format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(100*((XC.K-K_vabs)/K_vabs))

print('------')

float_formatter= '{:.4e}'.format
np.set_printoptions(formatter={'float_kind':float_formatter})
print("flexibility matrix from FROOTBAT:")
print(XC.S)
print("Flexibility matrix from VABS:")
print(S_vabs)
print("abs difference between VABS and FROOTBAT")
print(XC.S-S_vabs)
print('percentage difference between VABS and FROOT_BAT')
float_formatter= '{:.1f}'.format
np.set_printoptions(formatter={'float_kind':float_formatter})
print(100*((XC.S-S_vabs)/S_vabs))
