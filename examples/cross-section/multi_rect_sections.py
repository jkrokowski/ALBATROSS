'''
Demonstration of populated a fxn space with beam  a beam with a varying cross-section

First, a 1D mesh based with nodes located at the point the cross-sections are define
Then, a series of 2D meshes are analyzed and the stiffness matrices are used to populate
    a 2-tensor (6x6) tensor fxn space
Finally, a finer mesh is constructed (e.g. to be used in the 1D analysis) and the 
    course mesh is interpolated into the fine mesh 2-tensor (6x6) fxn space. This allows
    ufl expresssion using the stiffness matrix at any point to be used.
'''
from FROOT_BAT import cross_section,geometry,utils

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np

from dolfinx import fem,mesh

#create mesh
N = 3
W = .1
H = .1
H1 = 0.05
L = 5
nx = 1 #number of elements
pts = [(0,0,0),(L,0,0)]
filename = "output/test_beam.xdmf"
meshname = 'test_beam'

mesh1D = geometry.beamIntervalMesh3D(pts,[nx],meshname)

if True:
    utils.plot_xdmf_mesh(mesh1D,add_nodes=True)

mesh2d_0 = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
mesh2d_1 = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H1]]),[N,N], cell_type=mesh.CellType.quadrilateral)

meshes2d = [mesh2d_0,mesh2d_1]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100,'nu':0.2} }
                        }

K_list = []
for mesh2d in meshes2d:
    #analyze cross section
    squareXC = cross_section.CrossSection(mesh2d,mats)
    squareXC.getXCStiffnessMatrix()

    #output stiffess matrix
    K_list.append(squareXC.K)

    if True:
        utils.plot_xdmf_mesh(mesh2d)

K1 = fem.TensorFunctionSpace(mesh1D,('CG',1),shape=(6,6),symmetry=True)

k1 = fem.Function(K1)

def get_flat_sym_stiff(K_mat):
    K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
    return K_flat

K_entries = np.concatenate([get_flat_sym_stiff(K_list[i]) for i in range(2)])

k1.vector.array = K_entries

meshname = 'test_beam1'

mesh1D_fine = geometry.beamIntervalMesh3D(pts,[nx*10],meshname)

if True:
    utils.plot_xdmf_mesh(mesh1D_fine,add_nodes=True)

K2 = fem.TensorFunctionSpace(mesh1D_fine,('CG',1),shape=(6,6),symmetry=True)

k2 = fem.Function(K2)

k2.interpolate(k1)