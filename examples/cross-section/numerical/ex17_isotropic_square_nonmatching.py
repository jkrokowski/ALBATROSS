#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

# np.set_printoptions(precision=3)

N = 15
offset = 1

m1,n1 = N+offset,N*2+offset
m2,n2 = N-offset,N*2-offset

H = 1
W = 1
t_overlap = .05

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= H
mesh_0.geometry.x[:, 0] *= W/2+t_overlap
mesh_0.geometry.x[:, 0] += (W/2-t_overlap)/2

mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= H
mesh_1.geometry.x[:, 0] *= W/2+t_overlap
mesh_1.geometry.x[:, 0] -= (W/2-t_overlap)/2

mesh_1.name = 'w'

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

XS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e3)

XS_nm.plot_meshes()

XS_nm.get_xs_stiffness_matrix(correction=None)

XS_nm.plot_warping_fxns()

# for i in range(3):
#     for j in range(3):
#         TXS_nm.plot_warping_strains(component=(i,j))

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(XS_nm.K)

points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)