#simple example of cross-sectional analysis of an isotropic symmetric I-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

np.set_printoptions(precision=3)

m1,n1 = 36,3
m2,n2 = 4,45
m3,n3 = 54,5
m4,n4 = 6,65
H = 0.1
L = 0.1
T1 = .01
T2 = .01
T3 = .01


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_0.geometry.x[:, 1] += H/2 - T1/2

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= T2
mesh_1.geometry.x[:, 1] *= H
# mesh_1.geometry.x[:, 0] += T2/2

mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3)
mesh_2.geometry.x[:, :2] -= .5
mesh_2.geometry.x[:, 1] *= T1
mesh_2.geometry.x[:, 0] *= L
mesh_2.geometry.x[:, 1] -= H/2 - T1/2

#PLOT meshes:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'   
plotter = pyvista.Plotter()
def add_mesh(msh):
    topology, cell_types, geom = plot.vtk_mesh(msh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
    plotter.add_mesh(grid,show_edges=True,opacity=0.25)
add_mesh(mesh_0)
add_mesh(mesh_1)
add_mesh(mesh_2)

plotter.show_grid()
plotter.view_xy()
plotter.show()

meshes= [mesh_0,mesh_1,mesh_2]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

coupled_cross_section = ALBATROSS.cross_section.CoupledXSProblem(XSs)

coupled_cross_section.get_xs_stiffness_matrix()


np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(coupled_cross_section.K)

print("Analytical axial stiffness (EA):")
A = L*H - (L-T1)*(H-2*T2)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(coupled_cross_section.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (L*H**3)/12 -((L-T2)*(H-2*T1)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(coupled_cross_section.K[4,4])
