#simple example of cross-sectional analysis of an isotropic symmetric I-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

np.set_printoptions(precision=3)

# m1,n1 = 54,5
# m2,n2 = 4,45
# m3,n3 = 54,5
m1,n1 = 50,5
m2,n2 = 5,50
m3,n3 = 50,5

H = 0.1
L = 0.1
T1 = .01
T2 = .01
T3 = .01


# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_0.geometry.x[:, 1] += H/2 - T1/2

# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= T2
mesh_1.geometry.x[:, 1] *= H
# mesh_1.geometry.x[:, 0] += T2/2

# mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3,cell_type=mesh.CellType.quadrilateral)
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

IXS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=5e4)

IXS_nm.get_xs_stiffness_matrix()

IXS_nm.plot_warping_fxns()

np.set_printoptions(precision=3)

#output stiffness matrix
print('coupled Stiffness matrix:')
print(IXS_nm.K)

print("Analytical axial stiffness (EA):")
A = L*H - (L-T1)*(H-2*T2)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(IXS_nm.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (L*H**3)/12 -((L-T2)*(H-2*T1)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(IXS_nm.K[4,4])


#compare to conformal approach:
#create mesh
N = 10
H = .1
W= .1
tf = 0.01
tw = 0.01

dims = [H,W,tf,tw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_I_section(dims,num_el,'I_section')

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
IXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
IXS.plot_mesh()

#compute the stiffness matrix
IXS.get_xs_stiffness_matrix()

print('Conformal Stiffness matrix:')
print(IXS.K)

IXS.plot_warping_fxns()

#compute difference between matrix entries of beam constituitive matrix
abs_diff = IXS_nm.K - IXS.K
rel_diff = abs_diff/IXS.K

print("Total difference:")
print(abs_diff)

print("Relative Difference:")
print(rel_diff)

