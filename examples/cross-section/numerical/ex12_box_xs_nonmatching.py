from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

np.set_printoptions(precision=3)

# m1,n1 = 36,3
# m2,n2 = 4,45
# m3,n3 = 54,5
# m4,n4 = 6,65
N = 5
offset=4

m1,n1 = N*10+offset,N
m2,n2 = N,N*10+offset
m3,n3 = N*10+offset,N
m4,n4 = N,N*10+offset
H = 1
L = 1
T1 = .1
T2 = .1
T3 = .1
T4 = .1


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_0.geometry.x[:, 1] += H/2 - T1/2

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= T2
mesh_1.geometry.x[:, 1] *= H
mesh_1.geometry.x[:, 0] += L/2 - T2/2

mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3,cell_type=mesh.CellType.quadrilateral)
# mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3)
mesh_2.geometry.x[:, :2] -= .5
mesh_2.geometry.x[:, 1] *= T1
mesh_2.geometry.x[:, 0] *= L
mesh_2.geometry.x[:, 1] -= H/2 - T1/2

mesh_3 = mesh.create_unit_square(MPI.COMM_WORLD, m4, n4,cell_type=mesh.CellType.quadrilateral)
# mesh_3 = mesh.create_unit_square(MPI.COMM_WORLD, m4, n4)
mesh_3.geometry.x[:, :2] -= .5
mesh_3.geometry.x[:, 0] *= T2
mesh_3.geometry.x[:, 1] *= H
mesh_3.geometry.x[:, 0] -= L/2 - T2/2

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
add_mesh(mesh_3)
plotter.show_grid()
plotter.view_xy()
plotter.show()

meshes= [mesh_0,mesh_1,mesh_2,mesh_3]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

boxXS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e3)

boxXS_nm.get_xs_stiffness_matrix(correction=None)

boxXS_nm.plot_warping_fxns()

#output stiffness matrix
print('Stiffness matrix:')
print(boxXS_nm.K)

print("Analytical axial stiffness (EA):")
E = unobtainium.E
A = L*H - (H-T1-T3)*(L-T2-T4)
print(E*A)
print("Computed Axial Stiffness:")
print(boxXS_nm.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (L*H**3)/12 - ((L-T2-T4)*(H-T1-T3)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(boxXS_nm.K[4,4])


#conformal approach:
#create mesh
W = 1
H = 1
t1 = 0.1
t2 = 0.1
t3 = 0.1
t4 = 0.1

points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [t1,t2,t3,t4]
num_el = 4*[N] #number of elements through each wall thickness
domain = ALBATROSS.mesh.create_hollow_box(points,thicknesses,num_el,'box_xs')

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
boxXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
boxXS.plot_mesh()

#compute the stiffness matrix
boxXS.get_xs_stiffness_matrix()

boxXS.plot_warping_fxns()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(boxXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(boxXS.K)

print("Analytical axial stiffness (EA):")
E = unobtainium.E
A = W*H - (H-t1-t3)*(W-t2-t4)
print(E*A)
print("Computed Axial Stiffness:")
print(boxXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12 - ((W-t2-t4)*(H-t1-t3)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(boxXS.K[4,4])

#compute difference between matrix entries of beam constituitive matrix
abs_diff = boxXS_nm.K - boxXS.K
rel_diff = abs_diff/boxXS.K

print("Total difference:")
print(abs_diff)

print("Relative Difference:")
print(rel_diff)

print("maximum relative difference:")
print(np.max(np.abs(np.diag(rel_diff))))