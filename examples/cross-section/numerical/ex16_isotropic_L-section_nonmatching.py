#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

# np.set_printoptions(precision=3)

# m1,n1 = 54,5
# m2,n2 = 4,45
m1,n1 = 40,4
m2,n2 = 4,40
# m1,n1 = 10,1
# m2,n2 = 1,9

H = 1
W = 1
tf = .1
tw = .2


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2

mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
mesh_1.geometry.x[:, 0] -= H/2 - tw/2

# mesh_1.geometry.x[:, 0] += T2/2

mesh_1.name = 'w'


#PLOT meshes:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'   
plotter = pyvista.Plotter()
def add_mesh(msh):
    topology, cell_types, geom = plot.vtk_mesh(msh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
    plotter.add_mesh(grid,show_edges=True,opacity=0.25)
     
    # Add cell labels
    cell_centers = grid.cell_centers()
    for i, center in enumerate(cell_centers.points):
        plotter.add_point_labels(center, [msh.name+str(i)], font_size=10, point_color='black', text_color='black')
add_mesh(mesh_0)
add_mesh(mesh_1)
# add_mesh(mesh_2)

plotter.show_grid()
plotter.view_xy()
plotter.show()

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

LXS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e8)

LXS_nm.get_xs_stiffness_matrix()

LXS_nm.plot_warping_fxns()

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(LXS_nm.K)

print("Analytical axial stiffness (EA):")
A = tw*W + tf*(H-tw)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(LXS_nm.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ((tw*H**3)/12 
      + (tf**3*(W-tf))/12 
      + tf*(W-tw)*(H/2-tf/2)**2 )
print(E*I1)
print("Computed bending stiffness 1:")
print(LXS_nm.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tf*W**3)/12 
      + (tw**3*(H-tf))/12 
      + tw*(H-tf)*(W/2-tf/2)**2 )
print(E*I2)
print("Computed bending stiffness 2:")
print(LXS_nm.K[5,5])


#compare to conformal approach:
#create mesh
N = 4
H = 1
W= 1
tfh = 0.2
tfw = 0.1

dims = [H,W,tfh,tfw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_L_section(dims,num_el,'L_section')

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
LXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
LXS.plot_mesh()

#compute the stiffness matrix
LXS.get_xs_stiffness_matrix()

LXS.plot_warping_fxns()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(LXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(LXS.K)

print("Analytical axial stiffness (EA):")
A = tfw*(W-tfh) + tfh*(H-tfw) + tfh*tfw
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(LXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ((tfh*H**3)/12 
      + (tfw**3*(W-tfh))/12 
      + tfw*(W-tfh)*(H/2-tfw/2)**2 )
print(E*I1)
print("Computed bending stiffness 1:")
print(LXS.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tfw*W**3)/12 
      + (tfh**3*(H-tfw))/12 
      + tfh*(H-tfw)*(W/2-tfh/2)**2 )
print(E*I2)
print("Computed bending stiffness 2:")
print(LXS.K[5,5])



#compute difference between matrix entries of beam constituitive matrix
abs_diff = LXS_nm.K - LXS.K
rel_diff = abs_diff/LXS.K

print("Total difference:")
print(abs_diff)

print("Relative Difference:")
print(rel_diff)

print("maximum relative difference:")
print(np.max(np.abs(np.diag(rel_diff))))