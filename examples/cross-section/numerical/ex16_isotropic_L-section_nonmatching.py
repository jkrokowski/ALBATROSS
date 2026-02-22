#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np
from dolfinx.io import XDMFFile


# np.set_printoptions(precision=3)

# m1,n1 = 54,5
# m2,n2 = 4,45
# m1,n1 = 40,4
# m2,n2 = 4,40
# m1,n1 = 10,1
# m2,n2 = 1,9

N = 6
offset = 3

m1,n1 = N*10+offset,N
m2,n2 = N,N*10+offset

H = .1
W = .1
tf = .01
tw = .01


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.triangle)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2

mesh_0.name = f'L_f_N{N}'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.triangle)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
mesh_1.geometry.x[:, 0] -= H/2 - tw/2

# mesh_1.geometry.x[:, 0] += T2/2

mesh_1.name = f'L_w_N{N}'


# #PLOT meshes:
# pyvista.global_theme.background = [255, 255, 255, 255]
# pyvista.global_theme.font.color = 'black'   
# plotter = pyvista.Plotter()
# def add_mesh(msh):
#     topology, cell_types, geom = plot.vtk_mesh(msh, 2)
#     grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
#     plotter.add_mesh(grid,show_edges=True,opacity=0.25)
     
#     # Add cell labels
#     cell_centers = grid.cell_centers()
#     for i, center in enumerate(cell_centers.points):
#         plotter.add_point_labels(center, [msh.name+str(i)], font_size=10, point_color='black', text_color='black')
# add_mesh(mesh_0)
# add_mesh(mesh_1)
# # add_mesh(mesh_2)

# plotter.show_grid()
# plotter.view_xy()
# plotter.show()

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':70e9,'nu':0.33},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

val = 1e4
LXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen_u=1e-5,pen_t=1e8)

LXS_nm.plot_meshes()

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
np.set_printoptions(precision=3)
print(LXS_nm)
LXS_conformal_K = np.load("L_section_K_n_6_H0.1_W0.1.npy")

diff=LXS_nm.K-LXS_conformal_K
rel_diff = (LXS_nm.K-LXS_conformal_K)/LXS_conformal_K
abs_diff_diag = np.diag(diff)
rel_diff_diag = abs_diff_diag/np.diag(LXS_conformal_K)
print(f"max rel diagonal entry error: {rel_diff_diag}")

max_rel_fro_norm = np.linalg.norm(diff)/np.linalg.norm(LXS_conformal_K)
print(f"max rel frobenius norm error: {max_rel_fro_norm}")


#run the conformal case:
#create mesh
dims = [H,W,tw,tf]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_L_section(dims,num_el,'L_section')
domain.name
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
A = tw*W + tf*(H-tw)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(LXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ((tw*H**3)/12 
      + (tf**3*(W-tf))/12 
      + tf*(W-tw)*(H/2-tf/2)**2 )
print(E*I1)
print("Computed bending stiffness 1:")
print(LXS.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tf*W**3)/12 
      + (tw**3*(H-tf))/12 
      + tw*(H-tf)*(W/2-tf/2)**2 )
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

LXS_nm.setup_recovery()
disps = []
stresses = []
von_mises_list = []
for i,reaction in enumerate(['axial','shear_x','shear_y','torsion','bending_x','bending_y']):
    reactions = np.zeros((6,))
    reactions[i]=1
    disp = LXS_nm.recover_displacement(reactions)
    disp[0].name = reaction
    disp[1].name = reaction
    disps.append(disp)

    stress = LXS_nm.recover_stress(reactions)
    stress[0].name = 'sigma_'+ reaction
    stress[1].name = 'sigma_'+ reaction
    stresses.append(stress)

    von_mises = LXS_nm.get_von_mises(reactions)
    von_mises[0].name = 'von_mises_'+ reaction
    von_mises[1].name = 'von_mises_'+ reaction
    von_mises_list.append(von_mises)


def write_xdmfs(fxn_list):
    for i,fxn in enumerate(fxn_list):
        fn0 = f"output/{mesh_0.name}_{i}_{fxn[0].name}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fn0, "w") as xdmf:
            xdmf.write_mesh(mesh_0)
            xdmf.write_function(fxn[0],0.0)
        fn1 = f"output/{mesh_1.name}_{i}_{fxn[1].name}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fn1, "w") as xdmf:
            xdmf.write_mesh(mesh_1)
            xdmf.write_function(fxn[1],0.0)
#write displacements and stresses:
write_xdmfs(disps)
write_xdmfs(stresses)
write_xdmfs(von_mises_list)


print()