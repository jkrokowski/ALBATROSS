#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np

# np.set_printoptions(precision=3)

# m1,n1 = 102,10
# m2,n2 = 9,104
# m1,n1 = 54,5
# m2,n2 = 4,45
# m1,n1 = 40+1,4
# m2,n2 = 4,40+1
# m1,n1 = 30,3
# m2,n2 = 3,30
# m1,n1 = 10,1
# m2,n2 = 1,9

N = 8
offset = 1

m1,n1 = N*10+offset,N
m2,n2 = N,N*10+offset

H = 1
W = 1
tf = .1
tw = .1

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
# mesh_1.geometry.x[:, 0] += T2/2

mesh_1.name = 'w'

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10e6,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen_u=1e6,pen_t=1)

TXS_nm.plot_meshes()

TXS_nm.get_xs_stiffness_matrix()

TXS_nm.plot_warping_fxns()

# for i in range(3):
#     for j in range(3):
#         TXS_nm.plot_warping_strains(component=(i,j))

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(TXS_nm.K)

print("Analytical axial stiffness (EA):")
A = tf*W + tw*(H-tf)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(TXS_nm.K[0,0])

print("Analytical Bending stiffness (EI1):")
I1 = ( (tw*H**3)/12 + 
      ( (((W-tw)*tf**3)/12) 
       + ((tf*(W-tw)))*((H-tf)/2)**2) )
print(E*I1)
print("Computed bending stiffness 1:")
print(TXS_nm.K[4,4])

print("Analytical Bending stiffness (EI2):")
I2 = ((tf*W**3)/12) + ((H-tf)*tw**3)/12
print(E*I2)
print("Computed bending stiffness 2:")
print(TXS_nm.K[5,5])


#compare to conformal approach:
#create mesh
dims = [H,W,tf,tw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_T_section(dims,num_el,'T_section')

# unobtainium = ALBATROSS.material.Material(name='unobtainium',
#                                            mat_type='ISOTROPIC',
#                                            mech_props={'E':E,'nu':0.2},
#                                            density=2700)

#initialize cross-section object
TXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
TXS.plot_mesh()

#compute the stiffness matrix
TXS.get_xs_stiffness_matrix()

TXS.plot_warping_fxns()

# for i in range(3):
#     for j in range(3):
#         TXS.plot_warping_strain(component=(i,j))

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(TXS.K)

print("Analytical axial stiffness (EA):")
A = tf*W + tf*(H-tw)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(TXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ( (tw*H**3)/12 + 
      ( (((W-tw)*tf**3)/12) 
       + ((tf*(W-tw)))*((H-tf)/2)**2) )
print(E*I1)
print("Computed bending stiffness 1:")
print(TXS.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tf*W**3)/12) + ((H-tf)*tw**3)/12
print(E*I2)
print("Computed bending stiffness 2:")
print(TXS.K[5,5])

#compute difference between matrix entries of beam constituitive matrix
abs_diff = TXS_nm.K - TXS.K
rel_diff = abs_diff/TXS.K

print("Total difference:")
print(abs_diff)

print("Relative Difference:")
print(rel_diff)

print("Maximum Absolute Difference of Diagonal Entries:")
print(np.max(np.abs(np.diag(abs_diff))))

print("Maximum Relative Difference of Diagonal Entries:")
print(np.max(np.abs(np.diag(rel_diff))))

print("Frobenius Norm of Absolute Difference:")
print(np.linalg.norm(abs_diff))

print("Relative Frobenius Norm of Absolute Difference:")
print(np.linalg.norm(abs_diff)/np.linalg.norm(TXS.K))

print("energy norm:")
from scipy.linalg import eigh

lam = eigh(abs_diff,TXS.K,eigvals_only=True)
print(np.max(np.abs(lam)))
print()