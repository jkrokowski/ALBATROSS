#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS

import numpy as np

#create mesh
N = 20
H = 1
W= 1
tf = 0.1
tw = 0.1

dims = [H,W,tf,tw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_T_section(dims,num_el,'T_section')
domain.name = 'conformal_t-section'

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
TXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium],degree=1)

#show me what you got
TXS.plot_mesh()

#compute the stiffness matrix
TXS.get_xs_stiffness_matrix()

TXS.plot_warping_fxns()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(TXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(TXS.K)

print("Analytical axial stiffness (EA):")
A = tf*W + tw*(H-tw)
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

np.save(f"T_section_K_n_{N}.npy", TXS.K)

from dolfinx import io
from mpi4py import MPI
with io.XDMFFile(MPI.COMM_WORLD, f"output/{domain.name}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

TXS.compute_xs_stiffness_matrix_sensitivities()

TXS.plot_sensitivities()