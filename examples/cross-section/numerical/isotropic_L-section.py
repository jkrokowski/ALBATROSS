#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS

import numpy as np

#create mesh
N = 3
H = 1
W= 1.2
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


LXS.compute_xs_stiffness_matrix_sensitivities()

LXS.plot_sensitivities()