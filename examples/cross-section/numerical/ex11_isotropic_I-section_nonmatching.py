#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS

import numpy as np

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

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(IXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(IXS.K)

print("Analytical axial stiffness (EA):")
A = W*H - (W-tw)*(H-2*tf)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(IXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12 -((W-tw)*(H-2*tf)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(IXS.K[4,4])
