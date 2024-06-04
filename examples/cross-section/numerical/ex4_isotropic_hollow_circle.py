#simple example of cross-sectional analysis of an isotropic hollow circle:
import ALBATROSS

import numpy as np

#cross-section mesh definition
radius = 1
wall_thickness = 0.1
num_el = 3 #number of elements through wall thickness

domain = ALBATROSS.mesh.create_hollow_circle(radius,wall_thickness,num_el,'hollow_circle')

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-seciton object
hollowCircleXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
hollowCircleXS.plot_mesh()

#compute the stiffness matrix
hollowCircleXS.get_xs_stiffness_matrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(hollowCircleXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(hollowCircleXS.K)

print("Analytical axial stiffness (EA):")
E= unobtainium.E
A = np.pi*(radius**2-(radius-wall_thickness)**2)
print(E*A)
print("Computed Axial Stiffness:")
print(hollowCircleXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (np.pi/4)*(radius**4-(radius-wall_thickness)**4)
print(E*I)
print("Computed bending stiffness:")
print(hollowCircleXS.K[4,4])

print("torsional stiffness (GJ):")
nu = unobtainium.nu
G = E/(2*(1+nu))
J = 2*I #for rotationally symmetric
print(G*J)
print("Computed torsional stiffness:")
print(hollowCircleXS.K[3,3])







