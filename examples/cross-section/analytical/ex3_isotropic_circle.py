#simple example of cross-sectional analysis of an isotropic circle:
import ALBATROSS

import numpy as np

#cross-section mesh definition
r = 1 #circle radius  
E = 100
nu = 0.2

params={'shape': 'circle',
            'r': r,
            'E':E,
            'nu':0.2}

#initialize cross-seciton object
circleXS = ALBATROSS.cross_section.CrossSectionAnalytical(params)

#compute the stiffness matrix
circleXS.compute_stiffness()

np.set_printoptions(precision=6)

#output stiffness matrix
print('Stiffness matrix:')
print(circleXS.K)

print("Analytical axial stiffness (EA):")
A = np.pi*r**2
print(E*A)
print("Computed Axial Stiffness:")
print(circleXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (np.pi/4)*(r**4)
print(E*I)
print("Computed Bending Stiffness:")
print(circleXS.K[4,4])

print("Torsional Stiffness (GJ):")
G = E/(2*(1+nu))
J = 2*I #for rotationally symmetric
print(G*J)
print("Computed Torsional Stiffness:")
print(circleXS.K[3,3])