#simple example of cross-sectional analysis of an isotropic hollow circle:
import ALBATROSS

import numpy as np

#cross-section mesh definition
r = 1 #circle radius
t = .1 #wall thickness  
E = 100
nu = 0.2

params={'shape': 'hollow circle',
            'r': r,
            't': t,
            'E':E,
            'nu':0.2}

#initialize cross-seciton object
hollowCircleXS = ALBATROSS.cross_section.CrossSectionAnalytical(params)

#compute the stiffness matrix
hollowCircleXS.compute_stiffness()

np.set_printoptions(precision=6)

#output stiffness matrix
print('Stiffness matrix:')
print(hollowCircleXS.K)

print("Analytical axial stiffness (EA):")
A = np.pi*(r**2-(r-t)**2)
print(E*A)
print("Computed Axial Stiffness:")
print(hollowCircleXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (np.pi/4)*(r**4-(r-t)**4)
print(E*I)
print("Computed bending stiffness:")
print(hollowCircleXS.K[4,4])

print("torsional stiffness (GJ):")
G = E/(2*(1+nu))
J = 2*I #for rotationally symmetric
print(G*J)
print("Computed torsional stiffness:")
print(hollowCircleXS.K[3,3])