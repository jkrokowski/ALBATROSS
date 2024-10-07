#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS

import numpy as np

#cross-section mesh definition
W = 0.1 #square height  
H = 0.1 #square depth
E = 100
nu = 0.2

params={'shape': 'rectangle',
            'h': H,
            'w': W,
            'E':E,
            'nu':0.2}

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSectionAnalytical(params)

#compute the stiffness matrix
squareXS.compute_stiffness()

np.set_printoptions(precision=6)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)

print("Analytical axial stiffness (EA):")
A = W*H
print(E*A)
print("Computed Axial Stiffness:")
print(squareXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12
print(E*I)
print("Computed bending stiffness:")
print(squareXS.K[4,4])








