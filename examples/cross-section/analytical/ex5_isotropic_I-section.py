#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS

import numpy as np

#cross-section mesh definition
W = 0.1 #square height  
H = 0.1 #square depth
t_h = 0.01
t_w = 0.01
E = 100
nu = 0.2

params={'shape': 'I',
            'h': H,
            'w': W,
            't_h': t_h,
            't_w': t_w,
            'E':E,
            'nu':0.2}

#initialize cross-seciton object
boxXS = ALBATROSS.cross_section.CrossSectionAnalytical(params)

#compute the stiffness matrix
boxXS.compute_stiffness()

np.set_printoptions(precision=6)

#output stiffness matrix
print('Stiffness matrix:')
print(boxXS.K)

print("Analytical axial stiffness (EA):")
A = W*H - (W-t_w)*(H-2*t_h)
print(E*A)
print("Computed Axial Stiffness:")
print(boxXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12 -((W-t_w)*(H-2*t_h)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(boxXS.K[4,4])








