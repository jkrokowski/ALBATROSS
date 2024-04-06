#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS

import numpy as np

#cross-section mesh definition
N = 5 #number of quad elements per side
W = 1 #square height  
H = 1 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.utils.create_rectangle(points,[N,N])

# mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
#                         'MECH_PROPS':{'E':100.,'nu':.2} ,
#                         'DENSITY':2.7e3}
#         }
unobtainium = ALBATROSS.material.Material(name='Aluminium7075',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-seciton object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(squareXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)

print("Analytical axial stiffness (EA):")
E = mats['Unobtainium']['MECH_PROPS']['E']
A = W*H
print(E*A)
print("Computed Axial Stiffness:")
print(squareXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12
print(E*I)
print("Computed bending stiffness:")
print(squareXS.K[4,4])








