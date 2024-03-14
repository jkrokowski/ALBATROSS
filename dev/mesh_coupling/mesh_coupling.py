#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS

import numpy as np

#cross-section mesh definition
N = 5 #number of quad elements per side
W = 1 #square height  
H = .1 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
points2 = [[-H/2,0],[H/2, W]] #bottom left and upper right point of square


domain = ALBATROSS.utils.create_rectangle(points,[10*N,N])
domain2 = ALBATROSS.utils.create_rectangle(points2,[N,10*N])

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,mats)
squareXS2 = ALBATROSS.cross_section.CrossSection(domain2,mats)

#show me what you got
squareXS.plot_mesh()
squareXS2.plot_mesh()

#compute the stiffness matrix
squareXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)








