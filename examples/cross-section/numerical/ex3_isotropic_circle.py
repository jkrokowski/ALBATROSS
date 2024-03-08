#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS

from dolfinx import mesh
from mpi4py import MPI
import numpy as np

#cross-section mesh definition
radius = 1
num_el = 10 #number of elements through wall thickness

domain = ALBATROSS.utils.create_circle(radius,num_el,'disk')

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#initialize cross-seciton object
circleXS = ALBATROSS.cross_section.CrossSection(domain,mats)

#show me what you got
circleXS.plot_mesh()

#compute the stiffness matrix
circleXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(circleXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(circleXS.K)

print("Analytical axial stiffness (EA):")
E = mats['Unobtainium']['MECH_PROPS']['E']
A = np.pi*radius**2
print(E*A)
print("Computed Axial StiALBATROSS/examples/cross-section/ex4_isotropic_hollow_disk.pyffness:")
print(circleXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (np.pi/4)*(radius**4)
print(E*I)
print("Computed bending stiffness:")
print(circleXS.K[4,4])

print("torsional stiffness (GJ):")
nu = mats['Unobtainium']['MECH_PROPS']['nu']
G = E/(2*(1+nu))
J = 2*I #for rotationally symmetric
print(G*J)
print("Computed torsional stiffness:")
print(circleXS.K[3,3])







