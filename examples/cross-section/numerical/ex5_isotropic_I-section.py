#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np

#create mesh
N = 10
H = .1
W= .1
tf1 = 0.01
tf2 = 0.01
t_w = 0.01

points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [tf1,tf1,tw]
num_el = 3*[4] #number of elements through each wall thickness
domain = ALBATROSS.utils.create_I_section(points,thicknesses,num_el,'I_section')

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#initialize cross-section object
boxXS = ALBATROSS.cross_section.CrossSection(domain,mats)

#show me what you got
boxXS.plot_mesh()

#compute the stiffness matrix
boxXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(boxXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(boxXS.K)

print("Analytical axial stiffness (EA):")
E = mats['Unobtainium']['MECH_PROPS']['E']
A = W*H - (W-t_w)*(H-2*t_h)
print(E*A)
print("Computed Axial Stiffness:")
print(boxXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12 -((W-t_w)*(H-2*t_h)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(boxXS.K[4,4])
print("Computed bending stiffness:")
print(boxXS.K[4,4])
