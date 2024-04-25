#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS

from dolfinx import mesh,plot
import pyvista
from mpi4py import MPI
import numpy as np

#create mesh
N = 10
W = .1
H = .1
t1 = 0.01
t2 = 0.01
t3 = 0.01
t4 = 0.01

points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [t1,t2,t3,t4]
num_el = 4*[4] #number of elements through each wall thickness
domain = ALBATROSS.mesh.create_hollow_box(points,thicknesses,num_el,'box_xs')

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
boxXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
boxXS.plot_mesh()

#compute the stiffness matrix
boxXS.get_xs_stiffness_matrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(boxXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(boxXS.K)

print("Analytical axial stiffness (EA):")
E = unobtainium.E
A = W*H - (H-t1-t3)*(W-t2-t4)
print(E*A)
print("Computed Axial Stiffness:")
print(boxXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12 - ((W-t2-t4)*(H-t1-t3)**3)/12
print(E*I)
print("Computed bending stiffness:")
print(boxXS.K[4,4])
