#simple example of cross-sectional analysis of an isotropic T-section:
import ALBATROSS
from dolfinx.io import XDMFFile
import numpy as np

#create mesh
N = 2
H = 1.0
W= 1.0
tf = 0.1
tw = 0.1

dims = [H,W,tf,tw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_T_section(dims,num_el,'T_section')
domain.name = 'conformal_t-section'

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
TXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium],degree=1)

#show me what you got
TXS.plot_mesh()

#compute the stiffness matrix
TXS.get_xs_stiffness_matrix()

TXS.plot_warping_fxns()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(TXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(TXS.K)

print("Analytical axial stiffness (EA):")
A = tf*W + tw*(H-tw)
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(TXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ( (tw*H**3)/12 + 
      ( (((W-tw)*tf**3)/12) 
       + ((tf*(W-tw)))*((H-tf)/2)**2) )
print(E*I1)
print("Computed bending stiffness 1:")
print(TXS.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tf*W**3)/12) + ((H-tf)*tw**3)/12
print(E*I2)
print("Computed bending stiffness 2:")
print(TXS.K[5,5])

# np.save(f"T_section_K_n_{N}.npy", TXS.K)

from dolfinx import io
from mpi4py import MPI
with io.XDMFFile(MPI.COMM_WORLD, f"output/{domain.name}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

#demonstration of displacement and stress recovery for unit forces and moments applied to the cross-section
TXS.setup_recovery()
disps = []
stresses = []
von_mises_list = []
for i,reaction in enumerate(['axial','shear_x','shear_y','torsion','bending_x','bending_y']):
    reactions = np.zeros((6,))
    reactions[i]=1
    disp = TXS.recover_displacement(reactions)
    disp.name = reaction
    disps.append(disp)

    stress = TXS.recover_stress(reactions)
    stress.name = 'sigma_'+ reaction
    stresses.append(stress)

    von_mises = TXS.get_von_mises(reactions)
    von_mises.name = 'von_mises_'+ reaction
    von_mises_list.append(von_mises)
    
with XDMFFile(MPI.COMM_WORLD, "output/"+domain.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
with XDMFFile(MPI.COMM_WORLD, "output/"+domain.name+".xdmf", "a") as xdmf:
    # xdmf.write_function(disps[0],0.0)
    # xdmf.write_function(stresses[0],0.0)
    for fxn in disps:
        xdmf.write_function(fxn,0.0)
    for fxn in stresses:
        xdmf.write_function(fxn,0.0)
    for fxn in von_mises_list:
        xdmf.write_function(fxn,0.0)