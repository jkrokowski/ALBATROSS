#simple example of cross-sectional analysis of an isotropic symmetric box:
import ALBATROSS
from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI

#create mesh
N = 6
H = .1
W= .1
tfh = 0.01
tfw = 0.01

dims = [H,W,tfh,tfw]
num_el = [N,N]#number of elements through each wall thickness
domain = ALBATROSS.mesh.create_L_section(dims,num_el,'L_section')
domain.name = 'L_section'
unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':70e9,'nu':0.33},
                                           density=2700)

#initialize cross-section object
LXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
LXS.plot_mesh()

#compute the stiffness matrix
LXS.get_xs_stiffness_matrix()

# LXS.plot_warping_fxns()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(LXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(LXS.K)

print("Analytical axial stiffness (EA):")
A = tfw*(W-tfh) + tfh*(H-tfw) + tfh*tfw
E=unobtainium.E
print(E*A)
print("Computed Axial Stiffness:")
print(LXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I1 = ((tfh*H**3)/12 
      + (tfw**3*(W-tfh))/12 
      + tfw*(W-tfh)*(H/2-tfw/2)**2 )
print(E*I1)
print("Computed bending stiffness 1:")
print(LXS.K[4,4])

print("Analytical Bending stiffness (EI):")
I2 = ((tfw*W**3)/12 
      + (tfh**3*(H-tfw))/12 
      + tfh*(H-tfw)*(W/2-tfh/2)**2 )
print(E*I2)
print("Computed bending stiffness 2:")
print(LXS.K[5,5])

np.save(f"L_section_K_n_{N}_H{H}_W{W}.npy", LXS.K)


# LXS.compute_xs_stiffness_matrix_sensitivities()

# LXS.plot_sensitivities()

#demonstration of displacement and stress recovery for unit forces and moments applied to the cross-section
LXS.setup_recovery()
disps = []
stresses = []
von_mises_list = []
for i,reaction in enumerate(['axial','shear_x','shear_y','torsion','bending_x','bending_y']):
    reactions = np.zeros((6,))
    reactions[i]=1
    disp = LXS.recover_displacement(reactions)
    disp.name = reaction
    disps.append(disp)

    stress = LXS.recover_stress(reactions)
    stress.name = 'sigma_'+ reaction
    stresses.append(stress)

    von_mises = LXS.get_von_mises(reactions)
    von_mises.name = 'von_mises_'+ reaction
    von_mises_list.append(von_mises)
    
def write_xdmfs(fxn_list):
    for i,fxn in enumerate(fxn_list):
        fn = f"output/{domain.name}_{i}_{fxn.name}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fn, "w") as xdmf:
            xdmf.write_mesh(domain)
            xdmf.write_function(fxn,0.0)


#write displacements and stresses:
write_xdmfs(disps)
write_xdmfs(stresses)
write_xdmfs(von_mises_list)

print()