#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

#cross-section mesh definition
N = 20 #number of quad elements per side
W = .5 #square height  
H = .5 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
domain.name = 'ex1_square'

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()
    
#show the warping functions
squareXS.plot_warping_fxns()

#plot the warping strains (calcuated from warping functions) 
# for i in range(3):sq  
#     for j in range(3):
#         print(f'strain component ({i},{j}')
#         squareXS.plot_warping_strain(component=(i,j))
# squareXS.plot_warping_fxns()

# np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)

print("Analytical axial stiffness (EA):")
E = unobtainium.E
A = W*H
print(E*A)
print("Computed Axial Stiffness:")
print(squareXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12
print(E*I)
print("Computed bending stiffness:")
print(squareXS.K[4,4])

#demonstration of displacement and stress recovery for unit forces and moments applied to the cross-section
squareXS.setup_recovery()
disps = []
stresses = []
von_mises_list = []
for i,reaction in enumerate(['axial','shear_x','shear_y','torsion','bending_x','bending_y']):
    reactions = np.zeros((6,))
    reactions[i]=1
    disp = squareXS.recover_displacement(reactions)
    disp.name = reaction
    disps.append(disp)

    stress = squareXS.recover_stress(reactions)
    stress.name = 'sigma_'+ reaction
    stresses.append(stress)

    von_mises = squareXS.get_von_mises(reactions)
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


#improve displacement plotting:
#   define the "reaction" force for each mode as a unit vector in each direction
#   get the specific warping functions for each mode from the fundamental wapring function solutions
#   define displacement as the u_bar expression
#   interpolate this expression for the u_bar to a function and save

# improve stress plotting:
#   define the "reaction" force for each mode as a unit vector in each direction
#   get the stress expression from the stress expression defined in the begining of the process
#   interpolate this stress expression to a stress function and save

#there is a difference in a unit "force/moment applied" vs a "unit mode":
#   in some cases these will look similar (especially when the section is symmetrical)
#   however, in general a unit "force/moment" that is applied will show no coupling (as warping functions combine linearly to reproduce this mode)
#   but, a single warping function will not perfectly correspond to a deformation mode except in the symmetric, isotropic case




pApx = squareXS.compute_pApx()

pKpx = squareXS.compute_pKpx()

pKpw = squareXS.compute_pKpw()
pKpl = squareXS.compute_pKpl()


def computeFDcheck(xs,step_size,mode='x'):
    K1=xs.K1
    K2=xs.K2
    K2inv=xs.K2inv
    K=xs.K
    
    if mode =='x':
        x_num =0
        xs.msh.geometry.x[x_num,0] +=step_size
        label = mode+str(x_num)

    elif mode =='w':
        wf_num = 3
        xs.warping_functions[wf_num].x.array[0] +=step_size
        label = mode+str(wf_num)

    xs._compute_xs_stiffness_matrix()

    dK1dx = (xs.K1 - K1) /step_size
    dK2dx = (xs.K2 - K2) /step_size
    dKdx = (xs.K - K) /step_size
    dK2invdx = (xs.K2inv - K2inv) /step_size

    np.save('dK1d'+label+'_FD_d'+mode+'='+str(step_size)+'.npy',dK1dx)
    np.save('dK2d'+label+'_FD_d'+mode+'='+str(step_size)+'.npy',dK2dx)
    np.save('dK2invd'+label+'_FD_d'+mode+'='+str(step_size)+'.npy',dK2invdx)
    np.save('dKd'+label+'_FD_d'+mode+'='+str(step_size)+'.npy',dKdx)
    
    #return mesh to orginal position and recompute values
    if mode =='x':
        xs.msh.geometry.x[x_num,0] -=step_size
    elif mode =='w':
        xs.warping_functions[wf_num].x.array[0] -=step_size
    
    xs._compute_xs_stiffness_matrix()

# computeFDcheck(squareXS,step_size=0.0001,mode='x')
computeFDcheck(squareXS,step_size=0.0001,mode='w')

print()
# squareXS.compute_xs_stiffness_matrix_sensitivities()

# squareXS.plot_sensitivities()