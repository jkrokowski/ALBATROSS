#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS
import numpy as np

#cross-section mesh definition
N = 2 #number of quad elements per side
W = 1.2 #square height  
H = 1 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
# squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()
    
#show the warping functions
# squareXS.plot_warping_fxns()

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
# exit()

pKpx = squareXS.compute_pKpx()

pKpw = squareXS.compute_pKpw()
pKpl = squareXS.compute_pKpl()

def computeFDcheck(xs,step_size,mode='x'):
    K1=xs.K1
    K2=xs.K2
    K2inv=xs.K2inv
    K=xs.K
    
    if mode =='x':
        xs.msh.geometry.x[0,0] +=step_size
    elif mode =='w':
        xs.warping_functions[0].x.array[0] +=step_size
    
    xs._compute_xs_stiffness_matrix()

    dK1dx = (xs.K1 - K1) /step_size
    dK2dx = (xs.K2 - K2) /step_size
    dKdx = (xs.K - K) /step_size
    dK2invdx = (xs.K2inv - K2inv) /step_size

    np.save('dK1d'+mode+'_FD_d'+mode+'='+str(step_size)+'.npy',dK1dx)
    np.save('dK2d'+mode+'_FD_d'+mode+'='+str(step_size)+'.npy',dK2dx)
    np.save('dK2invd'+mode+'_FD_d'+mode+'='+str(step_size)+'.npy',dK2invdx)
    np.save('dKd'+mode+'_FD_d'+mode+'='+str(step_size)+'.npy',dKdx)
    
    #return mesh to orginal position and recompute values
    if mode =='x':
        xs.msh.geometry.x[0,0] -=step_size
    elif mode =='w':
        xs.warping_functions[0].x.array[0] -=step_size
    
    xs._compute_xs_stiffness_matrix()

computeFDcheck(squareXS,step_size=0.0001,mode='x')
computeFDcheck(squareXS,step_size=0.0001,mode='w')

print()
# squareXS.compute_xs_stiffness_matrix_sensitivities()

# squareXS.plot_sensitivities()