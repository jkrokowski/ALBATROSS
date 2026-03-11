import numpy as np
import matplotlib.pyplot as plt
import ALBATROSS
from dolfinx import mesh
from mpi4py import MPI

#################################################################
# PARAMETERS
#################################################################

N = 4
W = 0.1
H = 0.1
L = 2.0

offset = 1
h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

tf = H/h_to_f
tw = W/w_to_w

dx_vals = np.linspace(-0.045,0.045,101)
# dx_vals = np.linspace(-0.045,-0.0288,19)

#################################################################
# STORAGE
#################################################################

K44_vals = []
K55_vals = []
cond_vals = []

#################################################################
# SWEEP
#################################################################

for dx_w in dx_vals:

    print("dx_w =",dx_w)

    ############################################################
    # BUILD MESHES
    ############################################################

    mesh_0 = mesh.create_unit_square(
        MPI.COMM_WORLD,
        m1,
        n1,
        cell_type=mesh.CellType.quadrilateral)

    mesh_0.geometry.x[:, :2] -= .5
    mesh_0.geometry.x[:, 1] *= tf
    mesh_0.geometry.x[:, 0] *= W
    mesh_0.geometry.x[:, 1] += H/2 - tf/2
    mesh_0.name = 'f'


    mesh_1 = mesh.create_unit_square(
        MPI.COMM_WORLD,
        m2,
        n2,
        cell_type=mesh.CellType.quadrilateral)

    mesh_1.geometry.x[:, :2] -= .5
    mesh_1.geometry.x[:, 0] *= tw
    mesh_1.geometry.x[:, 1] *= W

    # design variable
    mesh_1.geometry.x[:,0] += dx_w

    mesh_1.name = 'w'


    ############################################################
    # MATERIAL
    ############################################################

    unobtainium = ALBATROSS.material.Material(
        name='unobtainium',
        mat_type='ISOTROPIC',
        mech_props={'E':100,'nu':0.33},
        density=2700)


    ############################################################
    # CROSS SECTIONS
    ############################################################

    meshes = [mesh_0,mesh_1]

    XSs = [
        ALBATROSS.cross_section.CrossSection(msh,[unobtainium])
        for msh in meshes
    ]


    ############################################################
    # COUPLED CROSS SECTION
    ############################################################

    TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(
        XSs,
        pen_u=1e0,
        pen_t=1e-7)

    TXS_nm.get_xs_stiffness_matrix()

    K = TXS_nm.K


    ############################################################
    # RECORD VALUES
    ############################################################

    K44_vals.append(K[4,4])
    K55_vals.append(K[5,5])

    try:
        cond_vals.append(np.linalg.cond(K))
    except:
        cond_vals.append(np.nan)


#################################################################
# PLOTTING
#################################################################

dx_vals = np.array(dx_vals)
K44_vals = np.array(K44_vals)
K55_vals = np.array(K55_vals)
cond_vals = np.array(cond_vals)

plt.figure()
plt.plot(dx_vals,K44_vals,'o-')
plt.xlabel("dx_w")
plt.ylabel("K[4,4]")
plt.title("Bending stiffness K[4,4]")
plt.grid()

plt.figure()
plt.plot(dx_vals,K55_vals,'o-')
plt.xlabel("dx_w")
plt.ylabel("K[5,5]")
plt.title("Bending stiffness K[5,5]")
plt.grid()

plt.figure()
plt.semilogy(dx_vals,cond_vals,'o-')
plt.xlabel("dx_w")
plt.ylabel("cond(K)")
plt.title("Condition number of sectional stiffness matrix")
plt.grid()

plt.show()

print()