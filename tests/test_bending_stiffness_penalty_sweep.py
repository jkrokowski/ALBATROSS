import numpy as np
import matplotlib.pyplot as plt
import ALBATROSS
from dolfinx import mesh
from mpi4py import MPI

#################################################################
# PARAMETERS
#################################################################

N = 6
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
eps = 1e-3
dx_vals = np.linspace(-0.045,0.045,201)

#################################################################
# PENALTY SWEEP
#################################################################

# pen_u_vals = [1e-2,1e-1,1e0,1e1,1e2]
# pen_t_vals = [1e-7,1e-5,1e-2,1e0,1e2]

# pen_u_vals = [1e0,1e1,1e2]
# pen_t_vals = [1e-2,1e0,1e2]
pen_u_vals = [1e1]
pen_t_vals = [1e-6]

#################################################################
# STORAGE
#################################################################

K44_results = {}
K55_results = {}
cond_results = {}

#################################################################
# MAIN SWEEP
#################################################################

for pen_u in pen_u_vals:
    for pen_t in pen_t_vals:

        key = (pen_u,pen_t)

        print("\n====================================")
        print(f"Running pen_u = {pen_u}, pen_t = {pen_t}")
        print("====================================")

        K44_vals = []
        K55_vals = []
        cond_vals = []

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
            mesh_1.geometry.x[:,0] += dx_w
            mesh_1.name = 'w'


            ############################################################
            # MATERIAL
            ############################################################

            unobtainium = ALBATROSS.material.Material(
                name='unobtainium',
                mat_type='ISOTROPIC',
                mech_props={'E':70e9,'nu':0.33},
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
                pen_u=pen_u,
                pen_t=pen_t,
                enable_overlap_correction=False)

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


        K44_results[key] = np.array(K44_vals)
        K55_results[key] = np.array(K55_vals)
        cond_results[key] = np.array(cond_vals)

#################################################################
# PLOTTING (AFTER FULL RUN)
#################################################################

print("\n\nAll sweeps complete. Plotting results...")

############################################################
# K44 plot
############################################################

plt.figure(figsize=(10,6))

for key,val in K44_results.items():
    pen_u,pen_t = key
    label = f"pu={pen_u:.0e}, pt={pen_t:.0e}"
    plt.plot(dx_vals,val,label=label)

plt.xlabel("dx_w")
plt.ylabel("K[4,4]")
plt.title("Bending stiffness K44 vs web position")
plt.legend(fontsize=7)
plt.grid()


############################################################
# K55 plot
############################################################

plt.figure(figsize=(10,6))

for key,val in K55_results.items():
    pen_u,pen_t = key
    label = f"pu={pen_u:.0e}, pt={pen_t:.0e}"
    plt.plot(dx_vals,val,label=label)

plt.xlabel("dx_w")
plt.ylabel("K[5,5]")
plt.title("Bending stiffness K55 vs web position")
plt.legend(fontsize=7)
plt.grid()


############################################################
# condition number plot
############################################################

plt.figure(figsize=(10,6))

for key,val in cond_results.items():
    pen_u,pen_t = key
    label = f"pu={pen_u:.0e}, pt={pen_t:.0e}"
    plt.semilogy(dx_vals,val,label=label)

plt.xlabel("dx_w")
plt.ylabel("cond(K)")
plt.title("Condition number of sectional stiffness matrix")
plt.legend(fontsize=7)
plt.grid()

plt.show()
