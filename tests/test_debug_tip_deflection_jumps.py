import numpy as np
import matplotlib.pyplot as plt
import ALBATROSS
from dolfinx import mesh
from mpi4py import MPI

#################################################################
# PARAMETERS
#################################################################

N = 16
W = 0.1
H = 0.1
L = 2.0
F = 1000.0

offset = 1
h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

tf = H/h_to_f
tw = W/w_to_w

p1 = (0,0,0)
p2 = (L,0,0)

dx_vals = np.linspace(-0.045,0.045,101)

#################################################################
# STORAGE
#################################################################

tip_vals = []
K_norms = []
K_conds = []
K_max = []
nan_counts = []

#################################################################
# SWEEP
#################################################################

for dx_w in dx_vals:

    print("\n==============================")
    print("Running dx_w =",dx_w)
    print("==============================")

    try:

        ############################################################
        # BUILD CROSS SECTION
        ############################################################

        mesh_0 = mesh.create_unit_square(
            MPI.COMM_WORLD,m1,n1,
            cell_type=mesh.CellType.quadrilateral)

        mesh_0.geometry.x[:, :2] -= .5
        mesh_0.geometry.x[:,1] *= tf
        mesh_0.geometry.x[:,0] *= W
        mesh_0.geometry.x[:,1] += H/2 - tf/2
        mesh_0.name = "f"


        mesh_1 = mesh.create_unit_square(
            MPI.COMM_WORLD,m2,n2,
            cell_type=mesh.CellType.quadrilateral)

        mesh_1.geometry.x[:, :2] -= .5
        mesh_1.geometry.x[:,0] *= tw
        mesh_1.geometry.x[:,1] *= W
        mesh_1.geometry.x[:,0] += dx_w
        mesh_1.name = "w"


        ############################################################
        # MATERIAL
        ############################################################

        unobtainium = ALBATROSS.material.Material(
            name='unobtainium',
            mat_type='ISOTROPIC',
            mech_props={'E':70e9,'nu':0.33},
            density=2700)

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
            pen_u=1e1,
            pen_t=1e1)

        TXS_nm.get_xs_stiffness_matrix()

        K = TXS_nm.K

        ############################################################
        # STIFFNESS DIAGNOSTICS
        ############################################################

        K_norm = np.linalg.norm(K)
        K_max_val = np.max(np.abs(K))

        try:
            K_cond = np.linalg.cond(K)
        except:
            K_cond = np.nan

        nan_count = np.isnan(K).sum() + np.isinf(K).sum()

        print("||K||        =",K_norm)
        print("cond(K)      =",K_cond)
        print("max(|K|)     =",K_max_val)
        print("nan entries  =",nan_count)

        K_norms.append(K_norm)
        K_conds.append(K_cond)
        K_max.append(K_max_val)
        nan_counts.append(nan_count)


        ############################################################
        # BEAM SOLVE
        ############################################################

        xs_list = [TXS_nm]

        nodal_points = [p1,p2]
        num_segments = 1
        num_ele = [10]

        beam_axis = ALBATROSS.axial.BeamAxis(
            nodal_points,
            num_ele,
            "beam")

        orientations = np.tile([0,1,0],num_segments+1)

        xs_adjacency_list = [[0,0]]

        xs_info = [xs_list,orientations,xs_adjacency_list]

        CantileverBeam = ALBATROSS.beam.Beam(
            beam_axis,
            xs_info,
            segment_type='LINEAR')

        CantileverBeam.add_clamped_point(p1)
        CantileverBeam.add_point_load([(0,0,-F)],[p2])

        CantileverBeam.solve()

        tip = CantileverBeam.get_local_disp([p2])[0][2]

        print("tip =",tip)

        tip_vals.append(tip)

    except Exception as e:

        print("FAILED:",e)

        tip_vals.append(np.nan)
        K_norms.append(np.nan)
        K_conds.append(np.nan)
        K_max.append(np.nan)
        nan_counts.append(np.nan)


#################################################################
# POST ANALYSIS
#################################################################

tip_vals = np.array(tip_vals)

print("\n\nSpike detection:")

med_step = np.median(np.abs(np.diff(tip_vals)))

for i in range(1,len(tip_vals)):

    if abs(tip_vals[i]-tip_vals[i-1]) > 5*med_step:
        print("Spike near dx =",dx_vals[i])


#################################################################
# PLOTS
#################################################################

plt.figure()
plt.plot(dx_vals,tip_vals,'o-')
plt.title("Tip displacement")
plt.xlabel("dx_w")
plt.ylabel("tip")
plt.grid()

plt.figure()
plt.plot(dx_vals,K_norms,'o-')
plt.title("||K||")
plt.xlabel("dx_w")
plt.grid()

plt.figure()
plt.plot(dx_vals,K_conds,'o-')
plt.title("cond(K)")
plt.xlabel("dx_w")
plt.grid()

plt.figure()
plt.plot(dx_vals,K_max,'o-')
plt.title("max(|K|)")
plt.xlabel("dx_w")
plt.grid()

plt.show()