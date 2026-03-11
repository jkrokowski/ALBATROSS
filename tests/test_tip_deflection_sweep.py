import numpy as np
import matplotlib.pyplot as plt
import ALBATROSS
from dolfinx import mesh
from mpi4py import MPI

#################################################################
# PARAMETERS
#################################################################

N = 8
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

dx_vals = np.linspace(-0.045,0.045,51)

tip_vals = []

#################################################################
# SWEEP LOOP
#################################################################

for dx_w in dx_vals:

    print("Running dx_w =",dx_w)

    #################################################################
    # BUILD CROSS SECTION GEOMETRY
    #################################################################

    mesh_0 = mesh.create_unit_square(
        MPI.COMM_WORLD, m1, n1,
        cell_type=mesh.CellType.quadrilateral)

    mesh_0.geometry.x[:, :2] -= .5
    mesh_0.geometry.x[:, 1] *= tf
    mesh_0.geometry.x[:, 0] *= W
    mesh_0.geometry.x[:, 1] += H/2 - tf/2
    mesh_0.name = 'f'


    mesh_1 = mesh.create_unit_square(
        MPI.COMM_WORLD, m2, n2,
        cell_type=mesh.CellType.quadrilateral)

    mesh_1.geometry.x[:, :2] -= .5
    mesh_1.geometry.x[:, 0] *= tw
    mesh_1.geometry.x[:, 1] *= W

    # <-- DESIGN VARIABLE: translate web
    mesh_1.geometry.x[:,0] += dx_w

    mesh_1.name = 'w'


    #################################################################
    # CROSS SECTION ANALYSIS
    #################################################################

    meshes = [mesh_0,mesh_1]

    unobtainium = ALBATROSS.material.Material(
        name='unobtainium',
        mat_type='ISOTROPIC',
        mech_props={'E':70e9,'nu':0.33},
        density=2700)

    XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

    TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(
        XSs,
        pen_u=1e2,
        pen_t=1e2)

    TXS_nm.get_xs_stiffness_matrix()

    xs_list = [TXS_nm]


    #################################################################
    # BEAM MODEL
    #################################################################

    nodal_points = [p1,p2]
    num_segments = len(nodal_points)-1
    num_ele = [10]

    beam_axis = ALBATROSS.axial.BeamAxis(
        nodal_points,
        num_ele,
        "txs_nm_beam")

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

    #################################################################
    # RECORD TIP DISPLACEMENT
    #################################################################

    tip = CantileverBeam.get_local_disp([p2])[0][2]

    print("Tip deflection:",tip)

    tip_vals.append(tip)


#################################################################
# PLOT RESULTS
#################################################################

dx_vals = np.array(dx_vals)
tip_vals = np.array(tip_vals)

plt.figure(figsize=(7,5))
plt.plot(dx_vals,tip_vals,'o-',lw=2)
plt.xlabel("Web translation dx_w")
plt.ylabel("Tip displacement (z)")
plt.title("Cantilever tip displacement vs web translation")
plt.grid(True)
plt.tight_layout()
plt.show()

print()