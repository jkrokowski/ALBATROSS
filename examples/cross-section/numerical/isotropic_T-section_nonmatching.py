import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from dolfinx.io import XDMFFile
import ALBATROSS
from petsc4py import PETSc

default_scalar_type = PETSc.ScalarType   


#=================== mesh construction ==================#
N = 6
offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

H = 1
W = 1
tf = 1/h_to_f
tw = 1/w_to_w

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2
mesh_0.name = f'f_N{N}'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
# mesh_1.geometry.x[:,0] += -0.45
mesh_1.name = f'w_N{N}'

#================= initialize individual cross-sections ===========#
meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

#================= initialize coupled cross-section ===========#
TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen_u=1e0,pen_t=1)
TXS_nm.plot_meshes()

#identify meshes:
mesh_A = TXS_nm.XSs[0].msh
mesh_B = TXS_nm.XSs[1].msh

TXS_nm.get_xs_stiffness_matrix()
mesh_C = TXS_nm.collisions[(0,1)].mortar_mesh.msh

TXS_nm.plot_warping_fxns()
K = TXS_nm.K

# dK = np.array([[1,0,0,0,0,0],
#                [0,0,0,0,0,0],
#                [0,0,0,0,0,0],
#                [0,0,0,0,0,0],
#                [0,0,0,0,0,0],
#                [0,0,0,0,0,0]])
# dx0 = TXS_nm._compute_pK_action(dK,0)
# dw0 = TXS_nm._compute_pK_action(dK,0,derivative_type = 'w')
# dx1 = TXS_nm._compute_pK_action(dK,1)
# dw1 = TXS_nm._compute_pK_action(dK,1,derivative_type = 'w')
# dl = TXS_nm._compute_pK_action(dK,0,derivative_type = 'l')

# np.set_printoptions(precision=3)
# print(K)
# TXS_conformal_K = np.load("T_section_K_n_20.npy")

# diff=K-TXS_conformal_K
# rel_diff = (K-TXS_conformal_K)/TXS_conformal_K
# abs_diff_diag = np.diag(diff)
# rel_diff_diag = abs_diff_diag/np.diag(TXS_conformal_K)

# max_rel_fro_norm = np.linalg.norm(diff)/np.linalg.norm(TXS_conformal_K)
# print(f"max rel frobenius norm error: {max_rel_fro_norm}")

#TODO: need to work on the visualization(2Dxy vs 3Dyz), but the functions seem to be correct
#demonstration of displacement and stress recovery for unit forces and moments applied to the cross-section
TXS_nm.setup_recovery()
disps = []
stresses = []
von_mises_list = []
for i,reaction in enumerate(['axial','shear_x','shear_y','torsion','bending_x','bending_y']):
    reactions = np.zeros((6,))
    reactions[i]=1
    disp = TXS_nm.recover_displacement(reactions)
    disp[0].name = reaction
    disp[1].name = reaction
    disps.append(disp)

    stress = TXS_nm.recover_stress(reactions)
    stress[0].name = 'sigma_'+ reaction
    stress[1].name = 'sigma_'+ reaction
    stresses.append(stress)

    von_mises = TXS_nm.get_von_mises(reactions)
    von_mises[0].name = 'von_mises_'+ reaction
    von_mises[1].name = 'von_mises_'+ reaction
    von_mises_list.append(von_mises)

# for msh in [mesh_A,mesh_B,mesh_C]:
#     with io.XDMFFile(MPI.COMM_WORLD, f"output/t-section_nm_{msh.name}.xdmf", "w") as xdmf:
#         xdmf.write_mesh(msh)
# for i,msh in enumerate([mesh_A,mesh_B]):
#     with io.XDMFFile(MPI.COMM_WORLD, f"output/t-section_nm_{msh.name}.xdmf", "a") as xdmf:
#         # xdmf.write_function(disps[0],0.0)
#         # xdmf.write_function(stresses[0],0.0)
#         for fxn in disps:
#             xdmf.write_function(fxn[i],0.0)
#         for fxn in stresses:
#             xdmf.write_function(fxn[i],0.0)
#         for fxn in von_mises_list:
#             xdmf.write_function(fxn[i],0.0)
    # for i,function in enumerate(functions):
    #     ubar = function.sub(0).collapse()
    #     # uhat = function.sub(0).collapse()
    #     # utilde = function.sub(2).collapse()
    #     # ubreve = function.sub(3).collapse()
    #     ubar.name = f'ubar_{i}'
    #     xdmf.write_function(ubar,t=0.0)

def write_xdmfs(fxn_list):
    for i,fxn in enumerate(fxn_list):
        fn0 = f"output/{mesh_0.name}_{i}_{fxn[0].name}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fn0, "w") as xdmf:
            xdmf.write_mesh(mesh_0)
            xdmf.write_function(fxn[0],0.0)
        fn1 = f"output/{mesh_1.name}_{i}_{fxn[1].name}.xdmf"
        with XDMFFile(MPI.COMM_WORLD, fn1, "w") as xdmf:
            xdmf.write_mesh(mesh_1)
            xdmf.write_function(fxn[1],0.0)
#write displacements and stresses:
write_xdmfs(disps)
write_xdmfs(stresses)
write_xdmfs(von_mises_list)


print()