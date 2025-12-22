import numpy as np
from mpi4py import MPI
from dolfinx import mesh, io
import ALBATROSS
from petsc4py import PETSc

default_scalar_type = PETSc.ScalarType   


#=================== mesh construction ==================#
N = 2
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
mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
# mesh_1.geometry.x[:,0] += -0.45
mesh_1.name = 'w'

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

dK = np.array([[1,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]])
dx0 = TXS_nm._compute_pK_action(dK,0)
dw0 = TXS_nm._compute_pK_action(dK,0,derivative_type = 'w')
dx1 = TXS_nm._compute_pK_action(dK,1)
dw1 = TXS_nm._compute_pK_action(dK,1,derivative_type = 'w')
dl = TXS_nm._compute_pK_action(dK,0,derivative_type = 'l')

np.set_printoptions(precision=3)
print(K)
TXS_conformal_K = np.load("T_section_K_n_20.npy")

diff=K-TXS_conformal_K
rel_diff = (K-TXS_conformal_K)/TXS_conformal_K
abs_diff_diag = np.diag(diff)
rel_diff_diag = abs_diff_diag/np.diag(TXS_conformal_K)

max_rel_fro_norm = np.linalg.norm(diff)/np.linalg.norm(TXS_conformal_K)
print(f"max rel frobenius norm error: {max_rel_fro_norm}")

#TODO: need to work on the visualization(2Dxy vs 3Dyz), but the functions seem to be correct

for msh in [mesh_A,mesh_B,mesh_C]:
    with io.XDMFFile(MPI.COMM_WORLD, f"output/t-section_nm_{msh.name}.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
    # for i,function in enumerate(functions):
    #     ubar = function.sub(0).collapse()
    #     # uhat = function.sub(0).collapse()
    #     # utilde = function.sub(2).collapse()
    #     # ubreve = function.sub(3).collapse()
    #     ubar.name = f'ubar_{i}'
    #     xdmf.write_function(ubar,t=0.0)

print()