import numpy as np
from mpi4py import MPI
from dolfinx import mesh,fem
import ufl
from dolfinx.io import XDMFFile
import ALBATROSS
from petsc4py import PETSc
import basix.ufl

default_scalar_type = PETSc.ScalarType   

#=================== mesh construction ==================#
N = 4
offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

H = 0.1 #m
W = 0.1 #m 
tf = H/h_to_f
tw = W/w_to_w

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
# mesh_1.geometry.x[:,0] += 0.0162
mesh_1.name = f'w_N{N}'

identifier_string = ''

#================= initialize individual cross-sections ===========#
meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':70e9,'nu':0.33},
                                           density=2700)

XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

#================= initialize coupled cross-section ===========#
# val = 
TXS_nm = ALBATROSS.cross_section.CoupledCrossSection(XSs,pen_u=1e1,
                                                     pen_t=1e-1,
                                                     enable_overlap_correction=False)
TXS_nm.plot_meshes()

#identify meshes:
mesh_A = TXS_nm.XSs[0].msh
mesh_B = TXS_nm.XSs[1].msh

TXS_nm.get_xs_stiffness_matrix()
mesh_C = TXS_nm.collisions[(0,1)].mortar_mesh.msh

TXS_nm.plot_warping_fxns()
K = TXS_nm.K

np.set_printoptions(precision=3)
print(K)
TXS_conformal_K = np.load("T_section_K_n_6_H0.1_W0.1.npy")

diff=K-TXS_conformal_K
rel_diff = (K-TXS_conformal_K)/TXS_conformal_K
abs_diff_diag = np.diag(diff)
rel_diff_diag = abs_diff_diag/np.diag(TXS_conformal_K)
print(f"max rel diagonal entry error: {rel_diff_diag}")

max_rel_fro_norm = np.linalg.norm(diff)/np.linalg.norm(TXS_conformal_K)
print(f"max rel frobenius norm error: {max_rel_fro_norm}")

print("compute L2 error:")
# TXS_nm.XSs[0].warping_functions[0]
# TXS_nm.XSs[1].warping_functions[0]
# collision=(0,1)
# L2_errors = []
# for idx in range(6):
#     TXS_nm.collisions[collision].PA.mult(TXS_nm.XSs[0].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_A = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     TXS_nm.collisions[collision].PB.mult(TXS_nm.XSs[1].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_B = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     diff =wC_A-wC_B
#     L2_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(diff,diff)*TXS_nm.collisions[(0,1)].dx)))
#     L2_errors.append(L2_error)
# print(L2_errors)

# print("compute H1 seminorm error:")
# H1semi_errors = []
# for idx in range(6):
#     TXS_nm.collisions[collision].PA.mult(TXS_nm.XSs[0].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_A = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     eps_A = TXS_nm.collisions[collision].mortar_xs.warping2strain(wC_A,0)
#     TXS_nm.collisions[collision].PB.mult(TXS_nm.XSs[1].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_B = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     eps_B = TXS_nm.collisions[collision].mortar_xs.warping2strain(wC_B,0)
#     diff = eps_A-eps_B
#     H1semi_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(diff,diff)*TXS_nm.collisions[(0,1)].dx)))
#     H1semi_errors.append(H1semi_error)
# print(H1semi_errors)

# print("compute strain error A to mortar:")
# #strain communtation test:
# #tests whether transfer --> gradient is the same as gradient --> transfer
# H1semi_strain_errors = []
# for idx in range(6):
#     #get the interpolated, then evaluated strain
#     TXS_nm.collisions[collision].PA.mult(TXS_nm.XSs[0].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_A = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     eps_A = TXS_nm.collisions[collision].mortar_xs.warping2strain(wC_A,1)

#     #get the strain function on the foreground mesh by interpolating an expression
#     eps_foreground = fem.Function(TXS_nm.XSs[0].V_sigma)
#     eps_foreground_ufl = TXS_nm.XSs[0].warping2strain(TXS_nm.XSs[0].warping_functions[idx],1)
#     eps_foreground.interpolate(fem.Expression(eps_foreground_ufl,TXS_nm.XSs[0].V_sigma.element.interpolation_points()))
    
#     #set up the quadrature function for projecting strain:
#     mortar_mesh = TXS_nm.collisions[collision].mortar_mesh.msh
#     degree = 1
#     Qe = basix.ufl.quadrature_element(
#         mortar_mesh.topology.cell_name(), value_shape=(3,3),degree=degree)
#     V_quadrature = fem.functionspace(mortar_mesh, Qe)
#     cell_map0 = mortar_mesh.topology.index_map(mortar_mesh.topology.dim)
#     num_cells_on_proc = cell_map0.size_local + cell_map0.num_ghosts
#     cells_coarse = np.arange(num_cells_on_proc, dtype=np.int32)
#     nmmid_q = fem.create_interpolation_data(V_quadrature,
#                                                 TXS_nm.XSs[0].V_sigma,
#                                                 cells_coarse,
#                                                     padding=1e-14)
        
#     q_func = fem.Function(V_quadrature)
#     q_func.interpolate_nonmatching(eps_foreground, cells_coarse,interpolation_data=nmmid_q)
    
#     eps_C = ufl.TrialFunction(TXS_nm.collisions[collision].mortar_xs.V_sigma)
#     eps_vC = ufl.TestFunction(TXS_nm.collisions[collision].mortar_xs.V_sigma)

#     # Project fine function at quadrature points to coarse grid
#     a_coarse = ufl.inner(eps_C, eps_vC) * ufl.dx
#     L_coarse = ufl.inner(q_func, eps_vC)*ufl.dx
#     problem = fem.petsc.LinearProblem(a_coarse, L_coarse)
#     eps_A_proj = problem.solve()

#     diff = eps_A-eps_A_proj
#     H1semi_strain_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(diff,diff)*TXS_nm.collisions[(0,1)].dx)))
#     H1semi_strain_errors.append(H1semi_error)
# print(H1semi_strain_errors)

# print("B to mortar:")
# H1semi_strain_errors = []
# for idx in range(6):
#     #get the interpolated, then evaluated strain
#     TXS_nm.collisions[collision].PB.mult(TXS_nm.XSs[1].warping_functions[idx].x.petsc_vec,TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].x.petsc_vec)
#     wC_B = TXS_nm.collisions[collision].mortar_xs.warping_functions[idx].copy()
#     eps_B = TXS_nm.collisions[collision].mortar_xs.warping2strain(wC_B,1)

#     #get the strain function on the foreground mesh by interpolating an expression
#     eps_foreground = fem.Function(TXS_nm.XSs[1].V_sigma)
#     eps_foreground_ufl = TXS_nm.XSs[1].warping2strain(TXS_nm.XSs[1].warping_functions[idx],1)
#     eps_foreground.interpolate(fem.Expression(eps_foreground_ufl,TXS_nm.XSs[1].V_sigma.element.interpolation_points()))
    
#     #set up the quadrature function for projecting strain:
#     mortar_mesh = TXS_nm.collisions[collision].mortar_mesh.msh
#     degree = 1
#     Qe = basix.ufl.quadrature_element(
#         mortar_mesh.topology.cell_name(), value_shape=(3,3),degree=degree)
#     V_quadrature = fem.functionspace(mortar_mesh, Qe)
#     cell_map0 = mortar_mesh.topology.index_map(mortar_mesh.topology.dim)
#     num_cells_on_proc = cell_map0.size_local + cell_map0.num_ghosts
#     cells_coarse = np.arange(num_cells_on_proc, dtype=np.int32)
#     nmmid_q = fem.create_interpolation_data(V_quadrature,
#                                                 TXS_nm.XSs[1].V_sigma,
#                                                 cells_coarse,
#                                                     padding=1e-14)
        
#     q_func = fem.Function(V_quadrature)
#     q_func.interpolate_nonmatching(eps_foreground, cells_coarse,interpolation_data=nmmid_q)
    
#     eps_C = ufl.TrialFunction(TXS_nm.collisions[collision].mortar_xs.V_sigma)
#     eps_vC = ufl.TestFunction(TXS_nm.collisions[collision].mortar_xs.V_sigma)

#     # Project fine function at quadrature points to coarse grid
#     a_coarse = ufl.inner(eps_C, eps_vC) * ufl.dx
#     L_coarse = ufl.inner(q_func, eps_vC)*ufl.dx
#     problem = fem.petsc.LinearProblem(a_coarse, L_coarse)
#     eps_B_proj = problem.solve()

#     diff = eps_B-eps_B_proj
#     H1semi_strain_error = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(diff,diff)*TXS_nm.collisions[(0,1)].dx)))
#     H1semi_strain_errors.append(H1semi_error)
# print(H1semi_strain_errors)



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