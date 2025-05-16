#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot
import pyvista
import ALBATROSS
import numpy as np
from petsc4py import PETSc

# np.set_printoptions(precision=3)

N = 7
offset = 2

# m1,n1 = N+offset,2*(N+offset)
# m2,n2 = N-offset,2*(N-offset)
m1,n1 = N-offset,2*(N-offset)
m2,n2 = N+offset,2*(N+offset)

H = 1
W = 1
t_overlap = 0

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= H
mesh_0.geometry.x[:, 0] *= W/2+t_overlap
mesh_0.geometry.x[:, 0] += (W/2-t_overlap)/2

mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= H
mesh_1.geometry.x[:, 0] *= W/2+t_overlap
mesh_1.geometry.x[:, 0] -= (W/2-t_overlap)/2

mesh_1.name = 'w'

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

XS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e4)

XS_nm.plot_meshes()

#nearest neighbor interpolation from coarse to fine:
fine_boundary_nodes=mesh.locate_entities_boundary(XSs[0].msh,0,lambda x: np.isclose(x[0],0))
coarse_boundary_nodes=mesh.locate_entities_boundary(XSs[1].msh,0,lambda x: np.isclose(x[0],0))
interp_mat = ALBATROSS.nonmatching_utils.get_nn_interpolation_matrix(XSs[1].msh.geometry.x[list(coarse_boundary_nodes),:],XSs[0].msh.geometry.x)
interp_mat_expanded = np.zeros((XSs[0].msh.geometry.x.shape[0],XSs[1].msh.geometry.x.shape[0]))
interp_mat_expanded[:,list(coarse_boundary_nodes)]=interp_mat

interp_mat2 = ALBATROSS.nonmatching_utils.get_nn_interpolation_matrix(XSs[0].msh.geometry.x[list(fine_boundary_nodes),:],XSs[1].msh.geometry.x)
interp_mat2_expanded = np.zeros((XSs[1].msh.geometry.x.shape[0],XSs[0].msh.geometry.x.shape[0]))
interp_mat2_expanded[:,list(fine_boundary_nodes)]=interp_mat2

# Create PETSc matrix
n, m = interp_mat_expanded.shape
A_petsc = PETSc.Mat().createAIJ(size=(n, m)) 
A_petsc.setUp()
# Fill PETSc matrix
for i in range(n):
    A_petsc.setValues(i, range(m), interp_mat_expanded[i, :])
A_petsc.assemble()
interp_mat_expanded_petsc = ALBATROSS.nonmatching_utils.permute_and_expand_matrix(XSs[0].V,XSs[1].V,A_petsc,mixed=True)

# Create PETSc matrix
n, m = interp_mat2_expanded.shape
A_petsc2 = PETSc.Mat().createAIJ(size=(n, m)) 
A_petsc2.setUp()
# Fill PETSc matrix
for i in range(n):
    A_petsc2.setValues(i, range(m), interp_mat2_expanded[i, :])
A_petsc2.assemble()
interp_mat2_expanded_petsc = ALBATROSS.nonmatching_utils.permute_and_expand_matrix(XSs[1].V,XSs[0].V,A_petsc2,mixed=True)

XS_nm._assemble_system_mats()

XS_nm.adjacency=np.array([[1,1],[0,1]])

# XS_nm._construct_coupled_system_matrix_nn()
# A01= PETSc.Mat()
# A01.mat.createAIJ([XSs[0].system_mat.getSize()[0],XSs[1].system_mat.getSize()[0][1]])
A01 = interp_mat_expanded_petsc.duplicate()
A01.scale(-XS_nm.pen)

pen_vec = PETSc.Vec().create()
pen_vec.setSizes(XSs[0].system_mat.getSize()[0])
pen_vec.setFromOptions()
indices = ALBATROSS.nonmatching_utils.convert_petsc_to_numpy(interp_mat_expanded_petsc).nonzero()[0]
for idx in indices:
    pen_vec.setValue(idx,XS_nm.pen)
pen_term = PETSc.Mat().createAIJ(XSs[0].system_mat.getSize())
pen_term.assemble()
pen_term.setDiagonal(pen_vec)
pen_term.assemble()
XSs[0].system_mat.axpy(1.0,pen_term)

A10= PETSc.Mat()
A10.createAIJ([XSs[1].system_mat.getSize()[0],XSs[0].system_mat.getSize()[1]])
A10.assemble()

# A10 = interp_mat2_expanded_petsc.duplicate()
# A10.scale(-XS_nm.pen)

# pen_vec2 = PETSc.Vec().create()
# pen_vec2.setSizes(XSs[1].system_mat.getSize()[0])
# pen_vec2.setFromOptions()
# indices = ALBATROSS.nonmatching_utils.convert_petsc_to_numpy(interp_mat2_expanded_petsc).nonzero()[0]
# for idx in indices:
#     pen_vec2.setValue(idx,XS_nm.pen)
# pen_term2 = PETSc.Mat().createAIJ(XSs[1].system_mat.getSize())
# pen_term2.assemble()
# pen_term2.setDiagonal(pen_vec2)
# pen_term2.assemble()
# XSs[1].system_mat.axpy(1.0,pen_term2)

A_list = [[XSs[0].system_mat,A01],[A10,XSs[1].system_mat]]
A = PETSc.Mat()
A.createNest(A_list)
A.assemble()
XS_nm.system_mat = A

#use QR factorization to get null modes:
XS_nm._get_modes()
print("null modes found!")

#need to "decouple" the modes
XS_nm._decouple_modes()

#map elastic solutions to construct warping functions
XS_nm._compute_xs_stiffness_matrix(correction=None)

# XS_nm.get_xs_stiffness_matrix(correction=None)

XS_nm.plot_warping_fxns()

# for i in range(3):
#     for j in range(3):
#         XS_nm.plot_warping_strains(component=(i,j))

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(XS_nm.K)

points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)

#compute difference between matrix entries of beam constituitive matrix
abs_diff = XS_nm.K - squareXS.K
rel_diff = abs_diff/squareXS.K

print("Total difference:")
print(abs_diff)

print("Relative Difference:")
print(rel_diff)

print("Maximum Absolute Difference of Diagonal Entries:")
print(np.max(np.abs(np.diag(abs_diff))))

print("Maximum Relative Difference of Diagonal Entries:")
print(np.max(np.abs(np.diag(rel_diff))))

print("Frobenius Norm of Absolute Difference:")
print(np.linalg.norm(abs_diff))

print("Relative Frobenius Norm of Absolute Difference:")
print(np.linalg.norm(abs_diff)/np.linalg.norm(squareXS.K))