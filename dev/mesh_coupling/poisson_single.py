try:
    from dolfinx import fem, mesh, plot, default_scalar_type
except:
    from petsc4py import PETSc
    default_scalar_type = PETSc.ScalarType   
from dolfinx.fem import (Constant, Function, FunctionSpace,
                         assemble_scalar, dirichletbc, form, locate_dofs_geometrical)
from dolfinx.fem.petsc import LinearProblem,assemble_vector,assemble_matrix
from dolfinx.mesh import create_rectangle
try:
    from dolfinx.plot import vtk_mesh
except:
    from dolfinx.plot import create_vtk_mesh as vtk_mesh

from mpi4py import MPI
from ufl import Measure,SpatialCoordinate, TestFunction, TrialFunction, dot, ds, dx, grad

import numpy as np
import pyvista

from dolfinx import geometry,cpp
import basix


w=1
h=0.1

#define first mesh 
mesh1 = create_rectangle(MPI.COMM_WORLD,((0,-h/2),(2*w, h/2)), [100, 5])
V1 = FunctionSpace(mesh1, ("Lagrange", 1))
u1 = TrialFunction(V1)
v1 = TestFunction(V1)

# #define second mesh
# mesh2 = create_rectangle(MPI.COMM_WORLD,((w-offset,-h/2),(2*w, h/2)), [50, 5])
# V2 = FunctionSpace(mesh2, ("Lagrange", 1))
# u2 = TrialFunction(V2)
# v2 = TestFunction(V2)

#mesh 1 problem
def u_bc1(x):
    print(x[0].shape)
    return np.zeros_like(x[0])
def boundary_D1(x):
    return np.logical_or(np.isclose(x[0], 0),np.isclose(x[0], 2*w))

dofs_D1 = locate_dofs_geometrical(V1, boundary_D1)
u_bc1 = Function(V1)
u_bc1.interpolate(u_bc1)
bc1 = dirichletbc(u_bc1, dofs_D1)

dx1 = Measure("dx", domain=mesh1)
ds1 = Measure("ds", domain=mesh1)
x1 = SpatialCoordinate(mesh1)
a1 = dot(grad(u1), grad(v1)) * dx1
# g = -1
f1 = Constant(mesh1, default_scalar_type(-6))
L1 = f1 * v1 * dx1 #- g * v1 * ds1

# #mesh 2 problem
# def u_bc2(x):
#     print(x[0].shape)
#     return np.zeros_like(x[0])
# def boundary_D2(x):
#     return np.isclose(x[0], 2*w)

# dofs_D2 = locate_dofs_geometrical(V2, boundary_D2)
# u_bc2 = Function(V2)
# u_bc2.interpolate(u_bc2)
# bc2 = dirichletbc(u_bc2, dofs_D2)

# dx2 = Measure("dx", domain=mesh2)
# ds2 = Measure("ds", domain=mesh2)
# x2 = SpatialCoordinate(mesh2)
# a2 = dot(grad(u2), grad(v2)) * dx2
# # x1 = SpatialCoordinate(mesh2)
# # g = -1
# f2 = Constant(mesh2, default_scalar_type(0.0))
# L2 = f2 * v2 * dx2 #- g * v1 * ds

# simple linear solve here:
# problem1 = LinearProblem(a1, L1, bcs=[bc1], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh1 = problem1.solve()
# problem2 = LinearProblem(a2, L2, bcs=[bc2], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# uh2 = problem2.solve()

#PETSc version of simple separate solves
#assemble system
A1 = assemble_matrix(form(a1),bcs=[bc1])
A1.assemble()
b1=fem.petsc.create_vector(form(L1))
with b1.localForm() as b_loc:
    b_loc.set(0)
assemble_vector(b1,form(L1))
fem.apply_lifting(b1,[form(a1)],bcs=[[bc1]])
b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b1,[bc1])

uh1 = Function(V1)

ksp1 = PETSc.KSP().create()
ksp1.setOperators(A1)
ksp1.setType('preonly')
pc=ksp1.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')
ksp1.setUp()

ksp1.solve(b1,uh1.vector)

# #KSP 2
# A2 = assemble_matrix(form(a2),bcs=[bc2])
# A2.assemble()
# b2=fem.petsc.create_vector(form(L2))
# with b2.localForm() as b_loc:
#     b_loc.set(0)
# assemble_vector(b2,form(L2))
# fem.apply_lifting(b2,[form(a2)],bcs=[[bc2]])
# b2.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# fem.set_bc(b2,[bc2])

# uh2 = Function(V2)

# ksp2 = PETSc.KSP().create()
# ksp2.setOperators(A2)
# ksp2.setType('preonly')
# pc=ksp2.getPC()
# pc.setType('lu')
# pc.setFactorSolverType('mumps')
# ksp2.setUp()

# ksp2.solve(b2,uh2.vector)

# found this code from online that does somthing similar (i think)
#https://fenicsproject.discourse.group/t/interpolation-matrix-with-non-matching-meshes/12204/13
def interpolation_matrix_nonmatching_meshes(V_1,V_0): # Function spaces from nonmatching meshes
    msh_0 = V_0.mesh
    msh_0.topology.dim
    msh_1 = V_1.mesh
    x_0   = V_0.tabulate_dof_coordinates()
    x_1   = V_1.tabulate_dof_coordinates()

    bb_tree         = geometry.BoundingBoxTree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions(bb_tree, x_1)
    cells           = []
    points_on_proc  = []
    index_points    = []
    colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

    for i, point in enumerate(x_1):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            index_points.append(i)
            
    index_points_   = np.array(index_points)
    points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
    cells_          = np.array(cells)

    ct      = cpp.mesh.to_string(msh_0.topology.cell_type)
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree(), basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), 2))

    for i in range(0, len(cells_)):
        geom_dofs  = msh_0.geometry.dofmap.links(cells_[i])
        x_ref[i,:] = msh_0.geometry.cmap.pull_back([points_on_proc_[i,:]], msh_0.geometry.x[geom_dofs])

    basis_matrix = element.tabulate(0, x_ref)[0,:,:,0]

    cell_dofs         = np.zeros((len(x_1), len(basis_matrix[0,:])))
    basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0,:])))


    for nn in range(0,len(cells_)):
        cell_dofs[index_points_[nn],:] = V_0.dofmap.cell_dofs(cells_[nn])
        basis_matrix_full[index_points_[nn],:] = basis_matrix[nn,:]

    cell_dofs_ = cell_dofs.astype(int) ###### REDUCE HERE

    # [JEF] I = np.zeros((len(x_1), len(x_0)), dtype=complex)
    # make a petsc matrix here instead of np- 
    # for Josh: probably more efficient ways to do this 
    I = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    I.setSizes((len(x_1), len(x_0)))
    I.setUp()
    for i in range(0,len(x_1)):
        for j in range(0,len(basis_matrix[0,:])):
            # [JEF] I[i,cell_dofs_[i,j]] = basis_matrix_full[i,j]
            I.setValue(i,cell_dofs_[i,j],basis_matrix_full[i,j])

    return I 

# M1 = interpolation_matrix_nonmatching_meshes(V1,V2)
# M2 = interpolation_matrix_nonmatching_meshes(V2,V1)
# M1.assemble()
# M2.assemble()

#interpolate solution from 

#visualize
pyvista_cells1, cell_types1, geometry1 = vtk_mesh(V1)
grid1 = pyvista.UnstructuredGrid(pyvista_cells1, cell_types1, geometry1)
grid1.point_data["u1"] = uh1.x.array
grid1.set_active_scalars("u1")

# pyvista_cells2, cell_types2, geometry2 = vtk_mesh(V2)
# grid2 = pyvista.UnstructuredGrid(pyvista_cells2, cell_types2, geometry2)
# grid2.point_data["u2"] = uh2.x.array
# # grid2.set_active_scalars("u")

plotter = pyvista.Plotter()
plotter.add_text("uh", position="upper_edge", font_size=14, color="black")
# plotter.add_mesh(grid2, show_edges=True,opacity=.5)
plotter.add_mesh(grid1, show_edges=True,opacity=.5)

plotter.view_xy()

if not pyvista.OFF_SCREEN:
    plotter.show()
else:
    figure = plotter.screenshot("neumann_dirichlet.png")