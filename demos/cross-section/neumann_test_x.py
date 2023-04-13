from dolfinx import mesh
from dolfinx.fem import FunctionSpace,Constant,Function,form
from dolfinx.fem.petsc import LinearProblem,create_vector,assemble_matrix
from mpi4py import MPI
from ufl import TrialFunction,TestFunction,inner,grad,dx,ds,SpatialCoordinate
from petsc4py.PETSc import ScalarType,KSP,NullSpace
import numpy as np
from scipy.linalg import null_space

# Create mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, 2,2, mesh.CellType.quadrilateral)

# P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(domain, ("CG", 1))

u = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
g = -4 * x[1]
f = Constant(domain, ScalarType(-6))
a = (inner(12*grad(u), grad(v)) )*dx
L = f*v*dx - g*v*ds

# Compute solution
uh = Function(V)
# problem = LinearProblem(a,L)
# uh = problem.solve()

A = assemble_matrix(form(a))
A.assemble()
b=create_vector(form(L))
A_copy = A.copy()

print(A.getLocalSize())
A_d = A.convert('dense').getDenseArray()
print(A_d)
print('Matrix rank of: %i ' % np.linalg.matrix_rank(A_d))
print(null_space(A_d).T)
# #gram-schmidt orthogonalization
# q,r = np.linalg.qr(A_d)
# print('q: \n%s' % np.array2string(q,precision=2,formatter={'float_kind':lambda x: "%.2f" % x}))
# print('r: \n%s' % np.array2string(r,precision=2,formatter={'float_kind':lambda x: "%.2f" % x}))

# print('Rank of Null Space: %i ' % np.linalg.matrix_rank(null_space(A_d)))

# null_removed =q[:,0:-2]@r[0:-2,:]
# print('nullspace removed: \n%s' % np.array2string(null_removed,precision=2,formatter={'float_kind':lambda x: "%.2f" % x}))
# null_not_removed=q@r
# print('nullspace not removed: \n%s' % np.array2string(null_not_removed,precision=2,formatter={'float_kind':lambda x: "%.2f" % x}))


# print(np.linalg.norm(A_d-null_removed))
# print(np.linalg.norm(null_not_removed-null_removed))

#remove nullspace
# nullspace = NullSpace().create(constant=True, comm=domain.comm)
# A.setNullSpace(nullspace)
# print('b:')
print(b.getArray())
print('uh:')
print(uh.vector.array)
nullspace = A.getNullSpace()
nullspace.remove(uh.vector)
print(nullspace.vector.array)
# print('b (null removed):')
print(b.getArray())
print('uh (null removed):')
print(uh.vector.array)

A_new = A_copy.convert('dense').getDenseArray()
print('nullspace removed: \n%s' % np.array2string(A_new,precision=2,formatter={'float_kind':lambda x: "%.0f" % x}))
# print(A_new)
# print(np.linalg.norm(A_new-A_d))
print('Matrix rank of: %i ' % np.linalg.matrix_rank(A_new))
# print(null_space(A_new))
print('Rank of Null Space: %i ' % np.linalg.matrix_rank(null_space(A_new).T))


uvec = uh.vector
uvec.setUp()
ksp = KSP().create()
ksp.setType(KSP.Type.CG)
ksp.setTolerances(rtol=1e-15)
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b,uvec)

print(uvec.getArray())

# #plotting stuff
# import pyvista
# from dolfinx import plot
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True)
# plotter.view_xy()
# # if not pyvista.OFF_SCREEN:
# #     plotter.show()
# # else:
# #     figure = plotter.screenshot("fundamentals_mesh.png")
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["u"] = uh.x.array.real
# u_grid.set_active_scalars("u")
# u_plotter = pyvista.Plotter()
# u_plotter.add_mesh(u_grid, show_edges=True)
# u_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     u_plotter.show()

