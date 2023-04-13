# Cross Sectional Analysis using the Principle of Minimum Total Complementary Energy
# ================

from mpi4py import MPI
from dolfinx import mesh,plot
from dolfinx.fem import Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import (FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
from scipy.linalg import null_space

# Create mesh and define function space
N = 2
domain = mesh.create_unit_square(MPI.COMM_WORLD,N,N, mesh.CellType.quadrilateral)
# domain = mesh.create_unit_square(MPI.COMM_WORLD,N,N, mesh.CellType.triangle)
Ue = VectorElement("CG",domain.ufl_cell(),1,dim=3)
V = FunctionSpace(domain, MixedElement([Ue,Ue,Ue,Ue]))

#displacement and test functions
u = TrialFunction(V)
v = TestFunction(V)

#displacement coefficient functions
ubar,uhat,utilde,ubreve=split(u)

#displacement coefficient functions
vbar,vhat,vtilde,vbreve=split(v)

#geometric dimension
d = 3
#indices
i,j,k,l=indices(4)
a,B = indices(2)
#integration measures
dx = Measure("dx",domain=domain)
ds = Measure("ds",domain=domain)
#spatial coordinate and facet normals
x = SpatialCoordinate(domain)
n = FacetNormal(domain)

#material parameters
E = 100 #70e9
nu = 0.2

# _lam = (E*nu)/((1+nu)*(1-2*nu))
# mu = E/(2*(1+nu))

#d x d indentity matrix
delta = Identity(d)

# C_ufl = (E/(1+nu))*(0.5*(delta_ik*delta_jl + delta_jk*delta_il) + (nu/(1-2*nu))*delta_ij*delta_kl)
#TODO: rewrite with lame parameters for clarity
C = as_tensor((E/(1+nu))*(0.5*(delta[i,k]*delta[j,l] \
                + delta[j,k]*delta[i,l]) \
                    + (nu/(1-2*nu))*delta[i,j]*delta[k,l])  ,(i,j,k,l))

#sub-tensors of stiffness tensor
Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
Ci1kB = as_tensor([[[C[i, 0, k, l] for l in range(1, 3)]
              for k in range(d)] for i in range(d)])
Ciak1 = as_tensor([[[C[i, j, k, 0] for k in range(d)]
              for j in range(1,3)] for i in range(d)])
CiakB = as_tensor([[[[C[i, j, k, l] for l in range(1,3)]
              for k in range(d)] for j in range(1,3)] 
              for i in range(d)])

#partial derivatives of displacement:
ubar_B = as_tensor([Dx(ubar,B) for B in range(2)]).T
uhat_B = as_tensor([Dx(uhat,B) for B in range(2)]).T
utilde_B = as_tensor([Dx(utilde,B) for B in range(2)]).T
ubreve_B = as_tensor([Dx(ubreve,B) for B in range(2)]).T


#partial derivatives of shape fxn:
vbar_a = as_tensor([Dx(vbar,a) for a in range(2)]).T
vhat_a = as_tensor([Dx(vhat,a) for a in range(2)]).T
vtilde_a = as_tensor([Dx(vtilde,a) for a in range(2)]).T
vbreve_a = as_tensor([Dx(vbreve,a) for a in range(2)]).T

#traction free boundary conditions
Tbar = as_tensor(Ciak1[i,a,k]*uhat[k]*n[a],(i)) \
     + as_tensor(CiakB[i,a,k,B]*ubar_B[k,B]*n[a],(i))
That = 2*as_tensor(Ciak1[i,a,k]*utilde[k]*n[a],(i)) \
     + as_tensor(CiakB[i,a,k,B]*uhat_B[k,B]*n[a],(i))
Ttilde = 3*as_tensor(Ciak1[i,a,k]*ubreve[k]*n[a],(i)) \
     + as_tensor(CiakB[i,a,k,B]*utilde_B[k,B]*n[a],(i))
Tbreve = as_tensor(CiakB[i,a,k,B]*ubreve_B[k,B]*n[a],(i))

# equation 1,2,3
L1 = inner(2*Ci1k1,(outer(vtilde,utilde)))*dx \
     + inner(Ci1kB,(outer(vhat,uhat_B)))*dx \
     - inner(Ciak1,(outer(vhat_a,uhat)))*dx \
     - inner(CiakB,outer(vbar_a,ubar_B))*dx \
     + inner(Tbar,vbar)*ds

# equation 4,5,6
L2 = inner(6*Ci1k1,(outer(vbreve,ubreve)))*dx\
     + inner(2*Ci1kB,(outer(vtilde,utilde_B)))*dx \
     - inner(2*Ciak1,(outer(vtilde_a,utilde)))*dx \
     - inner(CiakB,outer(vhat_a,uhat_B,))*dx\
     + inner(That,vhat)*ds

# equation 7,8,9
L3 = inner(3*Ci1kB,(outer(vbreve,ubreve_B)))*dx \
     - inner(3*Ciak1,(outer(vbreve_a,ubreve)))*dx \
     - inner(CiakB,outer(vbreve_a,utilde_B))*dx\
     + inner(Ttilde,vtilde)*ds

#equation 10,11,12
L4= -inner(CiakB,outer(vbreve_a,ubreve_B))*dx\
     + inner(Tbreve,vbreve)*ds

Residual = L1+L2+L3+L4

LHS = lhs(Residual)
# RHS = rhs(Residual)
RHS = Constant(domain,0.0)*v[1]*dx

#assemble system matrices
A = assemble_matrix(form(LHS))
A.assemble()
b=create_vector(form(RHS))
with b.localForm() as b_loc:
      b_loc.set(0)
assemble_vector(b,form(RHS))

print(A.getSize())
print(b.getSize())
m,n=A.getSize()

Anp = A.getValues(range(m),range(n))
print("rank:")
print(np.linalg.matrix_rank(Anp))

nullspace = null_space(Anp)
print(nullspace.shape)

print("stiffness matrix:")
print(Anp)
Usvd,sv,Vsvd = np.linalg.svd(Anp)
# nullspace = A.getNullSpace()
# print(nullspace.test(A))

print(sv[-13:-1])

exit()
# Solution Function
uh = Function(V)

# Create Krylov solver
solver = PETSc.KSP().create(A.getComm())
solver.setOperators(A)

# Create vector that spans the null space
nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
A.setNullSpace(nullspace)

# orthogonalize b with respect to the nullspace ensures that 
# b does not contain any component in the nullspace
nullspace.remove(b)

# Finally we are able to solve our linear system :
solver.solve(b,uh.vector)


# #======= COMPUTE TORSIONAL CONSTANT =========#
# # This requires computing the shear center and computing the warping 
# # function based on the location of that shear center

# #area
# A = assemble_scalar(form(1.0*dx))
# print(A)
# #first moment of area / A (gives the centroid)
# x_G = assemble_scalar(form(x[0]*dx)) / A
# y_G = assemble_scalar(form(x[1]*dx)) / A
# print(x_G)
# print(y_G)
# #second moment of area 
# Ixx= assemble_scalar(form(((x[1]-y_G)**2)*dx))
# Iyy = assemble_scalar(form(((x[0]-x_G)**2)*dx))
# print(Ixx)
# print(Iyy)

# Ixy = assemble_scalar(form(((x[0]-x_G)*(x[1]-y_G))*dx))
# Iwx = assemble_scalar(form(((x[1]-y_G)*uh)*dx))
# Iwy = assemble_scalar(form(((x[0]-x_G)*uh)*dx))
# print("product and warping")
# print(Ixy)
# print(Iwx)
# print(Iwy)

# xs = (Iyy*Iwy-Ixy*Iwx)/(Ixx*Iyy-Ixy**2)
# ys = -(Ixx*Iwx-Ixy*Iwy)/(Ixx*Iyy-Ixy**2)
# print("shear center:")
# print(xs)
# print(ys)

# w1_expr = Expression(-ys*x[0]+xs*x[1],V.element.interpolation_points())
# w1 = Function(V)
# w1.interpolate(w1_expr)

# w = Function(V)
# w.x.array[:] = uh.x.array[:] + w1.x.array[:]

# #compute derivatives of warping function
# W = VectorFunctionSpace(domain, ("CG", 1))
# grad_uh = grad(w)
# grad_uh_expr = Expression(grad_uh, W.element.interpolation_points())
# grad_u = Function(W)
# grad_u.interpolate(grad_uh_expr)

# #separate out partial derivatives
# dudx_expr = Expression(grad_uh[0], V.element.interpolation_points())
# dudx = Function(V)
# dudx.interpolate(dudx_expr)

# dudy_expr = Expression(grad_uh[1], V.element.interpolation_points())
# dudy = Function(V)
# dudy.interpolate(dudy_expr)

# Kwx = assemble_scalar(form(((x[0]-x_G)*dudy)*dx))
# Kwy = assemble_scalar(form(((x[1]-y_G)*dudx)*dx))

# print(Kwx)
# print(Kwy)

# K = Ixx+Iyy+Kwx-Kwy
# print("Torsional Constant:")
# print(K)

# #======= PLOT WARPING FUNCTION =========#
# # tdim = domain.topology.dim
# # topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# # plotter = pyvista.Plotter()
# # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# # u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# # u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# # u_grid.point_data["u"] = uh.x.array.real
# # u_grid.set_active_scalars("u")
# # plotter.add_mesh(u_grid.warp_by_scalar("u",factor=2), show_edges=True)
# # plotter.view_vector((-0.25,-1,0.5))
# # if not pyvista.OFF_SCREEN:
# #     plotter.show()

# # #plot warping
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["w"] = w.x.array.real
# u_grid.set_active_scalars("w")
# plotter.add_mesh(u_grid.warp_by_scalar("w",factor=2), show_edges=True)
# plotter.view_vector((-0.25,-1,0.5))
# if not pyvista.OFF_SCREEN:
#     plotter.show()

# #======= PLOT GRAD OF WARPING FUNCTION =========#
# grad_plotter = pyvista.Plotter()
# grad_plotter.add_mesh(grid, style="wireframe", color="k")

# values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
# values[:, :len(grad_u)] = grad_u.x.array.real.reshape((geometry.shape[0], len(grad_u)))

# function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# function_grid["grad_u"] = values
# glyphs = function_grid.glyph(orient="grad_u", factor=1)

# grad_plotter.add_mesh(glyphs)
# grad_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     grad_plotter.show()