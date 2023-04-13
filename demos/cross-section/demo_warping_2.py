from dolfin import *
import numpy as np`
# Create mesh
mesh = UnitSquareMesh.create(4, 4, CellType.Type.quadrilateral)

# Build function space with Lagrange multiplier
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, P1 * R)

# Define variational problem
(w0, c) = TrialFunction(W)
(v, d) = TestFunctions(W)
x=SpatialCoordinate(mesh)
# #shear centre for symmetric shape
# xs=0.5
# ys=0.5
n = FacetNormal(mesh)
g = (x[0])*n[1] + (x[1])*n[0]
f = Constant(0.0)
a = (inner(grad(w0), grad(v)) + c*v + w0*d)*dx
L = f*v*dx + g*v*ds

# Compute warping solution
u = Function(W)
solve(a == L, u)
(w0, c) = u.split()

# Save solution in VTK format
file = File("warping_solution.pvd")
file << w0

# #Compute derivative of solution
# DW = VectorFunctionSpace(mesh,'Lagrange',2)
# grad_w = project(grad(w),DW)

# # Save solution in VTK format
# file = File("grad_w_solution.pvd")
# file << grad_w

# dwdx,dwdy = grad_w.split(deepcopy=True)

# file = File("grad_w_x_solution.pvd")
# file << dwdx
# file = File("grad_w_y_solution.pvd")
# file << dwdy

print(w0.vector().get_local())
# print(grad_w.vector().get_local())
# print(ux.vector().get_local())
# print(uy.vector().get_local())

#=========== COMPUTE TORSIONAL CONSTANT ============#
#computing the torsional constant requires the computation of both the bending MOI
#   ,as well as derivatives of the warping function

dx = Measure("dx", domain=mesh)

A = assemble(Constant(1.0)*dx)
x_G = assemble(x[0]*dx) / A
y_G = assemble(x[1]*dx) / A

print(x_G)
print(y_G)
print()

Ix= assemble(((x[1]-y_G)**2)*dx)
Iy = assemble(((x[0]-x_G)**2)*dx)
Ixy = assemble(((x[0]-x_G)*(x[1]-y_G))*dx)

Ixw = assemble(((x[1]-y_G)*w0)*dx)
Iyw = assemble(((x[0]-x_G)*w0)*dx)

# Wo = assemble((dwdx**2+dwdy**2)*dx)
print(Ix)
print(Iy)
print(Ixy)
print(Ixw)
print(Iyw)
print()

xs = (Iy*Ixw - Ixy*Iyw)/(Ix*Iy-Ixy**2)
ys = (Ix*Iyw - Ixy*Ixw)/(Iy*Ix-Ixy**2)

print(xs)
print(ys)

w = w0 + ys*x[0] + xs*x[1]

Is = assemble((((x[1]-ys)**2)+((x[0]-xs)**2))*dx)
# print(grad(w))
W0 = assemble((grad(w)**2)*dx)

print(Is)
print(W0)

J = Is - W0 

print(J)

# K = Ix + Iy + Kwx + Kwy
# print(K)



