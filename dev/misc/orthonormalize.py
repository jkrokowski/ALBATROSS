from dolfinx import fem,mesh
from mpi4py import MPI
from ufl import inner,Measure,as_vector
import numpy as np
from petsc4py import PETSc

nx=10
ny=10
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1))
u = fem.Function(V)

u2 = fem.Function(V,u.x)
dx = Measure("dx",domain=domain)

# Some velocity function
class MyVelocity:
    def eval(self,x):
        return (-x[1]*x[1],x[0]*x[0])
f1 = MyVelocity()

# fem.Expression(("-x[1]*x[1]", "x[0]*x[0]"),V.element.interpolation_points())
u.interpolate(f1.eval)
bs = V.dofmap.bs
def x_translate(x):
    values = np.zeros((2,x.shape[1]),dtype=PETSc.ScalarType)
    values[0] = fem.Constant(domain,1.0)
    return values
    # return np.vstack([np.full(x.shape[1],fem.Constant(domain,1.0)),
    #                  np.full(x.shape[1],fem.Constant(domain,0.0)) ])
def y_translate(x):
    values = np.zeros((2,x.shape[1]),dtype=PETSc.ScalarType)
    values[1] = fem.Constant(domain,1.0)
    return values
    # return np.vstack([np.full(x.shape[1],fem.Constant(domain,0.0)),
    #                  np.full(x.shape[1],fem.Constant(domain,1.0)) ])
def rotation(x):
    values = np.zeros((2,x.shape[1]),dtype=PETSc.ScalarType)
    values[0] = -x[1]
    values[1] = x[0]
    return values
    # return (-x[1],x[0])

# Rigid body modes
rbms = [
    x_translate,
    y_translate,
    rotation
]
# List of functions to orthogonalise
ux = fem.Function(V)
uy = fem.Function(V)
ur = fem.Function(V)
ux.interpolate(rbms[0])
uy.interpolate(rbms[1])
ur.interpolate(rbms[2])

# ux.vector.assemble()
# uy.vector.assemble()
# ur.vector.assemble()

v = list((ux,uy,ur))
print(sum(ux.vector.array-ux.x.array))
# GS Projection
def proj(u, v):
    print(sum(u.vector.array-u.x.array))
    res = fem.assemble_scalar(fem.form(inner(u, v)*dx))/fem.assemble_scalar(fem.form(inner(u, u)*dx)) * u.x.array
    # res = fem.assemble_scalar(fem.form(inner(u, v)*dx))/fem.assemble_scalar(fem.form(inner(u, u)*dx)) * u.x.array
    return res

# GS orthogonalisation
def ortho(v):
    xi = [None]*len(v)
    xi[0] = v[0]
    for j in range(1, len(xi)):
        # xi[j] = fem.Function(V)
        # sum(proj(xi[i], v[j]) for i in range(j))
        # xi[j].x = v[j].x - sum(proj(xi[i], v[j]) for i in range(j))
        xi[j] = fem.Function(V)
        # # xi[j] = v[j].vector - sum(proj(xi[i], v[j]) for i in range(j))
        # xi[j].x.assemble()
        print(sum(u.vector.array-u.x.array))

        xi[j].vector.array = v[j].vector.array - sum(proj(xi[i], v[j]) for i in range(j))
    return xi

# Orthogonalised vectors
xi = ortho(v)

# Orthonormalised vector basis
# e = [fem.Function(V, xi_.vector.array/fem.assemble_scalar(inner(xi_, xi_)*dx)**0.5) for xi_ in xi]
e = [fem.Function(V)]*len(v)
for i,xi_ in enumerate(xi):
    e[i].vector.array = xi_.vector.array/fem.assemble_scalar(fem.form(inner(xi_, xi_)*dx))**0.5
    e[i].vector.assemble()

print("orthonormalisation test:")
for i in range(len(xi)):
    for j in range(i+1):
        print(f"inner(e[{i}], e[{j}])*dx {fem.assemble_scalar(fem.form(inner(e[i], e[j])*dx))}")

u_star = fem.Function(V)
print(sum(u_star.vector.array-u_star.x.array))

u_star.vector.array = u.vector.array - sum(proj(e_, u) for e_ in e)

print(f"u norm {u.vector.norm(2)}, u_star norm {u_star.vector.norm(2)}")
print(f"orthogonalisation of u_star with rigid body modes test:")
for j in range(len(v)):
    print(f"(rbms[{j}], u_star) = {fem.assemble_scalar(fem.form(inner(u_star, v[j])*dx))}")