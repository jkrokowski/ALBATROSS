from dolfinx import fem,mesh
from mpi4py import MPI
from ufl import inner,Measure,as_vector,SpatialCoordinate
from petsc4py import PETSc

nx=32
ny=32
domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, mesh.CellType.triangle)

V = fem.VectorFunctionSpace(domain, ("CG", 1),dim=3)
u = fem.Function(V)
x = SpatialCoordinate(domain)

#Some random expression
u_expr = fem.Expression(as_vector([-x[1]*x[1],x[0]*x[0],x[0]*x[1]]),V.element.interpolation_points())
u.interpolate(u_expr)

dx = Measure("dx",domain=domain)

#Rigid Body Modes expression (2D)
rbms = [
    fem.Expression(fem.Constant(domain,PETSc.ScalarType((1.0,0.0,0.0))),V.element.interpolation_points()),
    fem.Expression(fem.Constant(domain,PETSc.ScalarType((0.0,1.0,0.0))),V.element.interpolation_points()),
    fem.Expression(fem.Constant(domain,PETSc.ScalarType((0.0,0.0,1.0))),V.element.interpolation_points()),
    fem.Expression(as_vector([-x[1],x[0],0]),V.element.interpolation_points()),
    fem.Expression(as_vector([0,0,x[1]]),V.element.interpolation_points()),
    fem.Expression(as_vector([0,0,-x[0]]),V.element.interpolation_points())
]

# List of functions to orthogonalise
# vx = fem.Function(V)
# vy = fem.Function(V)
# vr = fem.Function(V)
# vx.interpolate(rbms[0])
# vy.interpolate(rbms[1])
# vr.interpolate(rbms[2])
v = list()
for i,rbm in enumerate(rbms):
    v_fxn = fem.Function(V)
    v_fxn.interpolate(rbm)
    v.append(v_fxn)
# v = list((vx,vy,vr))

# GS Projection
def proj(u, v):
    res = fem.assemble_scalar(fem.form(inner(u, v)*dx))/fem.assemble_scalar(fem.form(inner(u, u)*dx)) * u.vector.array
    return res

# GS orthogonalisation
def ortho(v):
    xi = [None]*len(v)
    xi[0] = v[0]
    for j in range(1, len(xi)):
        xi[j] = fem.Function(V)
        xi[j].vector.array = v[j].vector.array - sum(proj(xi[i], v[j]) for i in range(j))
    return xi

# Orthogonalised vectors
xi = ortho(v)

# Orthonormalised vector basis
e = [fem.Function(V) for i in range(len(v))]
for i,xi_ in enumerate(xi):
    e[i].vector.array = xi_.vector.array/fem.assemble_scalar(fem.form(inner(xi_, xi_)*dx))**0.5

print("orthonormalisation test:")
for i in range(len(xi)):
    for j in range(i+1):
        print(f"inner(e[{i}], e[{j}])*dx {fem.assemble_scalar(fem.form(inner(e[i], e[j])*dx))}")

u_star = fem.Function(V)
u_star.vector.array = u.vector.array - sum(proj(e_, u) for e_ in e)

print(f"u norm {u.vector.norm(2)}, u_star norm {u_star.vector.norm(2)}")
print("orthogonalisation of u_star with rigid body modes test:")
for j in range(len(v)):
    print(f"(rbms[{j}], u_star) = {fem.assemble_scalar(fem.form(inner(u_star, v[j])*dx))}")