from dolfin import *

mesh = UnitSquareMesh(32, 32)

V = VectorFunctionSpace(mesh, "CG", 1)
u = Function(V)
print(type(u.vector()))
# Some velocity function
# u.interpolate(Expression(("-x[1]*x[1]", "x[0]*x[0]"), degree=4))

# Rigid body modes
rbms = [
    Constant((1.0, 0.0)),
    Constant((0.0, 1.0)),
    Expression(("-x[1]", "x[0]"), degree=1)
]
# List of functions to orthogonalise
v = list(interpolate(rbm, V) for rbm in rbms)

# GS Projection
def proj(u, v):
    res = assemble(inner(u, v)*dx)/assemble(inner(u, u)*dx) * u.vector()
    return res

# GS orthogonalisation
def ortho(v):
    xi = [None]*len(v)
    xi[0] = v[0]
    for j in range(1, len(xi)):
        xi[j] = Function(V, v[j].vector() - sum(proj(xi[i], v[j]) for i in range(j)))
    return xi

# Orthogonalised vectors
xi = ortho(v)

# Orthonormalised vector basis
e = [Function(V, xi_.vector()/assemble(inner(xi_, xi_)*dx)**0.5) for xi_ in xi]

print("orthonormalisation test:")
for i in range(len(xi)):
    for j in range(i+1):
        print(f"inner(e[{i}], e[{j}])*dx {assemble(inner(e[i], e[j])*dx)}")

u_star = u.vector() - sum(proj(e_, u) for e_ in e)
u_star = Function(V, u_star)

print(f"u norm {u.vector().norm('l2')}, u_star norm {u_star.vector().norm('l2')}")
print(f"orthogonalisation of u_star with rigid body modes test:")
for j in range(len(rbms)):
    print(f"(rbms[{j}], u_star) = {assemble(inner(u_star, rbms[j])*dx)}")