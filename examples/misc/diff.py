from ufl import Measure,variable,diff,dot,as_vector,as_matrix,Constant,replace,Coefficient,interval,SpatialCoordinate
from dolfinx import fem,mesh
from mpi4py import MPI
# from petsc4py import PETSc

domain=mesh.create_unit_interval(MPI.COMM_WORLD,10)
cell = interval
# c = variable(Constant(cell))

c1 = variable(Constant(cell))
c2 = variable(Constant(cell))
c3 = variable(Constant(cell))
c4= variable(Constant(cell))
c5 = variable(Constant(cell))
c6 = variable(Constant(cell))
vars = [c1,c2,c3,c4,c5,c6]
c = as_vector(vars)

x = SpatialCoordinate(domain)
print(c.ufl_shape)
print(x[0].ufl_shape)
# u = dot(c,x)
u = c*x[0]
print(u.ufl_shape)

dudc = diff(u,c1)
print(dudc)
print(dudc.ufl_shape)

V = fem.FunctionSpace(domain,('CG',1))
ubar = fem.Function(V)
dx = Measure('dx',domain)
u_exp = u*ubar

#take derivative of expression with respect to a variable:
dudc1 = diff(u_exp,c1) #diff() takes partial derivatives
print(dudc1)
print(dudc1.ufl_shape)
dudc1 = replace(dudc1,{ubar:fem.Function(V,name='ubar')})
print(dudc)
print(dudc.ufl_shape)
dudc2 = diff(u_exp,c2)
print(dudc2)
dudc2 = replace(dudc2,{ubar:fem.Function(V,name='ubar')})
print(dudc2)

dudc1dc2 = diff(diff(u_exp,c1),c2)
print(dudc1dc2)
print(dudc1dc2.ufl_shape)

#construct Jacobian matrix of expression:
J = as_matrix([list(diff(u_exp,var)) for var in vars])
# J = as_matrix([diff(u_exp,var) for var in vars])
print(J.ufl_shape)







