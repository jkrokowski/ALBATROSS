from ufl import variable,derivative,Constant,Coefficient,interval
from dolfinx import fem,mesh
from mpi4py import MPI
domain=mesh.create_unit_interval(MPI.COMM_WORLD,10)
cell = interval
c = variable(Constant(cell))

V = fem.FunctionSpace(domain, ('CG',1))
u = fem.Function(V)
du = fem.Function(V)

u = c 
deriv_u = derivative(u,c,du)
print(du)