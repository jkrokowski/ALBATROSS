from ufl import indices,as_matrix,as_tensor,dx,TestFunction,TrialFunction,outer
from dolfinx import mesh
from mpi4py import MPI
from dolfinx.fem import FunctionSpace

domain = mesh.create_unit_square(MPI.COMM_WORLD, 2,2, mesh.CellType.quadrilateral)
V = FunctionSpace(domain, ("CG", 1))

d=3
i,j = indices(2)

#displacement and test function
v = TestFunction(V)
u= as_tensor([TrialFunction(V) for i in range(d)])

B = as_matrix([[a+b for b in range(2)] for a in range(2)])

# F = as_tensor(2*B[i,j]*as_tensor([outer(u[i],v)*dx for i in range(d)]))

# F = 2*B[0,j]*as_tensor([outer(u[j],v) for j in range(d)])*dx
F = as_tensor(2*B[0,j]*as_tensor([outer(u[j],v) for j in range(d)])[j])
# F = as_tensor(2*B[i,j]*as_tensor([outer(u[j],v) for j in range(d)]),(i))
print()
print(as_tensor([outer(u[j],v) for j in range(d)]).ufl_shape)
print(F.ufl_shape)

F_int = F*dx
F_int += as_tensor(2*B[1,j]*as_tensor([outer(u[j],v) for j in range(d)])[j])*dx

print(F_int)