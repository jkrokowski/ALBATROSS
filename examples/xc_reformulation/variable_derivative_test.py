from ufl import as_vector,sin,Coefficient,derivative,diff,inner,VectorElement,variable,TrialFunction,dx,TestFunction
from dolfinx import mesh,fem
from mpi4py import MPI
import numpy as np

#solution coeffs:
N = 1
W = .1
H = .1
domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
V_el = VectorElement("CG",domain.ufl_cell(),1,dim=6)
c1 = Coefficient(V_el)

# c.vector.array = np.concatenate([np.repeat(i,6) for i in range(domain.geometry.x.shape[0])])
# print(c.vector.array)
# c.vector.destroy()

w1 = c1*2

w1 = variable(w1)

F1 = w1**2
dF1 = diff(F1,w1)
print()


#not diff
V = fem.VectorFunctionSpace(domain,('CG',1),dim=6)
c_n = fem.Function(V)
c_n.vector.array = np.concatenate([np.arange(6)*i for i in range(domain.geometry.x.shape[0])])
print(c_n.vector.array)
c_n.vector.destroy()
c = TrialFunction(V)
q = TestFunction(V)
# w = variable(c**2)
F = inner(c_n,q)*dx
# F = 2*c
# dF = diff(F,c)
dF = derivative(F,c_n,c)
F2 = as_vector([c[i]**2 for i in range(6)])
c_v = variable(c)
dF2 = diff(F2,c_v)

A = fem.petsc.assemble_matrix(fem.form(dF))
A.assemble()
C = A.convert('dense')
print(C.getDenseArray())

v = TrialFunction(V)
u = TestFunction(V)
w = fem.Function(V)

f = (w**2)/2 * dx
F = derivative(f,w,v)
J = derivative(F,w,u)


print()