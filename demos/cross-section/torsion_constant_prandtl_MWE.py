from dolfinx import mesh
from dolfinx.fem import (FunctionSpace,Constant,Function,form,dirichletbc,
                         locate_dofs_geometrical,assemble_scalar)
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import TrialFunction,TestFunction,inner,grad,dx,ds,SpatialCoordinate
from petsc4py.PETSc import ScalarType
import numpy as np
'''
Computing the torsional constant of a square using a Prandtl Stress function 

'''

# Create mesh
square = mesh.create_unit_square(MPI.COMM_WORLD, 20,20, mesh.CellType.quadrilateral)
domain = square
V = FunctionSpace(domain, ("CG", 1))

phi = TrialFunction(V)
v = TestFunction(V)

x = SpatialCoordinate(domain)
#G and Theta are virutal loads that can be anything within numerical reason
# their purpose is to perturb the solution away from a "boring" LaPlace equation
# with the boundary equal to 0 (which would just be 0 everywhere)
G = 10
Theta = 0.1
f = Constant(domain, ScalarType(2*G*Theta))
a = (inner(grad(phi), grad(v)) )*dx
L = f*v*dx 

def left(x):
    return np.isclose(x[0], 0)
def right(x):
    return np.isclose(x[0],1)
def top(x):
    return np.isclose(x[1],0)
def bot(x):
    return np.isclose(x[1],1)

bc1 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, left),V)
bc2 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, right),V)
bc3 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, top),V)
bc4 = dirichletbc(ScalarType(0), locate_dofs_geometrical(V, bot),V)

# Compute solution
phih = Function(V)
problem = LinearProblem(a,L,bcs=[bc1,bc2,bc3,bc4])
phih = problem.solve()

It = (2/(G*Theta))*assemble_scalar(form(phih*dx))
print("torsional constant:")
print(It)
