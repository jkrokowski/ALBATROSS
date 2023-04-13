from dolfin import *
import numpy as np

#solve homogenous torsion problem with 2x Dirichlet BC
mesh2 = UnitIntervalMesh(10)
print(mesh2.ufl_cell())
P1 = FiniteElement("Lagrange", mesh2.ufl_cell(), 1)
Theta = FunctionSpace(mesh2,P1)

GJ = Constant(1.0e12)
f = Constant(0.0) #distributed torque (doesn't exist in the problem)

t = TrialFunction(Theta)
v_t = TestFunction(Theta)
#define Boundary condition
tol = 1e-8
theta_0 = Constant(0.0)
theta_L = Constant(10*(np.pi/180)) #1deg in rad

def boundary(x, on_boundary):
    return on_boundary 

def left(x, on_boundary):
    return on_boundary and near(x[0],0,tol)

def right(x, on_boundary):
    return on_boundary and near(x[0],1,tol)

bc_left = DirichletBC(Theta, theta_0, left)
bc_right = DirichletBC(Theta, theta_L, right)

bcs = [bc_left,bc_right]

a1 = inner(GJ*grad(t),grad(v_t))*dx
L1 = f*v_t*dx

# Compute solution
theta = Function(Theta)
solve(a1 == L1, theta, bcs)

# Save solution in VTK format
file = File("displacement_solution.pvd")
file << theta