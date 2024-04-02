from ufl import TestFunction,derivative,indices,as_tensor,Cell,cross,grad,split,Measure,TensorElement,MixedElement,variable,diff,dot,as_vector,as_matrix,Constant,replace,Coefficient,interval,SpatialCoordinate
from dolfinx import fem,mesh
from mpi4py import MPI

import numpy as np

from ALBATROSS.utils import get_vtx_to_dofs
from ALBATROSS.material import getMatConstitutive
from petsc4py import PETSc
# from petsc4py import PETSc

domain=mesh.create_unit_square(MPI.COMM_WORLD,3,3)
# cell = interval
cell = domain.ufl_cell()
# c = variable(Constant(cell,shape=(6,)))
# n = variable(Constant(cell,shape=(6,)))
# c7 = variable(Constant(cell))
# c = variable(fem.Constant(domain,PETSc.ScalarType((1.0,1.0,1.0,1.0,1.0,1.0))))
# n = variable(fem.Constant(domain,PETSc.ScalarType((1.0,1.0,1.0,1.0,1.0,1.0))))
# n = variable(fem.Constant(domain,PETSc.ScalarType(((1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
#                                                    (7.0, 8.0, 9.0, 10.0,11.0,12.0),
#                                                    (13.0,14.0,15.0,16.0,17.0,18.0)))))
n = variable(fem.Constant(domain,PETSc.ScalarType(((1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))))
c7 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c8 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c9 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c10 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c11 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c12 = variable(fem.Constant(domain,PETSc.ScalarType((0.0))))
c = as_tensor([c7,c8,c9,c10,c11,c12])
c_var = variable(c)
# c7 = variable(fem.Constant(domain,PETSc.ScalarType((1.0))))
x = SpatialCoordinate(domain)
dx = Measure('dx',domain=domain)
V = fem.FunctionSpace(domain,('CG',1))
V2 =fem.VectorFunctionSpace(domain,('CG',1),dim=3)
u = fem.Function(V)
v = TestFunction(V)

u2 = fem.Function(V2)
v2 = TestFunction(V2)

# u = dot(c,x)

# mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
#                         'MECH_PROPS':{'E':100.,'nu':.2} ,
#                         'DENSITY':2.7e3}
#         }
# i,j,k,l = indices(4)
# C_mat = getMatConstitutive(domain,mats[mats.keys()[0]])
# eps = as_tensor([[uhat_mode[0],uhat_mode[1],uhat_mode[2]],
#                             [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
#                             [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])sigma = as_tensor(C_mat[i,j,k,l]*eps[k,l],(i,j))
# sigma11 = sigma[0,0]
# P1 = fem.assemble_scalar(fem.form(c[0]*x[0]*x[1]*dx))
# P1 = dot(dot(n,c),u2[0])*dx
u2[0].ufl_shape
P1 = dot(n,c)*u2[0]*dx
P2 = dot(n,c)*u2[1]*dx
P3 = dot(n,c)*u2[2]*dx
P = P1 + P2 + P3
# P1 = c7*u*dx

K11 = diff(P,c7)
K12 = diff(P,c8)
K13 = diff(P,c9)
# Kx1= diff(P1,c)
K11_scalar = fem.assemble_scalar(fem.form(K11))
K12_scalar = fem.assemble_scalar(fem.form(K12))
K13_scalar = fem.assemble_scalar(fem.form(K13))
# Kx1_assembled = fem.petsc.assemble_vector(fem.form(Kx1))
# K11 = derivative(P1,u)

print(K11)

dKdc = diff(P,c_var)
print(dKdc.ufl_shape)
K_vec = fem.assemble_vector(fem.form(dKdc))
# print(dKdc.ufl_shape)

V = fem.FunctionSpace(domain,('CG',1))
# Uc = fem.TensorFunctionSpace(domain,('CG',1),shape=(3,6))
Uc_e = TensorElement("CG",domain.ufl_cell(),1,shape=(3,6))
Uc = fem.FunctionSpace(domain,MixedElement(4*[Uc_e]))
ubar = fem.Function(V)
u_c = fem.Function(Uc)
ubar_c,uhat_c,utilde_c,ubreve_c = split(u_c)
dx = Measure('dx',domain)
u_exp = ubar_c*c
print(u_exp.ufl_shape)

#construct fake test nullspace
N = np.random.random((12*domain.geometry.x.shape[0],6))

#get map from vertices to dofs for each displacment warping function
ubar_c_vtx_to_dof = get_vtx_to_dofs(domain,Uc.sub(0)) #this works for sub.sub as well

u_c.vector.array[ubar_c_vtx_to_dof.flatten()] = N.flatten()[ubar_c_vtx_to_dof.flatten()].flatten()