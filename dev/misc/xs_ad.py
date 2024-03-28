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
c = variable(Constant(cell,shape=(6,)))
n = variable(Constant(cell,shape=(6,)))
c7 = variable(Constant(cell))
# c7 = variable(fem.Constant(domain,PETSc.ScalarType((1.0))))
# c7 = fem.Constant(domain,PETSc.ScalarType((1.0)))
x = SpatialCoordinate(domain)
print(c.ufl_shape)
print(x[0].ufl_shape)
dx = Measure('dx',domain=domain)
V = fem.FunctionSpace(domain,('CG',1))
u = fem.Function(V)
v = TestFunction(V)
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
P1 = dot(c,n)*u*dx
# P1 = c7*u*dx

K11= diff(P1,c7)
Kx1= diff(P1,c)

# K11 = derivative(P1,u)

print(K11)

u = c*x[0]
print(u.ufl_shape)

dudc = diff(P1,c)
print(dudc)
print(dudc.ufl_shape)

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

# d2udc2 = dot(dudc.T,dudc)
# print(d2udc2.ufl_shape)

d2udc2 = diff(dudc,c)
print(d2udc2.ufl_shape)

#sum along last axis to get complementary potential energy function
K2 = sum([d2udc2[:,:,i] for i in range(c.ufl_shape[0])])
print(K2.ufl_shape)

if True:
    #cross-product notes
    x_vec = as_vector([x[0],x[1],0])
    u_vec = as_vector([ubar,ubar,ubar])
    # udisp = some_fxn
    #udisp.ufl_shape = (3,)
    moments = cross(x_vec,u_vec)


print()

# #take derivative of expression with respect to a variable:
# dudc1 = diff(u_exp,c1) #diff() takes partial derivatives
# print(dudc1)
# print(dudc1.ufl_shape)
# dudc1 = replace(dudc1,{ubar:fem.Function(V,name='ubar')})
# print(dudc)
# print(dudc.ufl_shape)
# dudc2 = diff(u_exp,c2)
# print(dudc2)
# dudc2 = replace(dudc2,{ubar:fem.Function(V,name='ubar')})
# print(dudc2)

# dudc1dc2 = diff(diff(u_exp,c1),c2)
# print(dudc1dc2)
# print(dudc1dc2.ufl_shape)

# #construct Jacobian matrix of expression:
# J = as_matrix([list(diff(u_exp,var)) for var in vars])
# # J = as_matrix([diff(u_exp,var) for var in vars])
# print(J.ufl_shape)







