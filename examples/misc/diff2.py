from ufl import Cell,cross,grad,split,Measure,TensorElement,MixedElement,variable,diff,dot,as_vector,as_matrix,Constant,replace,Coefficient,interval,SpatialCoordinate
from dolfinx import fem,mesh
from mpi4py import MPI

from ALBATROSS.utils import get_vtx_to_dofs
# from petsc4py import PETSc

domain=mesh.create_unit_square(MPI.COMM_WORLD,3,3)
# cell = interval
cell = domain.ufl_cell()
c = variable(Constant(cell,shape=(6,)))

x = SpatialCoordinate(domain)
print(c.ufl_shape)
print(x[0].ufl_shape)
# u = dot(c,x)
u = c*x[0]
print(u.ufl_shape)

dudc = diff(u,c)
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

#get map from vertices to dofs for each displacment warping function
ubar_vtx_to_dof = get_vtx_to_dofs(domain,Uc.sub(0)) #this works for sub.sub as well


# ubar_c.vector.array = 

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







