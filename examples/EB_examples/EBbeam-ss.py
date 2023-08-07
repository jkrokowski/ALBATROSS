#static 1D Cantilever Beam inspired by Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html
# this example uses euler-bernoulli ("classical") beam theory

import dolfinx
from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant)
from dolfinx.io import XDMFFile,VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary,create_interval
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, FiniteElement, TestFunction, TrialFunction,split,div,grad,dx)
from mpi4py import MPI
import numpy as np
# import pyvista
from dolfinx import plot
import basix

''' WARNING: this code is currently outputting the displacment solution for a pinned-fixed condition, not pinned-pinned'''

########## GEOMETRIC INPUT ####################
E = 70e3
L = 1.0
b = 0.1
h = 0.05
#NOTE: floats must be converted to dolfin constants on domain below

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
domain = create_interval(MPI.COMM_WORLD, 5000, [0, L])

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
thick = Constant(domain,h)
width = Constant(domain,b)
E = Constant(domain,E)
rho = Constant(domain,2.7e-3)
g = Constant(domain,9.81)
# nu = Constant(domain,0.3)
# G = E/2/(1+nu)

A = thick*width
EI = (E*width*thick**3)/12

#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

#define Moment expression
# def M(u):
#     return EI*grad(grad(u))
def M(u):
    return EI*div(grad(u))


# Create Hermite order 3 on a interval
beam_element = basix.ufl_wrapper.create_element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)
# beam_element = FiniteElement("Hermite", "interval", 3)
# beam_element = basix.ufl_wrapper.BasixElement(beam_element_basix)

#finite element function space on domain, with trial and test fxns
# W = FunctionSpace(domain,("HER", 3))
W = FunctionSpace(domain,beam_element)
u_ = TestFunction(W)
v = TrialFunction(W)

#distributed load value (due to weight)
q=rho*A*g

#bilinear form (LHS)
# k_form = div(grad(du))*M(u_)*dx
k_form = inner(div(grad(v)),M(u_))*dx

#linear form (RHS)
l_form = -q*u_*dx

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

# fixed_dof_num = [0,9]
# locate_BC = locate_dofs_topological(W,1,fixed_dof_num)

# def startpt(x):
#     return np.isclose(x[0], 0.000001)
# def endpt(x):
#     return np.isclose(x[0], 0.99)

# startbc = locate_entities(domain,0,startpt)
# endbc = locate_entities(domain,0,endpt)
startbc_dofs = locate_dofs_geometrical(W,lambda x : np.isclose(x[0], 0, 1e-8))
endbc_dofs = locate_dofs_geometrical(W,lambda x : np.isclose(x[0], L, 1e-8))

start_bcs = dirichletbc(ubc,startbc_dofs)
end_bcs = dirichletbc(ubc,endbc_dofs)

#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
u = Function(W)

# solve variational problem
problem = LinearProblem(k_form, l_form, u=u, bcs=[start_bcs,end_bcs])
# problem = LinearProblem(k_form, l_form, u=u, bcs=[])
uh=problem.solve()
uh.name = "Displacement"

print(np.min(uh.x.array))

#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################
with VTKFile(domain.comm, "output/output.pvd", "w") as vtk:
    vtk.write([uh._cpp_object])
# disp = VTKFile('output/disp.pvd')
# disp << uh

# with XDMFFile(MPI.COMM_WORLD, "output/output.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(uh)