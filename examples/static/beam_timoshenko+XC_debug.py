#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, TestFunction, TrialFunction,split,grad,dx)
from ufl import dx
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot,fem,mesh

# from FROOT_BAT.beam_model import BeamModelRefined
from FROOT_BAT import geometry,cross_section,beam_model

#################################################################
########### CONSTRUCT MESH FOR LOCATING BEAM XCs ################
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 3
W = .1
H = .1
mesh2d_0 = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
mesh2d_1 = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[0.75*W, 0.5*H]]),[N,N], cell_type=mesh.CellType.quadrilateral)

meshes2D = [mesh2d_0,mesh2d_1]

#define material parameters
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100,'nu':0.2} }
                        }

#define spanwise locations of XCs with a 1D mesh
p1 = (0,0,0)
p2 = (5,0,0)
ne_2D = len(meshes2D)-1
ne_1D = 10

meshname1D_2D = 'square_tapered_beam_1D_2D'
meshname1D_1D = 'square_tapered_beam_1D_1D'

#mesh for locating beam cross-sections along beam axis
mesh1D_2D = geometry.beamIntervalMesh3D([p1,p2],[ne_2D],meshname1D_2D)

#mesh used for 1D analysis
mesh1D_1D = geometry.beamIntervalMesh3D([p1,p2],[ne_1D],meshname1D_1D)

#get fxn for XC properties at any point along beam axis
mats2D = [mats for i in range(len(meshes2D))]
xcdata=[meshes2D,mats2D]
xcinfo = cross_section.defineXCsFor1D([mesh1D_2D,xcdata],mesh1D_1D)


#best test code

# Compute transformation Jacobian between reference interval and elements
def tangent(domain):
    t = Jacobian(domain)
    return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))

t = tangent(mesh1D_1D)

#compute section local axis
ez = as_vector([0, 0, 1])
a1 = cross(t, ez)
a1 /= sqrt(dot(a1, a1))
a2 = cross(t, a1)
a2 /= sqrt(dot(a2, a2))

#construct mixed element function space
Ue = VectorElement("CG", mesh1D_1D.ufl_cell(), 1, dim=3)
W = FunctionSpace(mesh1D_1D, Ue*Ue)

u_ = TestFunction(W)
du = TrialFunction(W)
(w_, theta_) = split(u_)
(dw, dtheta) = split(du)

def tgrad(u):
    return dot(grad(u), t)
def generalized_strains(u):
    (w, theta) = split(u)
    return as_vector([dot(tgrad(w), t),
                      dot(tgrad(w), a1)-dot(theta, a2),
                      dot(tgrad(w), a2)+dot(theta, a1),
                      dot(tgrad(theta), t),
                      dot(tgrad(theta), a1),
                      dot(tgrad(theta), a2)])
def generalized_stresses(u):
    # Q = diag(as_vector([1,2,3,4,5,6]))
    # return dot(Q, generalized_strains(u))
    return dot(xcinfo, generalized_strains(u))

Sig = generalized_stresses(du)
Eps =  generalized_strains(u_)

#modify quadrature scheme for shear components
dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
#LHS assembly
# k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear
k_form = (inner(Sig,Eps))*dx

#weight per unit length
rho = Constant(mesh1D_1D,2.7e-3)
g = Constant(mesh1D_1D,9.81)
S = Constant(mesh1D_1D,0.01)
q = rho*S*g
#RHS
l_form = -q*w_[2]*dx

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

fixed_dof_num = 0
locate_BC = locate_dofs_topological(W,1,fixed_dof_num)

bcs = dirichletbc(ubc,locate_BC)

#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
u = Function(W)
# solve variational problem
problem = LinearProblem(k_form, l_form, u=u, bcs=[bcs])
u = problem.solve()
