# static 1D Cantilever Beam with a rectangular cross section
# this example uses Euler-Bernoulli ("classical") beam theory
# import basix.ufl_wrapper
import numpy as np
from mpi4py import MPI
from basix.ufl import element, mixed_element
import basix
from dolfinx.fem import (Function, dirichletbc,functionspace,
                         locate_dofs_topological,Constant,Expression)
from dolfinx.io import VTKFile,XDMFFile,gmshio
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from ufl import (split,Jacobian,cross,sqrt,as_vector,dot,diag,SpatialCoordinate,inner, TestFunction, TrialFunction, div, grad, dx)
import matplotlib.pyplot as plt
import gmsh

########## GEOMETRIC INPUT ####################
E = 70e3
L = 1.0
b = 0.1
h = 0.05
rho= 2.7e-3
g = 9.81
#NOTE: floats must be converted to dolfin constants on domain below

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
gmsh.initialize()

# model and mesh parameters
gdim = 3
tdim = 1
lc = 1e-1 #TODO: find if there is an easier way to set the number of elements

#construct line in 3D space
gmsh.model.add("Beam")
gmsh.model.setCurrent("Beam")
p1 = gmsh.model.occ.addPoint(0,0,0,lc)
p2 = gmsh.model.occ.addPoint(1,2,3,lc)
line = gmsh.model.occ.addLine(p1,p2)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# add physical marker
gmsh.model.add_physical_group(tdim,[line])

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write("beam_mesh.msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

#read in xdmf mesh from generation process
fileName = "beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain = xdmf.read_mesh(name="beam_mesh")

#confirm that we have an interval mesh in 3D:
print('cell_type   :', domain.ufl_cell())

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
x = SpatialCoordinate(domain)

thick = Constant(domain,h)
width = Constant(domain,b)
E = Constant(domain,E)
rho = Constant(domain,rho)
g = Constant(domain,g)
nu = Constant(domain,0.3)
G = E/2/(1+nu)

A = thick*width
EA = E*A
GJ = G*0.26*thick*width**3
EI1 = E*width*thick**3/12
EI2 = E*width**3*thick/12
# Create Hermite order 3 on a interval (for more informations see:
#    https://defelement.com/elements/examples/interval-Hermite-3.html )
beam_element=element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)
axial_element = element("CG", domain.topology.cell_name(), 1, )
torsion_element = element("CG", domain.topology.cell_name(), 1, )

W = functionspace(domain,mixed_element([axial_element,
                                        torsion_element,
                                        beam_element,
                                        beam_element]))
print("Number of DOFs: %d" % W.dofmap.index_map.size_global)
# print("Number of elements (intervals): %d" % NUM_ELEM)
# print("Number of nodes: %d" % (NUM_ELEM+1))
u = TestFunction(W)
v = TrialFunction(W)
(u_x,theta,u_yy,u_zz)=split(u)

def tangent(domain):
    t = Jacobian(domain)
    return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))

t = tangent(domain)

#compute section local axis
ez = as_vector([0, 0, 1])
a1 = cross(t, ez)
a1 /= sqrt(dot(a1, a1))
a2 = cross(t, a1)
a2 /= sqrt(dot(a2, a2))

def tgrad(u):
    return dot(grad(u), t)

# def generalized_strains(u):
#     (u_x,theta,u_yy,u_zz)=split(u)
#     return as_vector([dot(tgrad(u_x), t),
#                       dot(tgrad(theta), t),
#                       div(dot(tgrad(u_yy), a1)),
#                       div(dot(tgrad(u_zz), a2))])

print(u_x.ufl_shape)
print(theta.ufl_shape)
print(u_yy.ufl_shape)
print(u_zz.ufl_shape)

def generalized_strains(u):
    (u_x,theta,u_yy,u_zz)=split(u)
    return as_vector([grad(u_x),
                      grad(theta),
                      div(grad(u_yy)),
                      div(grad(u_zz))])

def generalized_stresses(u):
    return dot(diag(as_vector([EA, GJ, EI1, EI2])), generalized_strains(u))

Eps =  generalized_strains(u)
Sig = generalized_stresses(v)

k_form = inner(Sig,Eps)*dx


#distributed load value (due to weight)
q=-rho*A*g

l_form = q*u*dx
#################################################################
########### EXACT SOLUTIONS FOR CANTILEVER BEAM #################
#################################################################
#cantilever
w_cl = q/EI1 * ( (x[0]**4)/24 -(L*x[0]**3)/6 +((L**2)*x[0]**2)/4 )

#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

#locate endpoints
startpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], 0))
# endpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], L))

#locate DOFs from endpoints

# startdof_axial = locate_dofs_topological(W.sub(2),0,startpt)
# startdof1=locate_dofs_topological(W.sub(2),0,startpt)
# # enddof1=locate_dofs_topological(W.sub(2),0,endpt)
# startdof2=locate_dofs_topological(W.sub(3),0,startpt)
# enddof2=locate_dofs_topological(W.sub(3),0,endpt)
clamped_dofs = locate_dofs_topological(W,0,startpt)

#fix displacement of start point and rotation as well
# axial_bc = dirichletbc(ubc)
# torsion_bc = dirichletbc(ubc)
# fixed_disp1 = dirichletbc(ubc,np.array([startdof1[0]]))
# fixed_rot1 = dirichletbc(ubc,np.array([startdof1[1]]))
# fixed_disp2 = dirichletbc(ubc,np.array([startdof2[0]]))
# fixed_rot2 = dirichletbc(ubc,np.array([startdof2[1]]))
clamped_bc = dirichletbc(ubc,clamped_dofs)

# bcs = [axial_bc,
#        torsion_bc,
#        fixed_disp1,
#        fixed_rot1,
#        fixed_disp2,
#        fixed_rot2]

#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
uh = Function(W)

# solve variational problem
problem = LinearProblem(k_form, l_form, u=u, bcs=clamped_bc)
uh=problem.solve()
uh.name = "Displacement and Rotation "

#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################

#save output (if uh is directly visualized in Paraview, the plot will look odd,
#  as the rotation DOFs are included in this  )
with VTKFile(domain.comm, "output/output.pvd", "w") as vtk:
    vtk.write([uh._cpp_object])
    
# #NOTE: The solution uh contains both the rotation and the displacement solutions
# #The rotation and displacment solutions can be separated as follows:
# disp = np.empty(0)
# rot = np.empty(0)
# for i,x in enumerate(uh.x.array):
#     if i % 2 != 0:
#         rot = np.append(rot,x)
#     else:
#         disp = np.append(disp,x)

# #evaluate derivatives and interpolate to higher order function space
# T = functionspace(domain,("CG",1))

# #interpolate exact ufl expression onto high-order function space
# disp_expr = Expression(w_cl,T.element.interpolation_points())
# disp_exact = Function(T)
# disp_exact.interpolate(disp_expr)

# #extract numpy arrays for plotting
# exact_disp = disp_exact.x.array

# x_exact = np.linspace(0,1,exact_disp.shape[0])

# x_fem = np.linspace(0,1,disp.shape[0])

# figure, axis = plt.subplots(1,1)

# print("Maximum magnitude displacement (cantilever exact solution) is: %e" % np.min(exact_disp))
# print("Maximum magnitude displacement (cantilever FEM solution) is: %e" % np.min(disp))

# ####PLOTTING####
# #Displacement
# axis.plot(x_exact,exact_disp,label='exact')
# axis.plot(x_fem, disp,marker='x',linestyle=':',label='FEM')
# axis.set_xlabel('x (location along beam)')
# axis.set_ylabel('u(x) (transverse displacement)')
# axis.set_title("Displacement of FEM vs EB theory")
# axis.legend()

# plt.show()