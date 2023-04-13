#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

#this example has a beam with a kink in it 
# (inspired by the sort of centerline of a helicopter rotor axis)

from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, TestFunction, TrialFunction,split,grad,dx)
import gmsh
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot

plot_with_pyvista = True
folder = 'output/'

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
gmsh.initialize()

# model and mesh parameters
gdim = 3
tdim = 1
R = 10.0
lc = 1e-1 #TODO: find if there is an easier way to set the number of elements

#construct line in 3D space
gmsh.model.add("Beam")
gmsh.model.setCurrent("Beam")
p1 = gmsh.model.occ.addPoint(0,0,0)
p2 = gmsh.model.occ.addPoint(R, 0, 0)
# p3 = gmsh.model.occ.addPoint(R, -0.025*R, -0.05*R)
line1 = gmsh.model.occ.addLine(p1,p2)
# line2 = gmsh.model.occ.addLine(p2,p3)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# add physical marker
gmsh.model.add_physical_group(tdim,[line1])

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01*R)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.1*R)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write(folder+"beam_mesh.msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, folder+f"beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

#read in xdmf mesh from generation process
fileName = folder+"beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain = xdmf.read_mesh(name="beam_mesh")

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
thick = Constant(domain,0.3)
width = thick/3
E = Constant(domain,70e3)
nu = Constant(domain,0.3)
G = E/2/(1+nu)
rho = Constant(domain,2.7e-3)
g = Constant(domain,9.81)

S = thick*width
ES = E*S
EI1 = E*width*thick**3/12
EI2 = E*width**3*thick/12
GJ = G*0.26*thick*width**3
kappa = Constant(domain,5./6.)
GS1 = kappa*G*S
GS2 = kappa*G*S

#construct constitutive matrix
Q = diag(as_vector([ES, GS1, GS2, GJ, EI1, EI2]))

#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

# Compute transformation Jacobian between reference interval and elements
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

#construct mixed element function space
Ue = VectorElement("CG", domain.ufl_cell(), 1, dim=3)
W = FunctionSpace(domain, Ue*Ue)

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
    return dot(Q, generalized_strains(u))

Sig = generalized_stresses(du)
Eps =  generalized_strains(u_)

#modify quadrature scheme for shear components
dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
#LHS assembly
k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear

#weight per unit length
q = rho*S*g
#RHS
l_form = -q*w_[2]*dx

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

fixed_dof_num = 0
locate_BC = locate_dofs_topological(W,tdim,fixed_dof_num)

bcs = dirichletbc(ubc,locate_BC)


#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
uh = Function(W)
# solve variational problem
problem = LinearProblem(k_form, l_form, u=uh, bcs=[bcs])
uh = problem.solve()

#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################

#save relevant fields for Paraview visualization
#save displacements
v= uh.sub(0).collapse()
v.name= "Displacement"
#save rotations
theta = uh.sub(1).collapse()
theta.name ="Rotation"

with XDMFFile(MPI.COMM_WORLD, folder+"output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(v)
    xdmf.write_function(theta)

#plot solution
topology, cell_types, geometry = plot.create_vtk_mesh(domain,tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
print(v.x.array.shape)
print(theta.x.array.shape)
# v_topology, v_cell_types, v_geometry = plot.create_vtk_mesh(W.sub(0))
# u_grid = pyvista.UnstructuredGrid(v_topology, v_cell_types, v_geometry)
grid.point_data["u"] = uh.sub(0).collapse().x.array.reshape((geometry.shape[0],3))
actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=10)
actor_1 = plotter.add_mesh(warped, show_edges=True)
plotter.show_axes()
plotter.show()