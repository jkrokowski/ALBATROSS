from mpi4py import MPI
import dolfinx.cpp.mesh
from dolfinx import mesh,plot
from dolfinx.fem import dirichletbc,locate_dofs_topological,Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace,locate_dofs_geometrical
from dolfinx.fem.petsc import  LinearProblem,create_vector,assemble_matrix,assemble_vector
from ufl import (diag,dx,cross,sqrt,Jacobian,sym,FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
from scipy.linalg import null_space
import matplotlib.pylab as plt
import time
import gmsh
from dolfinx.io import XDMFFile,gmshio

from dolfinx import geometry

# Create 2d mesh and define function space
N = 10
W = 1
H = 1
L = 5
Nx =1

# Q = as_matrix(np.linalg.inv(np.array([[ 1.00000000e-02, -9.55042934e-15,  9.88547364e-15, -2.43110605e-15, -2.47639244e-14, -3.58755674e-14],
#  [-9.79451567e-15,  2.86538070e-02, -8.53443335e-14,  2.26021536e-14,  2.20673380e-13,  3.05837338e-13],
#  [ 1.02544120e-14, -8.63386082e-14,  2.86538070e-02, -2.48375484e-14, -2.29409747e-13, -3.19401283e-13],
#  [-2.49591552e-15,  2.21173053e-14, -2.43662401e-14,  1.69251094e-01,  6.03246742e-14,  9.07611533e-14],
#  [-2.50345920e-14,  2.18061250e-13, -2.24140124e-13,  5.95064637e-14,  1.19926729e-01,  8.04328770e-13],
#  [-3.75481441e-14,  3.12573695e-13, -3.22686728e-13,  9.46644254e-14,  8.32584918e-13,  1.19926729e-01]])))
print(np.linalg.inv(np.array([[ 1.00000000e-02, -9.55042934e-15,  9.88547364e-15, -2.43110605e-15, -2.47639244e-14, -3.58755674e-14],
 [-9.79451567e-15,  2.86538070e-02, -8.53443335e-14,  2.26021536e-14,  2.20673380e-13,  3.05837338e-13],
 [ 1.02544120e-14, -8.63386082e-14,  2.86538070e-02, -2.48375484e-14, -2.29409747e-13, -3.19401283e-13],
 [-2.49591552e-15,  2.21173053e-14, -2.43662401e-14,  1.69251094e-01,  6.03246742e-14,  9.07611533e-14],
 [-2.50345920e-14,  2.18061250e-13, -2.24140124e-13,  5.95064637e-14,  1.19926729e-01,  8.04328770e-13],
 [-3.75481441e-14,  3.12573695e-13, -3.22686728e-13,  9.46644254e-14,  8.32584918e-13,  1.19926729e-01]])))
Q = diag(as_vector([100,34.8,34.8,5.9,8.3,8.3]))

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
p1 = gmsh.model.occ.addPoint(0.5,0.5,0.0)
p2 = gmsh.model.occ.addPoint(0.5, 0.5, L)
line1 = gmsh.model.occ.addLine(p1,p2)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# add physical marker
gmsh.model.add_physical_group(tdim,[line1])

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', .1)
gmsh.option.setNumber('Mesh.MeshSizeMax', 1)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write("output/beam_mesh.msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"output/beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

#read in xdmf mesh from generation process
fileName = "output/beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain2 = xdmf.read_mesh(name="beam_mesh")

# Compute transformation Jacobian between reference interval and elements
def tangent(domain):
    t = Jacobian(domain)
    return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))

t = tangent(domain2)

#compute section local axis
ez = as_vector([0, 0, 1])
a1 = cross(t, ez)
a1 /= sqrt(dot(a1, a1))
a2 = cross(t, a1)
a2 /= sqrt(dot(a2, a2))

#construct mixed element function space
Ue = VectorElement("CG", domain2.ufl_cell(), 1, dim=3)
W = FunctionSpace(domain2, Ue*Ue)

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
q = 1
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
# t2 = time.time()
# print("Total time:")
# print(t2-t0)
# print("Time for 2D problem:")
# print(t1-t0)
# print("Time for 3D problem:")
# print(t2-t1)

#==================================================#
#================ POST-PROCESS  ===================#
#========== (DISP. & STRESS RECOVERY)  ============#
#==================================================#

# #plot interval displacement
# if True:
#     #  tdim = domain.topology.dim
#     #  topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
#     #  grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#      plotter = pyvista.Plotter()
#     #  plotter.add_mesh(grid, show_edges=True,opacity=0.25)
#     #  plotter.view_isometric()
#      # plotter.view_vector((0.7,.7,.7))
#      # if not pyvista.OFF_SCREEN:
#      #      plotter.show()
#      tdim = domain2.topology.dim
#      topology, cell_types, geometry = plot.create_vtk_mesh(domain2, tdim)
#      grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
#      # plotter = pyvista.Plotter()
#      plotter.add_mesh(grid, show_edges=True,opacity=0.75)

#      # Attach vector values to grid and warp grid by vector
#      print(uh.x.array)
#      grid["u"] = uh.sub(0).collapse().x.array.reshape((geometry.shape[0], 3))
#      actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
#      warped = grid.warp_by_vector("u", factor=1000)
#      actor_1 = plotter.add_mesh(warped, show_edges=True)
#      if not pyvista.OFF_SCREEN:
#           plotter.show()

if True:
    # topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid)

    topology, cell_types, geometry = plot.create_vtk_mesh(domain2,tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()

    # v_topology, v_cell_types, v_geometry = plot.create_vtk_mesh(W.sub(0))
    # u_grid = pyvista.UnstructuredGrid(v_topology, v_cell_types, v_geometry)
    grid.point_data["u"] = uh.sub(0).collapse().x.array.reshape((geometry.shape[0],3))
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plot.screenshot("beam_mesh.png")