#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html
from dolfinx import *
from dolfinx.io import XDMFFile,gmshio
from ufl import Jacobian, diag
import meshio
import gmsh
from mpi4py import MPI

gmsh.initialize()

gdim = 3
lc = 1e-2
p1 = gmsh.model.occ.add_point(0,0,0,lc)
p2 = gmsh.model.occ.add_point(1,1,1,lc)
l1 = gmsh.model.occ.add_line(p1,p2)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# add physical markert
gmsh.model.add_physical_group(3,[l1])

#generate the mesh
gmsh.model.mesh.generate(gdim)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'

with XDMFFile(msh.comm, f"out_gmsh/beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# #create beam mesh (topologically 1D in 3D space)
# with pygmsh.geo.Geometry() as geom:
#     lc = 1e-2
#     p1 = geom.add_point((0,0,0),lc)
#     p2 = geom.add_point((1,1,1),lc)
#     l1 = geom.add_line(p1,p2)
#     mesh = geom.generate_mesh()
#     pygmsh.write("beam.msh")

# mesh_from_file = meshio.read("beam.msh")

# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     points = mesh.points[:,:2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
#     return out_mesh

# line_mesh = create_mesh(mesh_from_file,"line")
# meshio.write("beam.xdmf",line_mesh)
# mesh = Mesh()
# meshfile = XDMFFile('beam.xdmf')
# meshfile.read(mesh)

# # Compute transformation Jacobian between reference interval and elements
# def tangent(mesh):
#     t = Jacobian(mesh)
#     return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))

# t = tangent(mesh)

# #compute section local axis
# ez = as_vector([0, 0, 1])
# a1 = cross(t, ez)
# a1 /= sqrt(dot(a1, a1))
# a2 = cross(t, a1)
# a2 /= sqrt(dot(a2, a2))

# # material parameters and material model
# thick = Constant(0.3)
# width = thick/3
# E = Constant(70e3)
# nu = Constant(0.3)
# G = E/2/(1+nu)
# rho = Constant(2.7e-3)
# g = Constant(9.81)

# S = thick*width
# ES = E*S
# EI1 = E*width*thick**3/12
# EI2 = E*width**3*thick/12
# GJ = G*0.26*thick*width**3
# kappa = Constant(5./6.)
# GS1 = kappa*G*S
# GS2 = kappa*G*S

# Ue = VectorElement("CG", mesh.ufl_cell(), 1, dim=3)
# W = FunctionSpace(mesh, Ue*Ue)

# u_ = TestFunction(W)
# du = TrialFunction(W)
# (w_, theta_) = split(u_)
# (dw, dtheta) = split(du)

# def tgrad(u):
#     return dot(grad(u), t)
# def generalized_strains(u):
#     (w, theta) = split(u)
#     return as_vector([dot(tgrad(w), t),
#                       dot(tgrad(w), a1)-dot(theta, a2),
#                       dot(tgrad(w), a2)+dot(theta, a1),
#                       dot(tgrad(theta), t),
#                       dot(tgrad(theta), a1),
#                       dot(tgrad(theta), a2)])
# def generalized_stresses(u):
#     return dot(diag(as_vector([ES, GS1, GS2, GJ, EI1, EI2])), generalized_strains(u))

# Sig = generalized_stresses(du)
# Eps =  generalized_strains(u_)

# dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
# k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear
# l_form = Constant(-rho*S*g)*w_[2]*dx

# # Boundary conditions (clamped at p1)
# def bottom(x, on_boundary):
#     return near(x[1], 0.)
# bc = DirichletBC(W, Constant((0, 0, 0, 0, 0, 0)), bottom)

# # solve variational problem
# u = Function(W)
# solve(k_form == l_form, u, bc)


# #save relevant fields for Paraview visualization
# #save displacements
# v = u.sub(0, True)
# v.rename("Displacement", "")
# File('beam-disp.pvd') << v
# #save rotations
# theta = u.sub(1, True)
# theta.rename("Rotation", "")
# File('beam-rotate.pvd') << theta
# #save moments
# V1 = VectorFunctionSpace(mesh, "CG", 1, dim=2)
# M = Function(V1, name="Bending moments (M1,M2)")
# Sig = generalized_stresses(u)
# M.assign(project(as_vector([Sig[4], Sig[5]]), V1))
# File('beam-moments.pvd') << M
