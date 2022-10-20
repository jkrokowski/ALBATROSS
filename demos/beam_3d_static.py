#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

# from FRuIT_BAT 

from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, TestFunction, TrialFunction,split,grad,dx)
import meshio
import gmsh
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot

from FRuIT_BAT.beam_model import BeamModelRefined
from FRuIT_BAT.geometry import beamIntervalMesh3D

plot_with_pyvista = False


#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1
lc = 1e-1 #TODO: find if there is an easier way to set the number of elements

fileName = "output/beam_mesh.xdmf"
meshname = 'beam_mesh'

#allow for passing list of points
p1 = (0,0,0)
p2 = (1,2,3)

#TODO: make sure that one can just call this function and get back a beam mesh
beamIntervalMesh3D([p1,p2],lc,fileName,meshname)

# gmsh.initialize()

# #construct line in 3D space
# gmsh.model.add("Beam")
# gmsh.model.setCurrent("Beam")
# p1 = gmsh.model.occ.addPoint(0,0,0,lc)
# p2 = gmsh.model.occ.addPoint(1,2,3,lc)
# line = gmsh.model.occ.addLine(p1,p2)

# # Synchronize OpenCascade representation with gmsh model
# gmsh.model.occ.synchronize()

# # add physical marker
# gmsh.model.add_physical_group(tdim,[line])

# #generate the mesh and optionally write the gmsh mesh file
# gmsh.model.mesh.generate(gdim)
# # gmsh.write("output/beam_mesh.msh")

# #use meshio to convert msh file to xdmf
# msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
# msh.name = 'beam_mesh'
# # cell_markers.name = f"{msh.name}_cells"
# # facet_markers.name = f"{msh.name}_facets"

# #write xdmf mesh file
# with XDMFFile(msh.comm, f"output/beam_mesh.xdmf", "w") as file:
#     file.write_mesh(msh)

# # close gmsh API
# gmsh.finalize()

#read in xdmf mesh from generation process
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

#initialize cross sectional properties matrices
beam_properties = [ES,GS1,GS2,GJ,EI1,EI2]

#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

#initialize beam element using mesh and beam properties
beam_model = BeamModelRefined(domain,beam_properties)
#construct LHS of weak form
a_form = beam_model.elasticEnergy()

#add body force:
L_form = -rho*S*g*beam_model.u_[2]*dx

###################################

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(beam_model.beam_element.W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

fixed_dof_num = 0
locate_BC = locate_dofs_topological(beam_model.beam_element.W,tdim,fixed_dof_num)

bcs = dirichletbc(ubc,locate_BC)
#TODO: figure out application of geometrical dof location for mixed element function spaces
# locate_BC1 = locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),lambda x: np.isclose(x[0], 0. ,atol=1e-6))
# locate_BC2 = locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),lambda x: np.isclose(x[0], 0. ,atol=1e-6))
# bcs = [dirichletbc(ubc, locate_BC1, W.sub(0)),
#         dirichletbc(ubc, locate_BC2, W.sub(1)),
#        ]
#TODO: figure out intricacies of topological mesh marking
#the dof's might will need to be split to allow for independent application of 
# fixed displacements or fixed rotations

# def start_boundary(x):
#     return np.isclose(x[0],0)
# domain.topology.create_connectivity(0,tdim)
# beam_st_pt = locate_entities_boundary(domain,0,)
# fixed_endpoint=locate_entities(domain,tdim,lambda x: np.isclose(x[0], 0. ,atol=1e-6))
# print(fixed_endpoint)


#SOLVE VARIATIONAL PROBLEM
#initialize function for disp/rot vector in beam model function space
u = Function(beam_model.beam_element.W)
# solve variational problem
problem = LinearProblem(a_form, L_form, u=u, bcs=[bcs])
u = problem.solve()

#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################

#save relevant fields for Paraview visualization
#save displacements
v = u.sub(0)
v.name= "Displacement"
# File('beam-disp.pvd') << v
#save rotations
theta = u.sub(1)
theta.name ="Rotation"
# File('beam-rotate.pvd') << theta
#save moments
# V1 = VectorFunctionSpace(domain, "CG", 1, dim=2)
# M = Function(V1, name="Bending moments (M1,M2)")
# Sig = generalized_stresses(u)
#TODO: fix the projection function like Ru did for the shell tool
"""
    Solution from
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-
    function-dolfinx/4142/6
    """
# M.assign(project(as_vector([Sig[4], Sig[5]]), V1))
# File('beam-moments.pvd') << M


# with VTKFile(MPI.COMM_WORLD, "output.pvd", "w") as vtk:
#     vtk.write([v._cpp_object])
#     # vtk.write([theta._cpp_object])
#     # vtk.write([M._cpp_object])

with XDMFFile(MPI.COMM_WORLD, "output/output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(v)
    xdmf.write_function(theta)

# TODO: fix pyvista visualization using plotter3d()
#  and compute_nodal_disp() from shell module 

#visualize with pyvista:
if plot_with_pyvista == True:
    # topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    # plotter = pyvista.Plotter()
    # plotter.add_mesh(grid)

    topology, cell_types, geometry = plot.create_vtk_mesh(domain,tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()

    # v_topology, v_cell_types, v_geometry = plot.create_vtk_mesh(W.sub(0))
    # u_grid = pyvista.UnstructuredGrid(v_topology, v_cell_types, v_geometry)
    grid.point_data["u"] = u.x.array.reshape((geometry.shape[0],3))
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    warped = grid.warp_by_vector("u", factor=1.5)
    actor_1 = plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        figure = plot.screenshot("beam_mesh.png")
