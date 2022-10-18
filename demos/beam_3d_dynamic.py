#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html
# from dolfinx import fem, mesh, plot
from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant,form, set_bc)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, TestFunction, TrialFunction,split,grad,dx)

from dolfinx.fem.petsc import (assemble_vector, assemble_matrix, apply_lifting, NonlinearProblem)
from petsc4py import PETSc
from dolfinx.nls.petsc import NewtonSolver

import gmsh
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot

plot_with_pyvista = True


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
gmsh.model.add_physical_group(0,[p1],name='st_pt')
gmsh.model.add_physical_group(0,[p2],name='end_pt')

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
st_pt = 0
end_pt = gmsh.model.mesh.getMaxNodeTag()
print(end_pt)
#TODO: check on number of nodes vs number of elements?
# gmsh.write("output/beam_mesh.msh")

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
    domain = xdmf.read_mesh(name="beam_mesh")

#confirm that we have an interval mesh in 3D:
print('cell_type   :', domain.ufl_cell())

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

# Time-stepping parameters
T       = 5.0
Nsteps  = 50
dt = T/Nsteps
p0 = 1.
cutoff_Tc = T/5

#define intial displacement function
f_0 = Constant(domain, (0.0,0.0,0.0))
def f(t):
    f_val = 0.0
    # Added some spatial variation here. Expression is sin(t)*x
    if t <= cutoff_Tc:
        f_val = p0*t/cutoff_Tc
    return Constant(domain, (0.0,0.0,f_val))

#################################################################
########### INITIALIZE  #############################
#################################################################

#MARKED: Utility fxn for ref-->3D element transformation computation
#TODO: account for the case that a1, a2 are not principal axes of inertia 
#TODO: add normal force+bending moment coupling
def tangent(domain):
    t = Jacobian(domain)
    return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))
# Compute transformation Jacobian between reference interval and elements
t = tangent(domain)

#compute section local axis
ez = as_vector([0, 0, 1])
a1 = cross(t, ez)
a1 /= sqrt(dot(a1, a1))
a2 = cross(t, a1)
a2 /= sqrt(dot(a2, a2))

#MARKED: declare construction beam element using BeamElement()
#construct mixed element function space; trial+test fxns
#split functions into disp (w) and rotation (theta) 
Ue = VectorElement("CG", domain.ufl_cell(), 1, dim=3)
W = FunctionSpace(domain, Ue*Ue)

#values at current time step
#Trial and Test Functions
u_ = TestFunction(W)
du = TrialFunction(W)
(w_, theta_) = split(u_)
(dw, dtheta) = split(du)
#Current (unknown) displacement
u = Function(W)

#values at previous timestep
u_old = Function(W)
udot_old = Function(W)
uddot_old = Function(W)

#compute midpoint values:
u_mid = 0.5*(u_old+u)
# theta_mid =
# w_mid =

#velocities (computed via implicit midpoint)
#QUESTION: when should u and du be split?
udot = Constant(domain, 2/dt)*u - Constant(domain, 2/dt)*u_old - udot_old
# wdot = 
# thetadot =

#accelerations (computed via implicit midpoint)
uddot = (udot - udot_old)/dt
# theteaddot =
# wddot =

#MARKED: three utility functions for deriv. of element trans, strains, stress
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
    return dot(diag(as_vector([ES, GS1, GS2, GJ, EI1, EI2])), generalized_strains(u))

#CONSTRUCT LHS FORM:
Sig = generalized_stresses(du)
Eps =  generalized_strains(u)
#TODO: check if this is still an issue in FEniCSx and change if quadrature is not needed
#Construct shear term reduced integration measure
dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
#construct LHS, with the two shear terms recieving reduced integration measure
k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear

#CONSTRUCT RHS FORM (neglecting BCs)
#add distributed loading
l_form = -rho*S*g*w[2]*dx
#TODO: general load application method (including pointsource, etc)

F = 
#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
#set degrees of freedom values as 0.
with ubc.vector.localForm() as uloc:
     uloc.set(0.)
#choose beam node at the origin to lock DOF
fixed_dof_num = 0
locate_BC = locate_dofs_topological(W,tdim,fixed_dof_num)
bcs = dirichletbc(ubc,locate_BC)

#TODO: figure out application of geometrical dof location for mixed element function spaces
# locate_BC1 = locate_dofs_geometrical(W.sub(0),lambda x: np.isclose(x[0], 0. ,atol=1e-6))
# locate_BC2 = locate_dofs_geometrical(W.sub(1),lambda x: np.isclose(x[0], 0. ,atol=1e-6))
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

#####PROJECT UTILITY######
def project(v, target_func, bcs=[]):

    """
    Solution from
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-
    function-dolfinx/4142/6
    """

    # Ensure we have a mesh and attach to measure
    V = target_func.function_space
    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = inner(Pv, w) * dx
    L = inner(v, w) * dx
    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

def solveNonlinear(F, w, bcs, abs_tol=1e-50, max_it=3, log=False):

    """
    Wrap up the nonlinear solver for the problem F(w)=0 and
    returns the solution
    """

    problem = NonlinearProblem(F, w, bcs)

    # Set the initial guess of the solution
    with w.vector.localForm() as w_local:
        w_local.set(0.1)
    solver = NewtonSolver(MPI.COMM_WORLD, problem)
    # if log is True:
    #     dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

    # Set the Newton solver options
    solver.atol = abs_tol
    solver.max_it = max_it
    solver.error_on_nonconvergence = False
    opts = PETSc.Options()
    opts["nls_solve_pc_factor_mat_solver_type"] = "mumps"
    solver.solve(w)

#################################################################
########### TIME STEPPING LOOP ##########################
#################################################################
time = np.linspace(0,T,Nsteps+1)
u_tip = np.zeros((Nsteps+1,))
f_t = np.zeros((Nsteps+1,))
for i in range(0,round(cutoff_Tc/dt)+1):
    f_t[i] = p0*i*dt/cutoff_Tc
xdmf_file = XDMFFile(MPI.COMM_WORLD, "solutions/displacement.xdmf", "w")
xdmf_file.write_mesh(domain)
# xdmf_file_stress = XDMFFile(MPI.COMM_WORLD, "solutions/stress.xdmf", "w")
# xdmf_file_stress.write_mesh(domain)
t = 0.

for i in range(0,Nsteps):
    #increment time
    t += dt
    print("------- Time step "+str(i+1)
            +" , t = "+str(t)+" -------")
    # #CONSTRUCT LHS FORM:
    # #compute time based forcing:
    # dWext = -inner(f(t),dw)*dx
    # #compute stresses and strains:
    # Sig = generalized_stresses(du)
    # Eps =  generalized_strains(u)
    # #construct LHS, with the two shear terms recieving reduced integration measure
    # k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear

    # #SOLVE VARIATIONAL PROBLEM
    # #initialize function in functionspace for beam properties
    # # u = Function(W)
    # # solve variational problem
    # problem = LinearProblem(k_form, l_form, u=u, bcs=[bcs])
    # u = problem.solve()

    # project(u,u_old)
    # u_old.interpolate(u)
    # # Save solution to XDMF format
    # xdmf_file.write_function(w.sub(0), t)

    # Solve the nonlinear problem for this time step and put the solution
    # (in homogeneous coordinates) in y_hom.
    solveNonlinear(F,w,bcs,log=False)

    #record tip displacement
    u_tip[i+1] = w.sub(0).eval([[10., 1.0, 0.]], [38])[2]

from matplotlib import pyplot as plt
# Plot tip displacement evolution
plt.figure()
plt.plot(time, u_tip)
plt.xlabel("Time")
plt.ylabel("Tip displacement")
plt.ylim(-0.8, 0.8)
plt.savefig("solutions/tip_displacement.png")


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

# print(W.sub(0))
# print(W.sub(0).collapse()[0])

#visualize with pyvista:
if plot_with_pyvista == False:
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
