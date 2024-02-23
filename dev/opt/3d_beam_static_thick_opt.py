#single, static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant,assemble_scalar,form)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import LinearProblem,NonlinearProblem
from dolfinx.mesh import locate_entities,locate_entities_boundary,meshtags
from dolfinx import nls
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,SpatialCoordinate,
                VectorElement, TestFunction, TrialFunction,split,grad,dx,Measure)
import gmsh
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot
# from petsc4py import PETSc
import argparse
from femo.fea.fea_dolfinx import FEA
from femo.csdl_opt.fea_model import FEAModel
from python_csdl_backend import Simulator as py_simulator

plot_with_pyvista = True
folder = 'Project/output/'

L = 1.0
h = 0.1
b = 0.1
vol = b*h*L
print(vol)

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
gmsh.initialize()
# model and mesh parameters
gdim = 3
tdim = 1
L = 1.0
# lc = 1e-1 #TODO: find if there is an easier way to set the number of elements
#construct line in 3D space
gmsh.model.add("Beam")
gmsh.model.setCurrent("Beam")
p1 = gmsh.model.occ.addPoint(0,0,0)
p2 = gmsh.model.occ.addPoint(L, 0, 0)
line1 = gmsh.model.occ.addLine(p1,p2)
# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()
# add physical marker
gmsh.model.add_physical_group(tdim,[line1])
#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.02*L)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.02*L)
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

#################################################################
########### READ MESH, EXTRACT MESH DETAILS #####################
#################################################################

#read in xdmf mesh from generation process
fileName = folder+"beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain = xdmf.read_mesh(name="beam_mesh")

NUMEL = domain.topology.index_map(1).size_global
NUMNODES = domain.topology.index_map(0).size_global

print(NUMNODES)
print(NUMEL)
nt=NUMEL
parser = argparse.ArgumentParser()
parser.add_argument('--nel',dest='nel',default=nt,
                    help='Number of elements')

# args = parser.parse_args()
# num_el = int(args.nel)

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
E = Constant(domain,70e3)
nu = Constant(domain,0.3)
G = E/2/(1+nu)
rho = Constant(domain,2.7e-3)
g = Constant(domain,9.81)
# L = Constant(domain,L)
V = Constant(domain,vol)

#define spatial coordinate
x = SpatialCoordinate(domain)

def get_constitutive_matrix(t,w):
    S = t*w
    #axial
    ES = E*S
    #bending
    EI1 = E*w*t**3/12
    EI2 = E*w**3*t/12
    #approximate torsional constant (see wikipedia)
    def torsional_constant(a,b):
        return (a*b**3)*((1/3)-0.21*(b/a)*(1-((b**4)/(12*a**4))))
    # #TODO: handle this case:
    J = torsional_constant(t,w)
    # # if t>=w:
    # #     J = torsional_constant(t,w)
    # # else:
    # #     J = torsional_constant(w,t)
    GJ = G*J
    # GJ = G*0.26*t*w**3
    #shear
    kappa = Constant(domain,5./6.)
    GS1 = kappa*G*S
    GS2 = kappa*G*S

    #construct constitutive matrix
    Q = diag(as_vector([ES, GS1, GS2, GJ, EI1, EI2]))
    return Q



# thick = Constant(domain,0.3)
# width = Constant(domain,0.1)
# Q = get_constitutive_matrix(thick,width)

#################################################################
########### CONSTRUCT VARIATIONAL FORMULATION ###################
#################################################################

def PDEres(statefxn,testfxn,inputfxn,f, dss,wid):
    thi=inputfxn
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

    u_ = statefxn
    du = testfxn
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
        Q = get_constitutive_matrix(thi,wid)
        return dot(Q, generalized_strains(u))

    Sig = generalized_stresses(du)
    Eps =  generalized_strains(u_)

    #modify quadrature scheme for shear components
    dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
    #LHS assembly
    k_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear

    # #weight per unit length
    q = -rho*thi*wid*g
    # #RHS
    # # l_form = -q*dx
    # l_form = -q*w_[2]*dx

    l_form = dot(f,du[2])*dss #+ q*du[2]*dx

    res = k_form - l_form

    return res

#################################################################
########### DEFINE OPTIMIZATION RELATED FUNCTIONS ###############
#################################################################

def volume(t,w):
    return t*w*dx

def compliance(u,f,dss):
    # (w, theta) = split(u)
    return dot(f,u[2])*dss

'''
    2.2. Create function spaces for the input and the state variables
'''

fea = FEA(domain)
# Add input to the PDE problem(thicknesses):
input_name = 'thickness'
input_fxn_space = FunctionSpace(domain, ('DG', 0))
input_fxn = Function(input_fxn_space)
# print(input_fxn_space.dofmap.index_map.size_global)

#Add state to the PDE problem (Displacements)
#construct mixed element function space
state_name = 'displacements'
Ue = VectorElement("CG", domain.ufl_cell(), 1, dim=3)
W = FunctionSpace(domain, Ue*Ue)

state_fxn = Function(W)
v = TestFunction(W)

#tip load
f = Constant(domain, -1.0)

#locate endpoint
DOLFIN_EPS = 3E-16
def Endpoint(x):
    return np.isclose(abs(x[0] - L), DOLFIN_EPS)
fdim = domain.topology.dim - 1
print(domain.topology.dim)
endpoint_node = locate_entities_boundary(domain,fdim,Endpoint)
print(np.full(len(endpoint_node),100,dtype=np.int32))
endpoint_id =100
facet_tag = meshtags(domain, fdim, endpoint_node,
                    np.full(len(endpoint_node),endpoint_id,dtype=np.int32))
# Define measures of the endpoint
metadata = {"quadrature_degree":4}
# ds_ = Measure('ds',domain=domain,subdomain_data=facet_tag,metadata=metadata)
ds_ = Measure('ds',domain=domain,subdomain_data=facet_tag,subdomain_id=endpoint_id,metadata=metadata)
#Define Residual
width=Constant(domain,0.1)
residual_form=PDEres(state_fxn,v,input_fxn,f,ds_(endpoint_id),width)

# Add outputs to the PDE problem:
output_name_1 = 'compliance'
output_form_1 = compliance(state_fxn,f,ds_(endpoint_id))
output_name_2 = 'volume'
output_form_2 = volume(input_fxn,width)

#add
fea.add_input(input_name, input_fxn)
fea.add_state(name=state_name,
                function=state_fxn,
                residual_form=residual_form,
                arguments=[input_name])
fea.add_output(name=output_name_1,
                type='scalar',
                form=output_form_1,
                arguments=[input_name, state_name])
fea.add_output(name=output_name_2,
                type='scalar',
                form=output_form_2,
                arguments=[input_name])

'''
    2.3. Define the boundary conditions
'''

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)
fixed_dof_num = 0
locate_BC = locate_dofs_topological(W,tdim-1,fixed_dof_num)
# bc = dirichletbc(ubc,locate_BC)
locate_BC_list = [locate_BC]
fea.add_strong_bc(ubc, locate_BC_list)
# Turn off the log info for the Newton solver
fea.REPORT = False


'''
3. Set up the CSDL model and run simulation
'''
fea_model = FEAModel(fea=[fea])

fea_model.create_input("{}".format('thickness'),
                            shape=nt,
                            val=h) # initializing with constant thickness

fea_model.add_design_variable('thickness', upper=10., lower=1e-2)
fea_model.add_objective('compliance')
fea_model.add_constraint('volume', equals=b*h*L)
sim = py_simulator(fea_model,analytics=False)
# Run the simulation
sim.run()

# Check the derivatives
# sim.check_totals(compact_print=True)
#
'''
4. Set up and run the optimization problem
'''
# Run the optimization with modOpt
from modopt.csdl_library import CSDLProblem

prob = CSDLProblem(
    problem_name='beam_thickness_opt',
    simulator=sim,
)

from modopt.scipy_library import SLSQP
optimizer = SLSQP(prob, maxiter=1000, ftol=1e-9)

# from modopt.snopt_library import SNOPT
# optimizer = SNOPT(prob,
#                   Major_iterations = 1000,
#                   Major_optimality = 1e-9,
#                   append2file=False)
# Solve your optimization problem
optimizer.solve()
print("="*40)
optimizer.print_results()

print(vol)
print(assemble_scalar(form(volume(input_fxn,width))))


#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################


uh = state_fxn
#save relevant fields for Paraview visualization
#save displacements
v= uh.sub(0).collapse()
v.name= "Displacement"
#save rotations
theta = uh.sub(1).collapse()
theta.name ="Rotation"

#plot solution
topology, cell_types, geometry = plot.create_vtk_mesh(domain,tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
grid.point_data["u"] = uh.sub(0).collapse().x.array.reshape((geometry.shape[0],3))
actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
warped = grid.warp_by_vector("u", factor=1)
actor_1 = plotter.add_mesh(warped, show_edges=True)
plotter.show_axes()
plotter.show()

# with XDMFFile(MPI.COMM_WORLD, folder+"output.xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)
#     xdmf.write_function(v)
#     xdmf.write_function(theta)