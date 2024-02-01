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
import ALBATROSS

plot_with_pyvista = True
folder = 'output/'

#################################################################
########### CONSTRUCT & READ INPUT MESHES (1D & 2D) #############
#################################################################

#general geometry parameters
L = 1
W = .1
H = .1

#defining cross-section geometry
#box xs wall thickness:
t1 = 0.01
t2 = 0.01
t3 = 0.01
t4 = 0.01
xs_points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [t1,t2,t3,t4]
num_el = 4*[4]

#populate a list of the mesh objects
meshes = []
num_xs = 3
for i in range(num_xs):
    meshes.append(ALBATROSS.utils.create_2D_box(xs_points,thicknesses,num_el,'box_xs'))

#define xs material properties
mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#define position of each xs in space
node_x = np.array([0, L/2, L])
node_y = np.zeros_like(node_x)
node_z = np.zeros_like(node_x)
axial_pts = np.concatenate([node_x.reshape((num_xs,1)),node_y.reshape((num_xs,1)),node_z.reshape((num_xs,1))],axis=1)

#define orienation of primary cross-section axis
orientations = np.tile([0,1,0],len(node_x))

# CONSTRUCT MESH FOR POSITIONING XS'S IN SPACE
# 1: define beam nodes
axial_pos_pts = axial_pts
# 2: define number of 1D elements btwn each xs for axial position mesh
axial_pos_ne = list(np.ones((num_xs-1))) 
# 3: specify axial position mesh name
meshname_axial_pos = 'demo_axial_postion_mesh'
# construct mesh
axial_pos_mesh = ALBATROSS.utils.beam_interval_mesh_3D(axial_pos_pts,
                                                       axial_pos_ne,
                                                       meshname_axial_pos)

# CONSTRUCT MESH FOR 1D ANALYSIS
# 1: define beam nodes (previously done)
#       --> these are the same as the those used for the axial position mesh
# 2: define number of 1D elements btwn each xs for axial position mesh
beam_el_num = 100
axial_ne = list(beam_el_num*np.ones((num_xs-1)))
# 3: specify name for 1D analysis mesh
meshname_axial = 'demo_axial_mesh'
# construct mesh
axial_mesh = ALBATROSS.utils.beam_interval_mesh_3D(axial_pts,axial_ne,meshname_axial)

#analyze cross section
boxXS = ALBATROSS.cross_section.CrossSection(domain,mats)

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

def PDEres(statefxn,testfxn,inputfxn,f, dss,wid):
    
    
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