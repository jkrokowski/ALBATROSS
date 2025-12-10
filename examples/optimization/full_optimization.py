import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

#################################################################
########### DEFINE THE INPUTS FOR THE BEAM PROBLEM ##############
#################################################################

# model and mesh parameters
gdim = 3
tdim = 1

#create or read in series of 2D meshes
N = 50 #number of quad elements per side on xc mesh
W = .099 #xs width
H = .099 #xs height
A = W*H #xs area
L = 20 

#define tip load magnitude 
F = .01 

#beam endpoint locations
p1 = (0,0,0)
p2 = (L,0,0)

#create cross-sectional mesh
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square
# xs_msh = ALBATROSS.mesh.create_rectangle(points,[N,N])

#cross-section mesh definition
radius = 0.05
num_el = 30 #number of elements through thickness

xs_msh = ALBATROSS.mesh.create_circle(radius,num_el,'disk')

xs_filename = 'beam_mass_minimization'
xs_msh.name = xs_filename
# with XDMFFile(MPI.COMM_WORLD, "output/"+xs_filename+".xdmf", "w") as xdmf:
#     xdmf.write_mesh(xs_msh)

#initialize material object
unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10e6,'nu':0.2},
                                           density=2700)

#initialize and run cross-sectional analysis
xs = ALBATROSS.cross_section.CrossSection(xs_msh,[unobtainium])
# xs.plot_mesh()
xs.get_xs_stiffness_matrix()
xs_list = [xs]

#create a beam axis
meshname = 'ex_1'
nodal_points = [p1,p2]
# number of segments of the beams that use different cross-sections
num_segments = len(nodal_points)-1 
num_ele = [10] #number of subdivisions for each beam segment
beam_axis = ALBATROSS.axial.BeamAxis(nodal_points,num_ele,meshname)

#define orientation of each xs with a vector
orientations = np.tile([0,1,0],num_segments)

#collect all xs information
xs_adjacency_list = [[0]] #this is the trivial connectivity for a uniform beam 
xs_info = [xs_list,orientations,xs_adjacency_list]

#################################################################
######### INITIALIZE BEAM OBJECT, APPLY BCs ############
#################################################################

#initialize beam object using beam axis and definition of xs's
CantileverBeam = ALBATROSS.beam.Beam(beam_axis,xs_info)

#show the orientation of each xs and the interpolated orientation along the beam
# CantileverBeam.plot_xs_orientations()

#applied fixed bc to first endpoint
CantileverBeam.add_clamped_point(p1)

#TODO: update this to be updated in the custom op?
#apply force at free end in the negative z direction
CantileverBeam.add_point_load([(0,0,-F)],[p2])

# #solve the linear problem
# CantileverBeam.solve()

CantileverBeam.get_mass()
print('original beam mass:',CantileverBeam.M)

#get mesh geometry
xy = xs_msh.geometry.x[xs.boundary_nodes,0:2]
xy_interior = xs_msh.geometry.x[xs.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

#create geometry variables for cross-section:
xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')
xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
xy.set_as_design_variable(lower=-.05,upper=.05,scaler=500)

#create csdl variables for tip load force:


#=====mesh motion=======#
inputs_mm = csdl.VariableGroup()
inputs_mm.xy = xy
inputs_mm.xy_interior = xy_interior
meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(xs_msh,
                                                       xs.boundary_nodes,
                                                       xs.interior_nodes,
                                                       filename=xs_filename)
outputs_mm = meshSmoothing.evaluate(inputs_mm)

#===== warping function computation =======#
inputs_w = csdl.VariableGroup()
inputs_w.xy = xy
inputs_w.xy_interior = outputs_mm.xy_interior

warping_model = ALBATROSS.csdl_utils.WarpingFunctionState(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

outputs_w = warping_model.evaluate(inputs_w)

#======= cross-section stiffness matrix ==========#
section_model = ALBATROSS.csdl_utils.BeamMatrixFromWarping(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

inputs_sec = csdl.VariableGroup()
inputs_sec.xy = xy
inputs_sec.xy_interior = outputs_mm.xy_interior
inputs_sec.w = outputs_w.w
inputs_sec.lmbda = outputs_w.lmbda

outputs_sec = section_model.evaluate(inputs_sec)

K = outputs_sec.K
K.name = 'stiffness_mat'
A = outputs_sec.A
A.name = 'area'
# csdl.derivative(outputs_sec.K,inputs_sec.xy)
# csdl.derivative(outputs_sec.A,inputs_sec.xy)

#======= beam deflection ==========#
inputs_beam = csdl.VariableGroup()
inputs_beam.K = outputs_sec.K
# inputs_beam.A = outputs_sec.A 
# inputs_beam.F = csdl.Variable(value = F)

beam_model = ALBATROSS.csdl_utils.BeamDeflection(CantileverBeam,
                                            tip_point = p2)

outputs_beam = beam_model.evaluate(inputs_beam)

#  = csdl.Variable(shape=(1,))
tip_displacement = outputs_beam.d.get(csdl.slice[beam_model.output_dofs[2]])
tip_displacement.name = 'tip_deflection'
# dddK = csdl.derivative(outputs_beam.d,inputs_beam.K)

# dddxy = csdl.derivative(tip_displacement,xy)

# dAdxy = csdl.derivative(outputs_sec.A,xy)

#======= beam mass ==========#
inputs_mass = csdl.VariableGroup()
inputs_mass.A = outputs_sec.A

mass_model = ALBATROSS.csdl_utils.BeamMass(CantileverBeam)

outputs_mass = mass_model.evaluate(inputs_mass)

beam_mass = outputs_mass.M

#======= mesh quality constraint ==========#
inputs_mq = csdl.VariableGroup()
inputs_mq.xy = xy
inputs_mq.xy_interior = outputs_mm.xy_interior

mesh_quality = ALBATROSS.csdl_utils.MeshQuality(xs_msh,
                                                boundary_nodes=xs.boundary_nodes,
                                                interior_nodes=xs.interior_nodes)

outputs_mq = mesh_quality.evaluate(inputs_mq)

mesh_metric = outputs_mq.Q

with csdl.namespace('Objective'):
    f = beam_mass
    f.add_name('beam_mass')
    f.set_as_objective()

with csdl.namespace('Deflection constraint'):
    g1 = tip_displacement
    g1.add_name('g1')
    g1.set_as_constraint(lower=-.35) # constraint

with csdl.namespace('Mesh Quality constraint'):
    g2 = mesh_metric
    g2.add_name('g2')
    g2.set_as_constraint(lower=0.05) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()

# recorder.visualize_adjacency_matrix()
# dddx = csdl.derivative(tip_displacement,xy)
#uncommment this to check the total derivatives of the pipeline
# dddx = sim.check_totals(tip_displacement,xy,step_size=0.001)

# print('current K:      ', sim[K])
# # print('dKdx(FD):  ', sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=.0001)[K,xy], '\n')
# # dKdx_FD = sim.compute_totals(K,xy,use_finite_difference=True,finite_difference_step_size=0.002)[K,xy]
# dKdx = sim.compute_totals(K,xy)[K,xy]
# print('Derivatives w.r.t. b-spline ctrl pts')
# dKdcoeffs = sim.compute_totals(K,inputs.coeffs)

from modopt import CSDLAlphaProblem
# from modopt import SLSQP
from modopt import PySLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='beam_mass_minimization',simulator=sim)

# optimizer = SLSQP(prob,recording=True,solver_options={'ftol':1e-8, 'maxiter':20})
optimizer = PySLSQP(prob,recording=True,solver_options={'maxiter':100,'acc':1e-6,'iprint':2})

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)