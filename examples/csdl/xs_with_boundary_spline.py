import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs

import vtk

'''
This optimization problem is not well-posed with just the bending stiffness maximization
A potential way to counter this (without applying constraints on the boundary self-intersections)
would be to add a shear stiffness constraint as well as the area constraint?
the shear stiffness constraint prevents the "web" from necking down and self intersecting

UPDATE: the shear stiffness constraint didn't work because element inversion is not handled well by the cross-section model
maybe this needs to be "fixed" by the mesh smoothing?
'''

N = 10
W = .5
H = .6
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
filename = 'bending_stiffness_maximization'
domain.name = filename
with XDMFFile(MPI.COMM_WORLD, "output/"+filename+".xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

#initialize CrossSection object
material = ALBATROSS.material.Material(name='unobtainium',
                            mat_type='ISOTROPIC',
                            mech_props={'E':100.0,'nu':0.2},
                            density=2700)

xs = ALBATROSS.cross_section.CrossSection(domain,[material])
xs.get_xs_stiffness_matrix()

#get imp
xy = domain.geometry.x[xs.boundary_nodes,0:2]
xy_interior = domain.geometry.x[xs.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

# #===== CONVERT BOUNDARY TO POLAR COORDS =======#
# x0 = 0
# y0 = 0
# X = domain.geometry.x[:,0] - x0
# Y = domain.geometry.x[:,1] - y0

# theta = np.arctan2(Y,X)
# theta = np.mod(theta, 2*np.pi)   # Force into [0, 2π) 
# r = np.sqrt(X**2 + Y**2)

# ordering = np.argsort(theta)
# theta_sorted = theta[ordering]
# r_sorted = r[ordering]



#=====FIT BOUNDARY B-SPLINE ========#
#TODO: need to check on the errors in compute_basis_matrix() in the b_spline_space
num_parametric = 30
bspline_degree=3
boundary_spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
parametric_coords = np.array([(i,) for i in np.linspace(0,1,xs.boundary_nodes.shape[0]+1)])
boundary_points = csdl.concatenate([xy[list(xs.boundary_ordering)],xy[0:1,:]]) #duplicate the start/endpoint
boundary_spline_coeffs = boundary_spline_space.fit(values = boundary_points,parametric_coordinates= parametric_coords)
coeffs = boundary_spline_coeffs.value

# num_parametric = 30
# bspline_degree=3
# num_knots = num_parametric+(bspline_degree-1)*2
# knots = np.linspace(0,1,num_parametric+(bspline_degree-1)*2)
# boundary_spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,),knots=knots)
# parametric_coords = np.array([(i,) for i in np.linspace(0,1,xs.boundary_nodes.shape[0]+bspline_degree)])
# boundary_points = csdl.concatenate([xy[list(xs.boundary_ordering)],xy[list(xs.boundary_ordering)][:3,:]]) #duplicate the start/endpoints
# boundary_spline_coeffs = boundary_spline_space.fit(values = boundary_points,parametric_coordinates= parametric_coords)
# coeffs = boundary_spline_coeffs.value

# =======evaluate b-spline for boundary points ======#
inputs = csdl.VariableGroup()
inputs.coeffs = csdl.Variable(value=coeffs,name='boundary spline coeffs')
inputs.coeffs.set_as_design_variable(lower=-.75,upper=.75,scaler=100)
boundary_spline = lfs.Function(boundary_spline_space,inputs.coeffs,name='boundary_spline')
xy_boundary = boundary_spline.evaluate(parametric_coords)[list(xs.inverse_boundary_ordering)]
xy_boundary.name = 'xy'
# xy_boundary.set_as_design_variable(lower=-1,upper=1,scaler=1000)

#create interior node variable
xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')
# xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
# xy.set_as_design_variable(lower=-1,upper=1,scaler=100)

#===== output b-spline coefficient locations=======#
points = inputs.coeffs.value
num_coeffs = inputs.coeffs.value.shape[0]
vtk_points = vtk.vtkPoints()
for p in points:
    vtk_points.InsertNextPoint(p[0], p[1], 0.0)

# Create a polyline
polyline = vtk.vtkPolyLine()
polyline.GetPointIds().SetNumberOfIds(num_coeffs)
for i in range(num_coeffs):
    polyline.GetPointIds().SetId(i, i)

# Wrap in a cell array
cells = vtk.vtkCellArray()
cells.InsertNextCell(polyline)

# Make a PolyData object
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)
polydata.SetLines(cells)

# Write to file
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("boundary.vtp")
writer.SetInputData(polydata)
writer.Write()


#=====mesh motion=======#
inputs_mm = csdl.VariableGroup()
inputs_mm.xy = xy_boundary
inputs_mm.xy_interior = xy_interior
meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(domain,
                                                       xs.boundary_nodes,
                                                       xs.interior_nodes,
                                                       filename=filename)
outputs_mm = meshSmoothing.evaluate(inputs_mm)

#===== warping function computation =======#
inputs_w = csdl.VariableGroup()
inputs_w.xy = inputs_mm.xy
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
inputs_sec.xy = xy_boundary
inputs_sec.xy_interior = outputs_mm.xy_interior
inputs_sec.w = outputs_w.w
inputs_sec.lmbda = outputs_w.lmbda

outputs_sec = section_model.evaluate(inputs_sec)

K = outputs_sec.K
# K.name = 'stiffness_mat'
A = outputs_sec.A
# A.name = 'area'

#======= mesh quality constraint ==========#
inputs_mq = csdl.VariableGroup()
inputs_mq.xy = xy_boundary
inputs_mq.xy_interior = outputs_mm.xy_interior

mesh_quality = ALBATROSS.csdl_utils.MeshQuality(xs.msh,
                                                boundary_nodes=xs.boundary_nodes,
                                                interior_nodes=xs.interior_nodes)

outputs_mq = mesh_quality.evaluate(inputs_mq)

mesh_metric = outputs_mq.Q


with csdl.namespace('Objective'):
    f = -K[5,5]
    f.add_name('max_bending_stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = A 
    g1.add_name('g1')
    g1.set_as_constraint(upper=.31,lower=.29) # constraint

with csdl.namespace('Mesh Quality constraint'):
    g2 = mesh_metric
    g2.add_name('g2')
    g2.set_as_constraint(lower=0.05) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()

# recorder.visualize_adjacency_matrix()

from modopt import CSDLAlphaProblem
# from modopt import SLSQP
from modopt import PySLSQP

# Instantiate your problem using the csdl Simulator object and name your problem
prob = CSDLAlphaProblem(problem_name='bending_stiffness_max',simulator=sim)

# optimizer = SLSQP(prob,recording=True,solver_options={'ftol':1e-8, 'maxiter':20})
optimizer = PySLSQP(prob,recording=True,solver_options={'maxiter':100,'acc':1e-6,'iprint':2})

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)