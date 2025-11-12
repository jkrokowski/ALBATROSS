import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs


N = 2
W = 1.2
H = 1
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
domain.name = 'square_mesh_opt'
with XDMFFile(MPI.COMM_WORLD, "output/square_mesh_opt.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)

# # radius = 1
# # num_el = 40 #number of elements through wall thickness
# # domain = ALBATROSS.mesh.create_circle(radius,num_el,'disk')

#initialize CrossSection object
material = ALBATROSS.material.Material(name='unobtainium',
                            mat_type='ISOTROPIC',
                            mech_props={'E':100.0,'nu':0.2},
                            density=2700)

xs = ALBATROSS.cross_section.CrossSection(domain,[material])

xy=domain.geometry.x[xs.boundary_nodes,0:2]
xy_interior = domain.geometry.x[xs.interior_nodes,0:2]

#restrict custom explicit operation to 'x', 'w', or 'False'
check_partials = 'x'
#get the warping functions, since we are only interested in checking the beam matrix derivatives
xs._get_warping_functions()

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()

if check_partials != 'w':
    inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
    inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')


#set up
if check_partials != 'x':
    warping_input = csdl.Variable(value=np.vstack([xs.warping_functions[i].x.array for i in range(6)]).T)
    start = 0
    end = 12
    wf_num = 0
    warping_slice = csdl.Variable(value = warping_input.value[start:end,wf_num])

    inputs.w = warping_input.set(csdl.slice[start:end,wf_num],warping_slice)
    inputs.lmbda = csdl.Variable(value=np.vstack([xs.lmbdas[i].x.array for i in range(6)]).T)

section_model = ALBATROSS.csdl_utils.BeamMatrixFromWarping(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes,
                        check_partials=check_partials)
    
outputs_sec = section_model.evaluate(inputs)

# K = outputs.K
# K.name = 'stiffness_mat'
# A = outputs.A
# A.name = 'area'

# with csdl.namespace('Objective'):
#     f = -K[5,5]
#     f.add_name('max_bending_stiffness')
#     f.set_as_objective()

# with csdl.namespace('Area constraint'):
#     g1 = K[0,0]
#     g1.add_name('g1')
#     g1.set_as_constraint(upper=125,lower=75) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()

dKdx = csdl.derivative(outputs_sec.K,inputs.xy)
#might have to skip this line and run the full FD check first
dKdx_FD = np.load('dKdx_FD.npy')

#============ WARPING FUNCTION PARTIAL DERIVATIVE CHECK =========#
if check_partials == 'w':
    print('checking pK/pw...')
    dKdw_check = sim.check_totals(outputs_sec.K,warping_slice,step_size=1e-8,print_results=True)
    dKdw = dKdw_check[outputs_sec.K,warping_slice]['value']
    dKdw_FD = dKdw_check[outputs_sec.K,warping_slice]['fd_value']

    #check norms across K matrix entries (for all x)
    for i in range(36):
        print(i,np.linalg.norm(dKdw[i,:]-dKdw_FD[i,:]),np.linalg.norm(dKdw[i,:]),np.linalg.norm(dKdw_FD[i,:]))
    
    # step_size=0.0001
    # # wf_num = 'w3'
    # dK1dwn_FD=np.load('dK1dw'+str(wf_num)+'_FD_dw='+str(step_size)+'.npy')
    # dK2dwn_FD=np.load('dK2dw'+str(wf_num)+'_FD_dw='+str(step_size)+'.npy')
    # dK2invdwn_FD=np.load('dK2invdw'+str(wf_num)+'_FD_dw='+str(step_size)+'.npy')
    # dKdwn_FD=np.load('dKdw'+str(wf_num)+'_FD_dw='+str(step_size)+'.npy')
    
    # #check norms across x
    # for i in range(xs.boundary_nodes.shape[0]*2):
    #     print(i,np.linalg.norm(dKdw[:,i]-dKdw_FD[:,i]))

#============ SPATIAL PARTIAL DERIVATIVE CHECK ========#
if check_partials == 'x':
    print('checking pK/px...')
    dKdx_check = sim.check_totals(outputs_sec.K,inputs.xy,step_size=1e-6,print_results=True)
    dKdx = dKdx_check[outputs_sec.K,inputs.xy]['value']
    dKdx_FD = dKdx_check[outputs_sec.K,inputs.xy]['fd_value']

    #check norms across K matrix entries (for all x)
    for i in range(36):
        print(i,np.linalg.norm(dKdx[i,:]-dKdx_FD[i,:]),np.linalg.norm(dKdx[i,:]),np.linalg.norm(dKdx_FD[i,:]))

    #check norms across x
    for i in range(xs.boundary_nodes.shape[0]*2):
        print(i,np.linalg.norm(dKdx[:,i]-dKdx_FD[:,i]),np.linalg.norm(dKdx[:,i]),np.linalg.norm(dKdx_FD[:,i]))

    dKdx_FD = np.save('dKdx_FD',dKdx_FD)

    # step_size=0.0001
    # dK1dx0_FD=np.load('dK1dx_FD_dx='+str(step_size)+'.npy')
    # dK2dx0_FD=np.load('dK2dx_FD_dx='+str(step_size)+'.npy')
    # dK2invdx0_FD=np.load('dK2invdx_FD_dx='+str(step_size)+'.npy')
    # dKdx0_FD=np.load('dKdx_FD_dx='+str(step_size)+'.npy')

dK00dx = sim.compute_totals(K00,reduced_xy)
dK00dx_FD = sim.compute_totals(K00,reduced_xy,use_finite_difference=True,finite_difference_step_size=0.001)
dK00dw = sim.compute_totals(K00,reduced_w)
dK00dw_FD = sim.compute_totals(K00,reduced_w,use_finite_difference=True,finite_difference_step_size=0.001)

# sim.check_totals(outputs_sec.K[0,0],inputs.w[25:30,0])
# sim.check_totals(outputs_wf.w[:10,0],inputs.coeffs,step_size=0.0001,print_results=True)
# sim.check_totals(outputs_mm.xy_interior,inputs.coeffs,step_size=0.0001,print_results=True)


#TODO: need to figure out why check totals doesn't seem to affect FD, but does affect normal totals??
# sim.check_totals(outputs_wf.w[:10,0],inputs.xy,step_size=0.0001,print_results=True) 
dwdx = sim.compute_totals(outputs_wf.w[:10,0],inputs.coeffs)
dwdx_FD = sim.compute_totals(outputs_wf.w[:10,0],inputs.xy,use_finite_difference=True,finite_difference_step_size=0.002)

sim.check_totals(outputs_wf.lmbda,inputs.xy,print_results=True)
# sim.check_totals(outputs_wf.w,inputs.xy,print_results=True)

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
prob = CSDLAlphaProblem(problem_name='bending_stiffness_max',simulator=sim)

# optimizer = SLSQP(prob,recording=True,solver_options={'ftol':1e-8, 'maxiter':20})
optimizer = PySLSQP(prob,recording=True,solver_options={'maxiter':20,'acc':1e-6,'iprint':2})

# Check first derivatives at the initial guess, if needed
# optimizer.check_first_derivatives(prob.x0,step=0.001)

# Solve your optimization problem
optimizer.solve()

optimizer.print_results()

print("xy values:")
print(xy.value)