import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import locate_entities_boundary,locate_entities
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs


N = 2
W = .5
H = .6
points = [[-W/2,-H/2],[W/2, H/2]]

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])
filename = 'warping_op_test'
domain.name = filename
# with XDMFFile(MPI.COMM_WORLD, "output/"+filename+".xdmf", "w") as xdmf:
#     xdmf.write_mesh(domain)

#initialize CrossSection object
material = ALBATROSS.material.Material(name='unobtainium',
                            mat_type='ISOTROPIC',
                            mech_props={'E':100.0,'nu':0.2},
                            density=2700)

xs = ALBATROSS.cross_section.CrossSection(domain,[material])
xs.get_xs_stiffness_matrix()

#get imp
xy=domain.geometry.x[xs.boundary_nodes,0:2]
xy_interior = domain.geometry.x[xs.interior_nodes,0:2]

recorder = csdl.Recorder(inline=True)
recorder.start()

#create interior node variable
xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')
xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
xy.set_as_design_variable(lower=-1,upper=1,scaler=100)

#===== warping function computation =======#
inputs_w = csdl.VariableGroup()
inputs_w.xy = xy
inputs_w.xy_interior = xy_interior

warping_model = ALBATROSS.csdl_utils.WarpingFunctionState(xs=xs,
                        boundary_nodes=xs.boundary_nodes,
                        interior_nodes=xs.interior_nodes)

outputs_w = warping_model.evaluate(inputs_w)


#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)
sim.run()

warping_slice = outputs_w.w.get(csdl.slice[1,0])
dwdx_check = sim.check_totals(warping_slice,xy,step_size=0.001)
print()

# warping_input = csdl.Variable(value=np.vstack([xs.warping_functions[i].x.array for i in range(6)]).T)
# start = 0
# end = 108
# wf_num = 3
# warping_slice = csdl.Variable(value = warping_input.value[start:end,wf_num])

# inputs.w = warping_input.set(csdl.slice[start:end,wf_num],warping_slice)
# inputs.lmbda = csdl.Variable(value=np.vstack([xs.lmbdas[i].x.array for i in range(6)]).T)

# recorder.visualize_adjacency_matrix()

#uncommment this to check the total derivatives of the pipeline
# sim.check_totals(outputs_sec.K,inputs.coeffs)