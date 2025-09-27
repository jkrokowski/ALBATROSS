import csdl_alpha as csdl
import ALBATROSS
import numpy as np
from dolfinx.mesh import CellType,locate_entities_boundary,locate_entities,exterior_facet_indices,create_unit_square
from dolfinx.io import XDMFFile
from mpi4py import MPI
import lsdo_function_spaces as lfs

recorder = csdl.Recorder(inline=True)
recorder.start()

inputs = csdl.VariableGroup()
N = 4
offset = 1

h_to_f = 10
w_to_w = 10

m1,n1 = N*h_to_f+offset,N
m2,n2 = N,N*w_to_w+offset

H = 1
W = 1
tf = 1/h_to_f
tw = 1/w_to_w

mesh_A = create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=CellType.quadrilateral)
boundary_labels_A = {}
boundary_labels_A['left'] = -W/2
boundary_labels_A['right'] = W/2
boundary_labels_A['top'] = H/2
boundary_labels_A['bottom'] = H/2-tf
mesh_A.geometry.x[:, :2] -= .5          #center at 0
mesh_A.geometry.x[:, 0] *= W            #scale x
mesh_A.geometry.x[:, 1] *= tf           #scale y
mesh_A.geometry.x[:, 1] += H/2 - tf/2   #translate vertically
mesh_A.name = 'mesh_A'

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_A.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_A)

mesh_B = create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=CellType.quadrilateral)
boundary_labels_B = {}
boundary_labels_B['left'] =-W/2
boundary_labels_B['right'] = W/2
boundary_labels_B['top'] = H/2
boundary_labels_B['bottom'] = H/2-tf
mesh_B.geometry.x[:, :2] -= .5      #center at 0
mesh_B.geometry.x[:, 0] *= tw       #scale x
mesh_B.geometry.x[:, 1] *= W        #scale y
mesh_B.name = 'mesh_B'

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_B.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_B)

def get_labeled_nodes(msh,boundary_labels):
    node_labels = {}
    node_labels['all']=locate_entities(msh,0,lambda x: np.ones_like(x[0]))
    node_labels['boundary'] = locate_entities_boundary(msh,0,lambda x: np.ones_like(x[0]))
    node_labels['left'] = locate_entities_boundary(msh,0,lambda x: np.isclose(boundary_labels['left'],x[0]))
    node_labels['right'] = locate_entities_boundary(msh,0,lambda x: np.isclose(boundary_labels['right'],x[0]))
    node_labels['top'] = locate_entities_boundary(msh,0,lambda x: np.isclose(boundary_labels['top'],x[1]))
    node_labels['bottom'] = locate_entities_boundary(msh,0,lambda x: np.isclose(boundary_labels['bottom'],x[1]))
    node_labels['interior'] = node_labels['all'][~np.isin( node_labels['all'], node_labels['boundary'])]

    return node_labels

node_labels_A = get_labeled_nodes(mesh_A,boundary_labels_A)
node_labels_B = get_labeled_nodes(mesh_B,boundary_labels_B)

def fit_boundary_b_splines(msh,node_labels):
    '''
    return a set of b-splines that form a closed loop for the rectangle
    '''
    #use the same b-spline space for all edges
    num_parametric = 10
    bspline_degree=3
    spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
    
    boundary_splines = {}
    edges = ['left','right','top','bottom']
    for edge in edges:
        nodes = node_labels[edge]

        parametric_coords = np.array([(i,) for i in np.linspace(0,1,nodes.shape[0])])
        #TODO: need to check if this always returns points ordered in the same winding directions
        ordering = ALBATROSS.utils.order_boundary_nodes(msh.geometry.x[nodes,0:2])
        ordered_vertices = nodes[ordering]
        # inverse_ordering = np.argsort(ordering)
        points = msh.geometry.x[ordered_vertices,0:2]
        edge_spline_coeffs = spline_space.fit(values = points,parametric_coordinates= parametric_coords)
        edge_spline = lfs.Function(spline_space,edge_spline_coeffs,name=edge+'_spline')
        boundary_spline[edge] = edge_spline

    return boundary_splines

boundary_splines_A = fit_boundary_b_splines(mesh_A,node_labels_A)

class SignedDistanceFunction():
    def __init__(self,msh,boundary_splines):
        '''
        pass in the series of b-spline edges
        '''
        self.msh = msh
        self.boundary_splines = boundary_splines


    def evaluate(self,eval_pts):
        d_list = []
        W = csdl.Variable(shape=eval_pts.shape)
        for spline in self.boundary_splines:
            #compute squared distance for each point to each spline
            proj_eval_pts = spline.evaluate(spline.project(eval_pts))
            distance_eval = proj_eval_pts-eval_pts
            sq_dist = csdl.norm(distance_eval,axes=(1,))

            #compute winding number integral for each evaluation point for this b-spline
            W += self._winding_number_for_spline_segment(spline,eval_pts)


        D = csdl.minimum(csdl.vstack(d_list),rho=10000,axes=(0,))

        #compute winding number for each evaluation point
        W = csdl.Variable(shape=eval_pts)
        for spline in self.boundary_splines:
            W += self._winding_number_for_spline_segment(eval_pts)

        #get sign from winding number
        sign = csdl.tanh(100*(csdl.absolute(W)-0.5))

        return sign*D
    
    def _winding_number_for_spline_segment(self,spline,eval_pts):
        '''
        use the trapezoidal rule to approximate the winding number integral for each spline
        '''
        num_parametric = 11
        t = np.linspace(0,1,num_parametric)
        x_t = spline.evaluate(t).reshape(num_parametric,2)
        tangents = spline.evaluate(t,parametric_derivative_orders =(1)).reshape(num_parametric,2)
        h = 1/num_parametric
        w = csdl.Variable(shape=eval_pts.shape[0])
        for i in csdl.frange(eval_pts.shape[0]):
            for j in csdl.frange(x_t.shape[0]):
                self._winding_kernel(spline,x_t[j],eval_pts[i],tangents[j])

            # w[i] = 


        return w

    def _winding_kernel(spline,x_t,pt,tangent):
        '''
        x_t: physical point along spline at parametric location t
        p: physical point to compute distance
        tangent: tangent vector along spline at parametric location t
        '''
        d = x_t-pt
        cross = d[:,0]*tangent[1] - d[:,1]*tangent[0]
        denom = csdl.norm(d)
        return cross/denom
        

phi_A = SignedDistanceFunction(mesh_A,boundary_splines_A)

signedDistance = phi_A.evaluate(np.array([[0.25,0.25],[0.75,-.75],[.1,.52],[-.1,-.6],[0,0]]))


#TODO: need to make this so the SDF is a function of xy
#TODO: need to clean up how we handle multiple splines
#CONSTRUCT BOUNDARY B-SPLINES (with a closed, uniform knot vector)
num_parametric = 10
bspline_degree=3
spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
xy=domain.geometry.x[boundary_nodes,0:2]
I_list = []
d_list = []
xy_test = csdl.Variable(value = np.array([[0.25,0.25],[0.75,-.75],[.1,.52],[-.1,-.6],[0,0]]))

for nodes in [boundary_nodes_left,boundary_nodes_right,boundary_nodes_top,boundary_nodes_bottom]:
    parametric_coords = np.array([(i,) for i in np.linspace(0,1,nodes.shape[0])])
    ordering = ALBATROSS.utils.order_boundary_nodes(domain.geometry.x[nodes,0:2])
    ordered_vertices = nodes[ordering]
    inverse_ordering = np.argsort(ordering)
    # boundary_points = csdl.concatenate([xy[list(ordering)],xy[0:1,:]]) #duplicate the start/endpoint
    points = domain.geometry.x[ordered_vertices,0:2] #duplicate the start/endpoint
    # print(points)
    edge_spline_coeffs = spline_space.fit(values = points,parametric_coordinates= parametric_coords)
    # coeffs = boundary_spline_coeffs.value
    edge_spline = lfs.Function(spline_space,edge_spline_coeffs,name='edge_spline')
    
    #evaluation points:
    #points should be: 
    #   ----
    #   -+--
    #   --+-
    #   ---+

    eval_pts = xy_test
    proj_eval_pts = edge_spline.evaluate(edge_spline.project(eval_pts))
    distance_eval = proj_eval_pts-eval_pts
    sq_distance_eval = csdl.norm(distance_eval,axes=(1,))
    d_list.append(sq_distance_eval)
    tangent = edge_spline.evaluate(np.linspace(0,1,8).reshape((8,1)),parametric_derivative_orders =(1)).value
    
    #winding number computation:
    def winding_kernel(t,pts):
        '''
        t: array of parametric points
        p: physical points to compute distance to
        '''
        x_t = edge_spline.evaluate(t.reshape((t.shape[0],1)))
        d = csdl.expand(x_t,pts.shape,action='i->ji')-pts
        tangent = edge_spline.evaluate(t.reshape((t.shape[0],1)),parametric_derivative_orders =(1))
        tangents = csdl.expand(tangent,pts.shape,action='i->ji')
        cross = d[:,0]*tangents[:,1] - d[:,1]*tangents[:,0]
        denom = csdl.norm(d,axes=(1,))
        k = cross/denom
        return k

    w = winding_kernel(np.array([0.0]),eval_pts)

    #TODO: convert to csdl
    def trapezoidal_rule(f,m=10,a=0,b=1):
        t_list = np.linspace(a,b,m)
        h = (b-a) / (m-1)
        vals = np.array([f(np.array([t]),eval_pts) for t in t_list])
        return h * (0.5*vals[0] + vals[1:-1].sum() + 0.5*vals[-1])
    
    #approximate the winding number integral
    I = trapezoidal_rule(winding_kernel)
    print(f"winding value: {I.value}")
    I_list.append(I)
    
    # sign = (tangents[:,0]*distance[:,1]-tangents[:,1]*distance[:,0] ) / csdl.norm(tangents,distance)[:,0]
    # # signed_distance = csdl.expand(sign,distance.shape,action='i->ij')*csdl.norm(distance,axes=(1,))
    # signed_distance = sign*csdl.norm(distance,axes=(1,))
    
    print(f"unsigned distance for points: {sq_distance_eval.value}")
    # print(f"signed distance for points: {csdl.minimum(distance).value}")

    # print(csdl.minimum(distance).value)    

#==========================
W = I_list[0] +I_list[1] +I_list[2]+I_list[3]
D = csdl.minimum(csdl.vstack(d_list),rho=10000,axes=(0,)).value

#smooth sign function
sign = csdl.tanh(100*(csdl.absolute(W)-0.5))

SDF = sign*D

print(f'points: {eval_pts}')
print(f'SDF at points: {SDF.value}')

#dSDFdx = csdl.derivative(SDF,wrts=x)

#order the boundary using a nearest neighbor search:
ordering = ALBATROSS.utils.order_boundary_nodes(domain.geometry.x[boundary_nodes,0:2])
ordered_vertices = boundary_nodes[ordering]
inverse_ordering = np.argsort(ordering)

xy=domain.geometry.x[boundary_nodes,0:2]
xy_interior = domain.geometry.x[interior_nodes,0:2]

inputs.xy = csdl.Variable(value=xy,shape=xy.shape,name='xy')
inputs.xy_interior = csdl.Variable(value=xy_interior,shape=xy_interior.shape,name='xy_interior')

xy = inputs.xy
xy_interior = inputs.xy_interior

#CONSTRUCT A BOUNDARY B-SPLINE (with a closed, uniform knot vector)
num_parametric = 16
bspline_degree=3
boundary_spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
parametric_coords = np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0]+1)])
boundary_points = csdl.concatenate([xy[list(ordering)],xy[0:1,:]]) #duplicate the start/endpoint
boundary_spline_coeffs = boundary_spline_space.fit(values = boundary_points,parametric_coordinates= parametric_coords)
coeffs = boundary_spline_coeffs.value

# #TODO: USE A **PERIODIC** B-SPLINE to prevent the corner from being 
# #make a uniform knot vector of length (num_parametric+6)
# num_ctrl_pts = num_parametric+bspline_degree*2+2
# knot_indices = np.arange(0,num_ctrl_pts)
# num_repeated_ctrl_pts = 3
# # knots = (knot_indices)/(num_ctrl_pts-1)
# knots = (knot_indices-bspline_degree)/(num_parametric-1)
# periodic_bspline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,),knots=knots,knot_indices=knot_indices)
# parametric_coords2 = np.array([(i,) for i in np.linspace(0,1,boundary_nodes.shape[0])])
# periodic_boundary_spline_coeffs = periodic_bspline_space.fit(values = xy[list(ordering)],parametric_coordinates= parametric_coords2)

# boundary_spline_coeffs.name = 'boundary spline coeffs'
inputs.coeffs = csdl.Variable(value=coeffs*2)
inputs.coeffs.name = 'boundary spline coeffs'
inputs.coeffs.set_as_design_variable(lower=-2,upper=2,scaler=2)
boundary_spline = lfs.Function(boundary_spline_space,inputs.coeffs,name='boundary_spline')
# evaluated_points = boundary_spline.evaluate(parametric_coords,plot=True)

#TODO: increase knot multiplicity or use a composite spline for the boundary
inputs.xy = boundary_spline.evaluate(parametric_coords)[list(inverse_ordering)]

meshSmoothing = ALBATROSS.csdl_utils.EllipticSmoothing(domain,boundary_nodes,interior_nodes)

outputs_mm = meshSmoothing.evaluate(inputs)

inputs.xy_interior = outputs_mm.xy_interior

crosssection = ALBATROSS.csdl_utils.CrossSection(domain=domain,
                            xs_analysis_type='TS',
                            material_type='ISOTROPIC',
                            material_name='unobtainium',
                            mech_props={'E':100.0,'nu':0.2},
                            boundary_nodes=boundary_nodes,
                            interior_nodes=interior_nodes)

#only evaluate once
outputs = crosssection.evaluate(inputs)
K = outputs.K
K.name = 'stiffness_mat'
A = outputs.A
A.name = 'area'

with csdl.namespace('Objective'):
    f = -K[5,5]
    f.add_name('max_bending_stiffness')
    f.set_as_objective()

with csdl.namespace('Area constraint'):
    g1 = K[0,0]
    g1.add_name('g1')
    g1.set_as_constraint(upper=125,lower=75) # constraint

#APPARENTLY the simulator still needs to access csdl stuff, so stopping the recorder causes issues
# recorder.stop()

sim = csdl.experimental.PySimulator(recorder)

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