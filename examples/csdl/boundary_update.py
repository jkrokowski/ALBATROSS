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

delta = +0.18

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
boundary_labels_B['left'] =-tf/2 + delta
boundary_labels_B['right'] = tf/2 + delta
boundary_labels_B['top'] = H/2 #+ delta
boundary_labels_B['bottom'] = -H/2 #+ delta
mesh_B.geometry.x[:, :2] -= .5      #center at 0
mesh_B.geometry.x[:, 0] *= tw       #scale x
mesh_B.geometry.x[:, 1] *= W        #scale y
mesh_B.geometry.x[:,0] += delta     #translate horizontally for test
# mesh_B.geometry.x[:,1] += delta     #translate vertically for test
mesh_B.name = 'mesh_B'

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_B.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_B)

mesh_C = create_unit_square(MPI.COMM_WORLD, 8, 8,cell_type=CellType.quadrilateral)
boundary_labels_C = {}
boundary_labels_C['left'] =-tf/2
boundary_labels_C['right'] = tf/2
boundary_labels_C['top'] = H/2
boundary_labels_C['bottom'] = H/2-tw/2
mesh_C.geometry.x[:, :2] -= .5      #center at 0
mesh_C.geometry.x[:, 0] *= tw       #scale x
mesh_C.geometry.x[:, 1] *= tf         #scale y
mesh_C.geometry.x[:, 1] += H/2 - tf/2   #translate vertically
mesh_C.name = 'mesh_C'

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)

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
node_labels_C = get_labeled_nodes(mesh_C,boundary_labels_C)

def orientation_xy(points):
    # points: array-like of shape (n,2); will be treated as closed (last connects to first)
    pts = np.asarray(points)
    x, y = pts[:,0], pts[:,1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    A2 = np.sum(x * y_next - x_next * y)  # equals 2*A
    return "CCW" if A2 > 0 else "CW" if A2 < 0 else "degenerate/collinear"

def order_edge_like_boundary(ordered_boundary_nodes: np.ndarray,
                             edge_nodes: np.ndarray) -> np.ndarray:
    """
    Reorder `edge_nodes` to match the traversal order of `ordered_boundary_nodes`.
    Works even if the edge straddles the boundary wrap-around (end -> start).

    Assumptions:
      - All edge_nodes are present in ordered_boundary_nodes (unique IDs).
      - Edge is a contiguous arc along the boundary.
    """
    # Map node id -> position along the global boundary
    pos = {int(n): i for i, n in enumerate(ordered_boundary_nodes)}
    idx = np.array([pos[int(n)] for n in edge_nodes], dtype=int)

    # Sort by boundary position (still may be split by wrap-around)
    order = np.argsort(idx)
    nodes_sorted = edge_nodes[order]
    idx_sorted = idx[order]

    # Detect wrap-around break and rotate so the sequence is contiguous
    N = len(ordered_boundary_nodes)
    # modular forward gaps, including last->first (wrap)
    gaps = (np.diff(idx_sorted, append=idx_sorted[0] + N)) % N
    # the biggest gap is the cut between segments; start after it
    cut = np.argmax(gaps)
    nodes_ccw = np.concatenate([nodes_sorted[cut+1:], nodes_sorted[:cut+1]])

    return nodes_ccw  # in the boundary's forward (CCW) order if boundary is CCW

def fit_boundary_b_splines(msh,node_labels):
    '''
    return a set of b-splines that form a closed loop for the rectangle
    '''
    #use the same b-spline space for all edges
    num_parametric = 5
    bspline_degree=3
    spline_space = lfs.BSplineSpace(1,(bspline_degree,),(num_parametric,))
    ordering = ALBATROSS.utils.order_boundary_nodes(msh.geometry.x[node_labels['boundary'],0:2])
    ordered_boundary_vertices = node_labels['boundary'][ordering]


    #check for CCW or CW:
    direction = orientation_xy(msh.geometry.x[ordered_boundary_vertices,:2])

    if direction == 'CW':
        ordering = ordering[::-1]
        ordered_boundary_vertices = node_labels['boundary'][ordering]
    
    boundary_splines = {}
    edges = ['left','right','top','bottom']
    for edge in edges:
        nodes = node_labels[edge]

        parametric_coords = np.array([(i,) for i in np.linspace(0,1,nodes.shape[0])])
        #TODO: need to check if this always returns points ordered in the same winding directions
        # inverse_ordering = np.argsort(ordering)
        #TODO: need to check that the starting point is properly included in both edges it belongs to
        ordered_nodes = order_edge_like_boundary(ordered_boundary_vertices,nodes)
        points = msh.geometry.x[ordered_nodes,0:2]
        edge_spline_coeffs = spline_space.fit(values = points,parametric_coordinates= parametric_coords)
        edge_spline = lfs.Function(spline_space,edge_spline_coeffs,name=edge+'_spline')
        boundary_splines[edge] = edge_spline

    return boundary_splines,ordered_boundary_vertices

boundary_splines_A,boundary_order_A = fit_boundary_b_splines(mesh_A,node_labels_A)
boundary_splines_B,boundary_order_B  = fit_boundary_b_splines(mesh_B,node_labels_B)
boundary_splines_C,boundary_order_C  = fit_boundary_b_splines(mesh_C,node_labels_C)

class SignedDistanceFunction():
    def __init__(self,msh,boundary_splines,rho=100000):
        '''
        pass in the series of b-spline edges
        '''
        self.msh = msh
        self.boundary_splines = boundary_splines
        self.rho = rho


    def evaluate(self,eval_pts):
        d_list = []
        w_list = []

        for spline in list(self.boundary_splines.values()):
            #compute squared distance for each point to each spline
            proj_eval_pts = spline.evaluate(spline.project(eval_pts)).reshape(eval_pts.shape)
            distance_eval = proj_eval_pts-eval_pts
            d_list.append(csdl.norm(distance_eval,axes=(1,)))

            #compute winding number integral for each evaluation point for this b-spline
            w_list.append(self._winding_number_for_spline_segment(spline,eval_pts,num_parametric=50))


        D = csdl.minimum(csdl.vstack(d_list),rho=self.rho,axes=(0,))
        W = csdl.sum(csdl.vstack(w_list),axes=(0,))

        #get sign from winding number
        sign = -csdl.tanh(100*(csdl.absolute(W)-0.5))

        return sign*D
    
    def _winding_number_for_spline_segment(self,spline,eval_pts,num_parametric=10):
        '''
        use the trapezoidal rule to approximate the winding number integral for each spline
        '''
        t = np.linspace(0,1,num_parametric)
        x_t = spline.evaluate(t).reshape(num_parametric,2)
        tangents = spline.evaluate(t,parametric_derivative_orders =(1)).reshape(num_parametric,2)
        h = 1/(num_parametric-1) #interval length is in parametric space so 1-0 = 1 for numerator
        
        #evaluate winding kernel
        fxn_values = self._winding_kernel(x_t,eval_pts,tangents) #input parametric point vals & eval pts , output (i,j) shaped fxn values
        
        #use trapezoidal rule to integrate and find winding number
        w = (h/(2*np.pi)) * (0.5*fxn_values[:,0] 
                        + csdl.sum(fxn_values[:,1:-1],axes=(1,))
                        + 0.5*fxn_values[:,-1]      )

        return w

    def _winding_kernel(self,x_in,pts_in,tangents_in):
        '''
        x_t: physical point along spline at parametric location t
        p: physical point to compute distance
        tangent: tangent vector along spline at parametric location t
        '''
        x = csdl.expand(x_in,(pts_in.shape[0],x_in.shape[0],2),'jk->ijk')
        t = csdl.expand(tangents_in,(pts_in.shape[0],x_in.shape[0],2),'jk->ijk')
        p = csdl.expand(pts_in,(pts_in.shape[0],x_in.shape[0],2),'ik->ijk')
        d = x-p
        cross = d[:,:,0]*t[:,:,1] - d[:,:,1]*t[:,:,0]
        eps = 1e-8 #prevent numerical blowups from dividing by the square of a very small number
        denom = csdl.norm(d,axes=(2,))
        return cross/(csdl.square(denom)+eps)
        
class SignedDistanceIntersection():
    def __init__(self,sdfs,rho=10000):
        self.sdfs =sdfs
        self.rho = rho
    
    def evaluate(self,eval_pts):
        phi_A = self.sdfs[0].evaluate(eval_pts)
        phi_B = self.sdfs[1].evaluate(eval_pts)

        return csdl.maximum(phi_A,phi_B,rho=self.rho)
    

phi_A = SignedDistanceFunction(mesh_A,boundary_splines_A,rho=100000)

phi_B = SignedDistanceFunction(mesh_B,boundary_splines_B,rho=100000)

#test with mesh C boundary points:
mesh_C_boundary_pts = csdl.Variable(value=mesh_C.geometry.x[boundary_order_C,:2])

signed_distance_A = phi_A.evaluate(mesh_C_boundary_pts)
signed_distance_B = phi_B.evaluate(mesh_C_boundary_pts)

phi_C = SignedDistanceIntersection([phi_A,phi_B],rho=100)

#test for a single evaluation pt:
phi_C.evaluate(mesh_C_boundary_pts[0].reshape(1,2))


# This is the naive projection
signed_distance_intersection = phi_C.evaluate(mesh_C_boundary_pts)
#use derivative of SDF w.r.t. boundary nodes to compute new boundary node points
dphindxc = csdl.derivative(signed_distance_intersection,mesh_C_boundary_pts)
dphindxc_norm = csdl.norm(dphindxc,axes=(1,))
step_c = dphindxc/csdl.expand(dphindxc_norm,dphindxc.shape,'i->ij')

new_mortar_mesh_pts = mesh_C_boundary_pts - csdl.matvec(step_c.T(),signed_distance_intersection).reshape(mesh_C_boundary_pts.shape[0],2)

# #update mortar mesh boundary nodes and output
# mesh_C.geometry.x[boundary_order_C,:2] = new_mortar_mesh_pts.value

# with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update_smooth.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh_C)

def project_to_new_boundary(phi,eval_pts,step_size=1.0,delta=0.0):
    signed_distance = phi.evaluate(eval_pts)
    delta = csdl.Variable(value = delta*np.ones_like(signed_distance))
    dSDdx=csdl.derivative(signed_distance,eval_pts)
    dSDdx_norm = csdl.norm(dSDdx,axes=(1,))
    step = dSDdx/csdl.expand(dSDdx_norm,dSDdx.shape,'i->ij')

    projected_pts = eval_pts - step_size*csdl.matvec(step.T(),signed_distance-delta).reshape(eval_pts.shape[0],2)

    return projected_pts

def return_SDF_normal(phi,eval_pts,independence=False):
    signed_distance = phi.evaluate(eval_pts)
    dSDdx=csdl.derivative(signed_distance,eval_pts)
    dSDdx_norm = csdl.norm(dSDdx,axes=(1,))
    if independence:
        size = signed_distance.shape[0]
        dSDdx = dSDdx.reshape(size**2,2)[list(np.arange((size))*size+np.arange(size)),:]
    step = dSDdx/csdl.expand(dSDdx_norm,dSDdx.shape,'i->ij')
    return step

anchor_point= csdl.Variable(value = mesh_C.geometry.x[boundary_order_C[0],:2].reshape(1,2))
x_k = csdl.Variable(value=mesh_C.geometry.x[boundary_order_C,:2])
R = np.array([[0,-1],[1,0]])
delta = 0.02
lk = csdl.norm(x_k[1:,:]-x_k[:-1,:],axes=(1,))
lbar = csdl.average(lk)
x_k = x_k.set(csdl.slice[0:1,:2],project_to_new_boundary(phi_C,anchor_point,delta=delta))
#TODO: seems like the .project() and .evaluate() for the b-splines in the SDFs are likely suspect
#       as a result, the effect of any change in the projected points is not propogated. 
# for ind in csdl.frange(1,10):
# for ind in csdl.frange(1,x_k.shape[0]):
for ind in range(1,x_k.shape[0]):
    n_k = return_SDF_normal(phi_C,x_k[ind-1].reshape(1,2))
    t_k = n_k@R.T 

    #predictor (take a step in the tangent direction)
    xhat_k = (x_k[ind-1].reshape(1,2) + lbar * t_k) #may need sign correction?
    
    #corrector (project back to boundary)
    x_proj = project_to_new_boundary(phi_C,xhat_k,delta=delta).reshape(2)
    x_k = x_k.set(csdl.slice[ind],x_proj)

#update mortar mesh boundary nodes and output
mesh_C.geometry.x[boundary_order_C,:2] = x_k

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)

lk_ = np.linalg.norm(np.roll(mesh_C_boundary_pts.value,-1,axis=0)-mesh_C_boundary_pts.value,axis=1)


x_flatty = csdl.ImplicitVariable(name='x_update',value=x_k.flatten())
x_update = x_flatty.reshape(x_k.shape)
e_k = csdl.vstack([x_update[1:,:],x_update[:1,:]]) - x_update

n_k = return_SDF_normal(phi_C,csdl.vstack([x_update[0],x_update[0]]))[0,:2]
t_k = R @n_k
# n_k = return_SDF_normal(phi_C,csdl.vstack([x_update[0],x_update[0]]),independence=True)
# n_k = return_SDF_normal(phi_C,x_update,independence=True)
# n_k = return_SDF_normal(phi_C,x_update)
# t_k = n_k @ R
lk = csdl.norm(e_k,axes=(1,))

#numpy version:
# np.linalg.norm(np.einsum('ij,ik,ik->ij', t_k.value, t_k.value, e_k.value),axis=1)
# l_perp = csdl.norm(csdl.einsum(t_k, t_k, e_k,action='ij,ik,ik->ij'),axes=(1,))
# l_perp = csdl.einsum( t_k, e_k,action='ij,ij->i')
level_set_value = 0.02
residual_boundary = phi_C.evaluate(x_update) -level_set_value
# residual_spacing =e_k l_perp-csdl.average(l_perp)
# residual_spacing = l_perp[1:]-l_perp[0]
# residual_spacing = l_perp[0]-l_perp[1:]
# residual_spacing = l_perp-lbar
# residual_edge = lbar - csdl.average(lk)
residual_spacing = lk[1:]-lk[0]
# residual_bc = t_k[0].reshape(1,2) @ (x_update[0]-anchor_point_update[0])
residual_bc = t_k.reshape(1,2) @ (x_update[0]-anchor_point_update[0])

residual_boundary.add_name('delta_level_set')
residual_spacing.add_name('boundary_spacing')
residual_bc.add_name('anchor_point')

system_residual = csdl.concatenate([residual_boundary,
                               residual_spacing,
                               residual_bc])
system_residual.add_name('boundary_respacing')

# solver = csdl.nonlinear_solvers.GaussSeidel('boundary_redistribution',max_iter=1,tolerance=1e-3)
solver = csdl.nonlinear_solvers.Newton('boundary_redistribution',max_iter=1,tolerance=1e-3)
solver.add_state(x_flatty,system_residual)
# solver.add_state(state,system_residual)

solver.run()


#COMPUTE TANGENTS BETWEEN POINTS:
# x_k = mesh_C_boundary_pts.value
x_kp1 = np.roll(x_k,-1,axis=0)
e_k = x_kp1 - x_k
lk = np.linalg.norm(e_k,axis=1)
lbar = np.average(lk)
ehat_k = e_k/lk.reshape(40,1)

xcsdl = csdl.Variable(value = x_kp1)
n_k_full = return_SDF_normal(phi_C,xcsdl).value
#just normals at each point:
n_k = np.zeros((x_kp1.shape[0],2))
for i in range(x_kp1.shape[0]):
    n_k[i,:] = n_k_full[i,2*i:2*i+2]
t_k = (R@n_k.T).T

#only have jacobian entries corresponding to nodes that should move
#   Nodes 0 and N should not move (same node due to closed loop)
A = np.zeros((x_kp1.shape[0],(x_kp1.shape[0]-1)*2))
for i in range(1,x_kp1.shape[0]-1):
    P_k = t_k[i,:].reshape(2,1)@t_k[i,:].reshape(1,2)
    A[i,[i-1,i-1+(x_kp1.shape[0]-1)]] = -ehat_k[i].reshape(1,2)@P_k
    
    P_k = t_k[i+1,:].reshape(2,1)@t_k[i+1,:].reshape(1,2)
    A[i,[i,i+(x_kp1.shape[0]-1)]] = ehat_k[i+1].reshape(1,2)@P_k

#start
P_k = t_k[0,:].reshape(2,1)@t_k[0,:].reshape(1,2)
A[0,[0,(x_kp1.shape[0]-1)]] = ehat_k[0].reshape(1,2)@P_k

#end
P_k = t_k[-2,:].reshape(2,1)@t_k[-2,:].reshape(1,2)
A[x_kp1.shape[0]-1,[x_kp1.shape[0]-2,2*(x_kp1.shape[0]-1)-1]] = ehat_k[-2].reshape(1,2)@P_k


delta_lk = lbar-lk

#least squares fit
dT, *_ = np.linalg.lstsq(A,-delta_lk)

#take a step in dT, then reproject to the delta level-set, then relinearize and solve again until convergence
# dT = np.linalg.solve(A.T@A+1e-6*np.eye(78),-A.T@delta_lk)

x_respace = x_k[1:] + 0.025*dT.reshape(2,39).T

#update mortar mesh boundary nodes and output
mesh_C.geometry.x[boundary_order_C[1:],:2] = x_respace

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update_tangent_respace.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)

x_respace_csdl = csdl.Variable(value=x_respace)
x_reproject = project_to_new_boundary(phi_C,x_respace_csdl,delta=0.02)


#update mortar mesh boundary nodes and output
mesh_C.geometry.x[boundary_order_C[1:],:2] = x_reproject.value

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update_reproject.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)


#####
# check residual
x_k = np.vstack([mesh_C.geometry.x[boundary_order_C[0],:2],x_reproject.value])
x_kp1 = np.roll(x_k,-1,axis=0)
e_k = x_kp1 - x_k
lk = np.linalg.norm(e_k,axis=1)
lbar = np.average(lk)
delta_lk =lbar-lk
res=np.linalg.norm(delta_lk)
print('residual norm:',res)
####

#COMPUTE TANGENTS BETWEEN POINTS:
# x_k = mesh_C_boundary_pts.value
x_k = np.vstack([mesh_C.geometry.x[boundary_order_C[0],:2],x_reproject.value])
x_kp1 = np.roll(x_k,-1,axis=0)
e_k = x_kp1 - x_k
lk = np.linalg.norm(e_k,axis=1)
lbar = np.average(lk)
ehat_k = e_k/lk.reshape(40,1)

xcsdl = csdl.Variable(value = x_kp1)
n_k_full = return_SDF_normal(phi_C,xcsdl).value
#just normals at each point:
n_k = np.zeros((x_kp1.shape[0],2))
for i in range(x_kp1.shape[0]):
    n_k[i,:] = n_k_full[i,2*i:2*i+2]
t_k = (R@n_k.T).T

#only have jacobian entries corresponding to nodes that should move
#   Nodes 0 and N should not move (same node due to closed loop)
A = np.zeros((x_kp1.shape[0],(x_kp1.shape[0]-1)*2))
for i in range(1,x_kp1.shape[0]-1):
    P_k = t_k[i,:].reshape(2,1)@t_k[i,:].reshape(1,2)
    A[i,[i-1,i-1+(x_kp1.shape[0]-1)]] = -ehat_k[i].reshape(1,2)@P_k
    
    P_k = t_k[i+1,:].reshape(2,1)@t_k[i+1,:].reshape(1,2)
    A[i,[i,i+(x_kp1.shape[0]-1)]] = ehat_k[i+1].reshape(1,2)@P_k

#start
P_k = t_k[0,:].reshape(2,1)@t_k[0,:].reshape(1,2)
A[0,[0,(x_kp1.shape[0]-1)]] = ehat_k[0].reshape(1,2)@P_k

#end
P_k = t_k[-2,:].reshape(2,1)@t_k[-2,:].reshape(1,2)
A[x_kp1.shape[0]-1,[x_kp1.shape[0]-2,2*(x_kp1.shape[0]-1)-1]] = ehat_k[-2].reshape(1,2)@P_k


delta_lk = lbar-lk

#least squares fit
# dT, *_ = np.linalg.lstsq(A,-delta_lk)

#take a step in dT, then reproject to the delta level-set, then relinearize and solve again until convergence
# dT = np.linalg.solve(A.T@A+1e-4*np.eye(78),A.T@delta_lk)

x_respace = x_k[1:] + 0.025*dT.reshape(2,39).T

#update mortar mesh boundary nodes and output
mesh_C.geometry.x[boundary_order_C[1:],:2] = x_respace

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update_tangent_respace2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)

x_respace_csdl = csdl.Variable(value=x_respace)
x_reproject = project_to_new_boundary(phi_C,x_respace_csdl,delta=0.02)


#update mortar mesh boundary nodes and output
mesh_C.geometry.x[boundary_order_C[1:],:2] = x_reproject.value

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update_reproject2.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_C)

#####
# check residual
x_k = np.vstack([mesh_C.geometry.x[boundary_order_C[0],:2],x_reproject.value])
x_kp1 = np.roll(x_k,-1,axis=0)
e_k = x_kp1 - x_k
lk = np.linalg.norm(e_k,axis=1)
lbar = np.average(lk)
delta_lk =lbar-lk
res=np.linalg.norm(delta_lk)
print('residual norm:',res)
####

print()



















