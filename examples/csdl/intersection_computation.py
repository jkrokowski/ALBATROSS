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

delta = 0.2

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
boundary_labels_B['top'] = H/2
boundary_labels_B['bottom'] = -H/2
mesh_B.geometry.x[:, :2] -= .5      #center at 0
mesh_B.geometry.x[:, 0] *= tw       #scale x
mesh_B.geometry.x[:, 1] *= W        #scale y
mesh_B.geometry.x[:,0] += delta     #translate horizontally for test
mesh_B.name = 'mesh_B'

with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_B.name+".xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh_B)

mesh_C = create_unit_square(MPI.COMM_WORLD, 10, 10,cell_type=CellType.quadrilateral)
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

    return boundary_splines

boundary_splines_A = fit_boundary_b_splines(mesh_A,node_labels_A)
boundary_splines_B = fit_boundary_b_splines(mesh_B,node_labels_B)
boundary_splines_C = fit_boundary_b_splines(mesh_C,node_labels_C)

class SignedDistanceFunction():
    def __init__(self,msh,boundary_splines):
        '''
        pass in the series of b-spline edges
        '''
        self.msh = msh
        self.boundary_splines = boundary_splines


    def evaluate(self,eval_pts):
        d_list = []
        w_list = []
        # W = csdl.Variable(shape=eval_pts.shape)
        for spline in list(self.boundary_splines.values()):
            #compute squared distance for each point to each spline
            proj_eval_pts = spline.evaluate(spline.project(eval_pts))
            distance_eval = proj_eval_pts-eval_pts
            d_list.append(csdl.norm(distance_eval,axes=(1,)))

            #compute winding number integral for each evaluation point for this b-spline
            w_list.append(self._winding_number_for_spline_segment(spline,eval_pts,num_parametric=25))


        D = csdl.minimum(csdl.vstack(d_list),rho=100000,axes=(0,))
        W = csdl.sum(csdl.vstack(w_list),axes=(0,))

        # #compute winding number for each evaluation point
        # W = csdl.Variable(shape=eval_pts)
        # for spline in self.boundary_splines:
        #     W += self._winding_number_for_spline_segment(eval_pts)

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
        # w = csdl.Variable(shape=(eval_pts.shape[0],))
        # function_values = csdl.Variable(shape=(eval_pts.shape[0],num_parametric))
        fxn_values = self._winding_kernel(x_t,eval_pts,tangents) #input parametric point vals & eval pts , output (i,j) shaped fxn values
        # for i in csdl.frange(eval_pts.shape[0]):
            #TODO: replace this with a vectorized function evaluation:
            # for j in csdl.frange(x_t.shape[0]):
            # fxn_values = self._winding_kernel(x_t,eval_pts,tangents)
            # w[i] = h * (0.5*function_values[i,0] 
            #             + csdl.sum(function_values[i,1:-1])
            #             + 0.5*function_values[i,-1]      )
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
        denom = csdl.norm(d,axes=(2,))
        return cross/csdl.square(denom)
        
class SignedDistanceIntersection():
    def __init__(self,sdfs):
        self.sdfs =sdfs
    
    def evaluate(self,eval_pts):
        phi_A = self.sdfs[0].evaluate(eval_pts)
        phi_B = self.sdfs[1].evaluate(eval_pts)

        return csdl.maximum(phi_A,phi_B,rho=100000)
    

phi_A = SignedDistanceFunction(mesh_A,boundary_splines_A)

phi_B = SignedDistanceFunction(mesh_B,boundary_splines_B)


#Points should be   A: INSIDE, OUTSIDE, OUTSIDE, OUTSIDE, INSIDE
#                   B: INSIDE, INSIDE, OUTSIDE, OUTSIDE, OUTSIDE
eval_pts = csdl.Variable(value= np.array([  [0.0,   0.45],
                                            [0.0,   0.0],
                                            [.1,    0.52],
                                            [-.51,  0.45],
                                            [0.25,  0.48]]))

signed_distance_A = phi_A.evaluate(eval_pts)
signed_distance_B = phi_B.evaluate(eval_pts)

signed_distance_intersection = csdl.maximum(signed_distance_A,signed_distance_B,rho=100000)

dphindx = csdl.derivative(signed_distance_intersection,eval_pts)

#seems to work great!
# in order to update mortar mesh boundary nodes, use:
dphindx_norm = csdl.norm(dphindx,axes=(1,))
step = dphindx/csdl.expand(dphindx_norm,dphindx.shape,'i->ij')
new_eval_pts = eval_pts - csdl.matvec(step.T(),signed_distance_intersection).reshape(eval_pts.shape[0],2)


#test with mesh C boundary points:
mesh_C_boundary_pts = csdl.Variable(value=mesh_C.geometry.x[node_labels_C['boundary'],:2])

signed_distance_A = phi_A.evaluate(mesh_C_boundary_pts)
signed_distance_B = phi_B.evaluate(mesh_C_boundary_pts)

phi_C = SignedDistanceIntersection([phi_A,phi_B])
    
# signed_distance_intersection = csdl.maximum(signed_distance_A,signed_distance_B,rho=100000)
signed_distance_intersection = phi_C.evaluate(mesh_C_boundary_pts)
#use derivative of SDF w.r.t. boundary nodes to compute new boundary node points
dphindxc = csdl.derivative(signed_distance_intersection,mesh_C_boundary_pts)
dphindxc_norm = csdl.norm(dphindxc,axes=(1,))
step_c = dphindxc/csdl.expand(dphindxc_norm,dphindxc.shape,'i->ij')

new_mortar_mesh_pts = mesh_C_boundary_pts - csdl.matvec(step_c.T(),signed_distance_intersection).reshape(mesh_C_boundary_pts.shape[0],2)

# #update mortar mesh boundary nodes and output
# mesh_C.geometry.x[node_labels_C['boundary'],:2] = new_mortar_mesh_pts.value

# with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_boundary_update.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh_C)


#===== CONSTRUCT UPDATE TO B-SPLINE COEFFICIENTS (LEFT EDGE) =========# 

parametric_coords = np.array([(i,) for i in np.linspace(0,1,node_labels_C['left'].shape[0])])
# spline_space = lfs.BSplineSpace(1,(3,),(5,)) #TODO: update with the orginal spline space from the spline construction
# spline_coeffs = spline_space.fit(values = new_mortar_mesh_pts,parametric_coordinates= parametric_coords)
# physical_coords = boundary_splines_C['left'].evaluate(parametric_coords)
basis_mat = boundary_splines_C['left'].space.compute_basis_matrix(parametric_coords).toarray()
physical_points = basis_mat@boundary_splines_C['left'].coefficients.value

mesh_C_left_boundary_pts = csdl.Variable(value=mesh_C.geometry.x[node_labels_C['left'],:2])

#equivalent expressions:
#boundary_splines_C['left'].evaluate(parametric_coords) == mesh_C_left_boundary_pts
# phi_k = phi_C.evaluate(mesh_C_left_boundary_pts)
mesh_C_left_boundary_pts_from_spline=boundary_splines_C['left'].evaluate(parametric_coords)
phi_k = phi_C.evaluate(mesh_C_left_boundary_pts_from_spline)

grad_phi_k  = csdl.derivative(phi_k,mesh_C_left_boundary_pts_from_spline)
#construct "duplicated" basis matrix
Nxy = np.kron(basis_mat,np.eye((2)))

A = grad_phi_k@Nxy
Anp = A.value

dP,_,_,_ =np.linalg.lstsq(Anp,-phi_k.value)


#===== CONSTRUCT UPDATE TO B-SPLINE COEFFICIENTS (TOP EDGE) =========# 
parametric_coords = np.array([(i,) for i in np.linspace(0,1,node_labels_C['top'].shape[0])])
# spline_space = lfs.BSplineSpace(1,(3,),(5,)) #TODO: update with the orginal spline space from the spline construction
# spline_coeffs = spline_space.fit(values = new_mortar_mesh_pts,parametric_coordinates= parametric_coords)
# physical_coords = boundary_splines_C['top'].evaluate(parametric_coords)
basis_mat = boundary_splines_C['top'].space.compute_basis_matrix(parametric_coords).toarray()
physical_points = basis_mat@boundary_splines_C['top'].coefficients.value

mesh_C_top_boundary_pts = csdl.Variable(value=mesh_C.geometry.x[node_labels_C['top'],:2])

#equivalent expressions:
#boundary_splines_C['top'].evaluate(parametric_coords) == mesh_C_top_boundary_pts
# phi_k = phi_C.evaluate(mesh_C_top_boundary_pts)
Nprime = boundary_splines_C['top'].space.compute_basis_matrix(parametric_coords,parametric_derivative_orders =(1)).toarray()
Nprimexy = np.kron(Nprime,np.eye((2)))

# weird check of the iterative approach:
for i in range(5):
    mesh_C_top_boundary_pts_from_spline = boundary_splines_C['top'].evaluate(parametric_coords)
    phi_k = phi_C.evaluate(mesh_C_top_boundary_pts_from_spline)

    grad_phi_k  = csdl.derivative(phi_k,mesh_C_top_boundary_pts_from_spline)
    #construct "duplicated" basis matrix

    A = grad_phi_k@Nxy
    Anp = A.value
    alpha = 1
    dP,_,_,_ =np.linalg.lstsq(Anp,-phi_k.value)

    boundary_splines_C['top'].coefficients.value += alpha*dP.reshape(5,2)

    #"Turning" enforcement
    tangents = boundary_splines_C['top'].evaluate(parametric_coords,parametric_derivative_orders =(1))
    normals_k = grad_phi_k.value.reshape(11,11,2)[np.arange(11),np.arange(11),:]
    r_k = np.sum(tangents.value*normals_k,axis=1)
    B =  grad_phi_k.value@Nprimexy
    eta = 1e-3
    alpha = .1
    A_aug = np.vstack([Anp,np.sqrt(eta)*B])
    b_aug = np.hstack([-alpha*phi_k.value,-np.sqrt(eta)*r_k])
    dP_tan, *_ = np.linalg.lstsq(A_aug,b_aug)

    dP_tan2, *_ = np.linalg.lstsq(B,r_k)

    # boundary_splines_C['top'].coefficients.value += dP_tan.reshape(5,2)
    

    turning_val = (phi_k.value@grad_phi_k.value).reshape(11,2)@np.array([[0,-1],[1,0]])
    #1 - 1/10
    lk= csdl.norm(mesh_C_top_boundary_pts_from_spline[1:,:]-mesh_C_top_boundary_pts_from_spline[:-1,:],axes=(1,)).value
    # H = np.sum(lmlbar)*np.eye(lmlbar.shape[0])-np.average(lmlbar)

    Tx = ( np.hstack([-np.diag(tangents[:-1,0].flatten().value/lk,0),np.zeros((tangents[:-1,0].value.shape[0],1))])
            + np.diag(tangents[1:,0].flatten().value/lk,1)[:-1,:] )
    Ty = ( np.hstack([-np.diag(tangents[:-1,1].flatten().value/lk,0),np.zeros((tangents[:-1,1].value.shape[0],1))])
            + np.diag(tangents[1:,1].flatten().value/lk,1)[:-1,:] )
    # Tx = ( np.hstack([np.diag(-tangents[:-1,0].flatten().value,0),np.zeros((tangents[:-1,0].value.shape[0],1))])
    #         + np.diag(tangents[1:,0].flatten().value,1)[:-1,:] )
    # Ty = ( np.hstack([np.diag(-tangents[:-1,1].flatten().value,0),np.zeros((tangents[:-1,1].value.shape[0],1))])
    #         + np.diag(tangents[1:,1].flatten().value,1)[:-1,:] )
    # Tx = ( np.hstack([np.diag(-turning_val[:-1,0].flatten(),0),np.zeros((turning_val[:-1,0].shape[0],1))])
    #         + np.diag(turning_val[1:,0].flatten(),1)[:-1,:] )
    # Ty = ( np.hstack([np.diag(-turning_val[:-1,1].flatten(),0),np.zeros((turning_val[:-1,1].shape[0],1))])
    #         + np.diag(turning_val[1:,1].flatten(),1)[:-1,:] )
    # Ty = np.diag(-tangents[:-1,1].flatten().value,0)+np.diag(tangents[1:,1].flatten().value,1)[:-1,:]
    Jlx = np.kron(Tx,np.array([[1,0]]))+np.kron(Ty,np.array([[0,1]]))
    
    
    # B = H@Jlx@Nxy
    B = Jlx@Nxy
    # print()
    eta = 1e-9
    alpha = 1
    # LHS = Anp.T@Anp #+ eta*B.T@B
    # RHS = -alpha*Anp.T@phi_k.value #-eta*B.T@lmlbar
    A_aug = np.vstack([Anp,np.sqrt(eta)*B])
    b_aug = np.hstack([-alpha*phi_k.value,-np.sqrt(eta)*np.log(lk)])
    dP_equi, *_ = np.linalg.lstsq(A_aug,b_aug)

    # LHS = Anp.T@Anp #+ eta*B.T@B
    # RHS = -alpha*Anp.T@phi_k.value #-eta*B.T@lmlbar
    # dP_equi = np.linalg.solve(LHS,RHS)
    
    # boundary_splines_C['top'].coefficients.value += dP_equi.reshape(5,2)
    # new_mortar_mesh_pts = basis_mat@(boundary_splines_C['top'].coefficients.value + dP_equi.reshape(5,2))

    # boundary_splines_C['top'].coefficients.value += dP.reshape(5,2)
    new_mortar_mesh_pts = boundary_splines_C['top'].evaluate(parametric_coords).value

    #update mortar mesh boundary nodes and output
    mesh_C.geometry.x[node_labels_C['top'],:2] = new_mortar_mesh_pts

    with XDMFFile(MPI.COMM_WORLD, "output/sdf_test_"+mesh_C.name+"_step_"+str(i)+"_boundary_update.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh_C)

print()