import cffi
from petsc4py.PETSc import ScalarType
from dolfinx import fem,mesh
import numpy as np

import os
os.environ['SCIPY_USE_PROPACK'] = "1"

from scipy.sparse import lil_matrix,csgraph,csr_matrix,coo_matrix,dok_matrix
from FROOT_BAT.utils import get_vtx_to_dofs
import ufl
from mpi4py import MPI



class LocalAssembler():
    def __init__(self, form):
        self.form = fem.form(form)
        self.update_coefficients()
        self.update_constants()
        subdomain_ids = self.form.integral_ids(fem.IntegralType.cell)
        assert(len(subdomain_ids) == 1)
        assert(subdomain_ids[0] == -1)
        is_complex = np.issubdtype(ScalarType, np.complexfloating)
        nptype = "complex128" if is_complex else "float64"
        self.kernel = getattr(self.form.ufcx_form.integrals(fem.IntegralType.cell)[0], f"tabulate_tensor_{nptype}")
        self.active_cells = self.form.domains(fem.IntegralType.cell, -1)
        assert len(self.form.function_spaces) == 2
        self.local_shape = [0,0]
        for i, V in enumerate(self.form.function_spaces):
            self.local_shape[i] = V.dofmap.dof_layout.block_size * V.dofmap.dof_layout.num_dofs

        e0 = self.form.function_spaces[0].element
        e1 = self.form.function_spaces[1].element
        needs_transformation_data = e0.needs_dof_transformations or e1.needs_dof_transformations or \
            self.form.needs_facet_permutations
        if needs_transformation_data:
            raise NotImplementedError("Dof transformations not implemented")

        self.ffi = cffi.FFI()
        V = self.form.function_spaces[0]
        self.x_dofs = V.mesh.geometry.dofmap

    def update_coefficients(self):
        self.coeffs = fem.assemble.pack_coefficients(self.form)[(fem.IntegralType.cell, -1)]

    def update_constants(self):
        self.consts = fem.assemble.pack_constants(self.form)

    def update(self):
        self.update_coefficients()
        self.update_constants()

    def assemble_matrix(self, i:int):

        x = self.form.function_spaces[0].mesh.geometry.x
        x_dofs = self.x_dofs.links(i)
        geometry = np.zeros((len(x_dofs), 3), dtype=np.float64)
        geometry[:, :] = x[x_dofs]

        A_local = np.zeros((self.local_shape[0], self.local_shape[0]), dtype=ScalarType)
        facet_index = np.zeros(0, dtype=np.intc)
        facet_perm = np.zeros(0, dtype=np.uint8)
        if self.coeffs.shape == (0, 0):
            coeffs = np.zeros(0, dtype=ScalarType)
        else:
            coeffs = self.coeffs[i,:]
        ffi_fb = self.ffi.from_buffer
        self.kernel(ffi_fb(A_local), ffi_fb(coeffs), ffi_fb(self.consts), ffi_fb(geometry),
               ffi_fb(facet_index), ffi_fb(facet_perm))
        return A_local

def get_nullspace(domain,ufl_form,fxn_space):
    #build list of element matrices using custom local assembler
    Ae = get_element_matrices(domain,ufl_form)

    # for formulations involving natural groupings of indices, get the
    # relationship between the mesh rigidity and the dof grouping
    #   -allows for a "consistent" extension in the fretsaw construction
    #   -this involves making 'cuts' between the same elements for 
    #    different vector/function directions e.g for 2D linear elasticity,
    #    severing the same connections between x- and y-directions 
    dof_to_G = get_node_to_G_map(domain,fxn_space)

    # get local to global index map
    Ie = get_local_to_global(dof_to_G,fxn_space)

    # # constructs global stiffness matrix
    print('assembling stiffness matrix:')
    A = assemble_stiffness_matrix(ufl_form)

    #extract element connectivity from dolfin mesh object
    print('computing rigidity graph')
    G = get_rigidity_graph(domain)

    #get which elements are connected to each node
    print('getting element to node map')
    V = get_node_to_el_map(domain)
    
    #compute the fretsaw extension (returns a sparse matrix)
    print('computing fretsaw extension')
    from cProfile import Profile
    from pstats import SortKey, Stats
    
    # with Profile() as profile:
    #     print(f"{compute_fretsaw_extension(Ae,Ie,G,V) = }")
    #     (   Stats(profile)
    #         .strip_dirs()
    #         .sort_stats(SortKey.CALLS)
    #         .print_stats() )
    import time
    t0 =time.time()
    # F = compute_fretsaw_extension(Ae,Ie,G,V)
    t1 = time.time()
    #compute the LU factorization of the fretsaw extension
    
    # vhF_sym = sym_LU_inv_iter(F)
    t2 = time.time()
    vhA_sym = sym_LU_inv_iter(A)
    t3 = time.time()
    #utilize LU factors to perform subspace inverse iteration

    #restrict to first n factors and orthogonalize


    #compare nullspace from full matrix and F(A):
    #numpy fxns (numerical conditioning issues)
    # u,s,v = np.linalg.svd(A.toarray())
    # uF,sF,vHF = np.linalg.svd(F.toarray())
    
    print('computing sparse nullspace:')
    #get rank and nullity (ONLY FOR TESTING, DON'T use large matrices)
    n = np.max(Ie) + 1
    # rank = np.linalg.matrix_rank(A.toarray())
    # nullity = A.shape[0]-rank
    nullity=1

    #get nullspace from A
    from scipy.sparse.linalg import svds
    # u,s,vh = svds(A,k=nullity,which='SM',maxiter=100000)
    t4 = time.time()
    # uF,sF,vhF = svds(F,k=nullity,which='SM',maxiter=100000)
    t5 = time.time()
    # u,s,vh = svds(A,k=1,which='SM',solver='propack', return_singular_vectors="vh")
    # uF,sF,vhF = svds(F,k=1,which='SM',solver='propack', return_singular_vectors="vh")

    # vhF_restricted=vhF[-nullity:,:n]
    # nullspace_basis = orthogonalize(vh.T)

    # nullspace_basis_F = orthogonalize(vhF_restricted.T)

    print('fretsaw computation:')
    print(t1-t0)
    print('fretsaw inverse iteration:')
    print(t2-t1)
    print('standard inverse iteration:')
    print(t3-t2)
    print('standard sparse svd:')
    print(t4-t3)
    print('fretsaw sparse svd:')
    print(t5-t4)
    return

def compute_fretsaw_extension(Ae,Ie,G,V):
    '''
    method developed from psuedocode of Shklarski and toledo (2008) Rigidity in FE matrices, Algorithm 1
    
    Ae: list of length (num_elements) of assembled local stiffness matrices of shape (ne x ne)
    
    Ie: list of length (num_elements) of vectors of (num_element_indices) linking global indices to 
        local stiffness matrix element indices

    G: rigidity graph for the mesh detailing element to element connections
    
    V: node to element connectivity (provided for easy subgraph detection)
    '''
    
    #TODO: need to modify this algorithm to handle the 12dim vector construction

    #get number of indices for each local stiffness matrix
    (ne,_) = Ae[0].shape
    #get number of elements from the rigidity graph
    num_el = G.shape[0]
    #get number of matrix indices as the max node marker + 1
    n = np.max(Ie) + 1
    #TODO: get number of natural groupings
    num_groups = 1
    #get number of local stiffness matrices
    k = len(Ae)
    #initialize to last non-zero row of the extension matrices
    r = n-1

    T = compute_spanning_tree(G)
    
    #initialize extension matrices (k total matrices)
    #these will be trimmed later using the final resultant r value
    Qe = [lil_matrix((n+k,n),dtype=np.int8) for i in range(k)]

    for j in range(n):
        #initialize to the identity matrix
        #note that setting the j-th column of Qp to ej is equivalent to Qp[j,j]=1
        Qe[0][j,j]=1

        #find all elements connected to node j and build "connectivity components"
        #connectivity components are all the portions of the spanning tree T that 
        # have elements connected to node j 
        
        #indices of the elements that are connected to the j-th node:
        ele = V[j]
        
        #get the subgraph induced by the elements incident on the j-th node
        Gj=T[ele,:].tocsc()[:,ele]

        #get the number of connected components and their labels:
        n_comp,labels = csgraph.connected_components(Gj)

        #get list of indices for selecting elements in each connectivity component
        idx=[np.where(labels==i) for i in range(n_comp)]

        #get the connectivity components of the subgraph Gj
        Gcj = [ele[idx[i]] for i in range(n_comp)]

        #reorder to make sure "master element" is always in the first connectivity component
        if 0 in ele:
            #TODO: reorder Gcj to have Ae[0] in Gcj[0]
            ele = np.sort(ele)
            
        #set column j of Qp to ej for all elements in the first connectivity 
        # component 
        for p in Gcj[0]:
            #zero out the j-th column and eliminate zeros
            # nonzeros = Qe[p][:,j].nonzero()[0]
            # Qe[p][nonzeros,j] = np.zeros_like(nonzeros)
            
            Qe[p][:,j] = np.zeros((n+k,1))

            #assign ej to column j
            Qe[p][j,j] = 1

        # for each connectivity component after the first one, modify the
        # entries of the appropriate extension matrices:
        for i in range(1,n_comp):
            #increment last non-zero row tracker
            r += 1
            #for all the elements in the given connectivity component Gcj[p],
            #  set the j-th column of Qp to er
            for p in Gcj[i]:
                # set the j-th column of Qp to e_r (e.g. a 1 in the r-th column )
                #first, zero out any nonzero entries in the column
                # nonzeros = Qe[p].tocsc()[:,j].nonzero()[0]
                # Qe[p][nonzeros,j] = np.zeros_like(nonzeros)
                Qe[p][:,j] = np.zeros((n+k,1))

                #set the r-th row pf the j-th column to be 1
                Qe[p][r,j] = 1
        #for each extension matrix 
        # print()
    # ######
    # #convert to dok
    # Qe = [Qe[i].todok() for i in range(k)]
    # for j in range(n):

    # ####    
        for Q,I in zip(Qe,Ie):
            #if ej is not in the nonzero columns of Ai, then set Qi to ej
            #check each NONZERO column for a 1 in the j-th row
            #if no 1 in any row, set j-th column to ej Qi(j,j) = 1
            if j not in I:
                #first, zero out any nonzero entries in the column
                # nonzeros = Q.tocsc()[:,j].nonzero()[0]
                # Q[nonzeros,j] = np.zeros_like(nonzeros)
                # Q[:,j] = np.zeros((n+k,1))
                #set the j-th row to ej
                Q[j,j] = 1
        # print()
    #assemble the fretsaw extension
    #trim to (r x n) and convert to csr for efficient matrix operations
    for i,Q in enumerate(Qe):
        Qe[i] = Q[0:r+1,0:n].tocsr()
    F = csr_matrix((r+1,r+1))

    for A,I,Q in zip(Ae,Ie,Qe):
        A_global = localAe_to_globalAe(A,I,n)
        F += Q.dot(A_global.dot(Q.transpose()))
    #F = sum(Qi@Ai@Qi^T)

    return F

def compute_nullspace(F):
    '''
    F: Fretsaw extension of a finite element problem
    
    returns:
    basis for nullspace of the matrix A and F(A)
    '''
    

def get_rigidity_graph(domain,direction='directed'):
    ''''
    returns the ridigity graph structure for a 2d mesh
    needs the mesh from which the edge to cell connectivity can be created

    returns the rigidity graph in the form of an adjacency matrix
    '''
    #create connectivity (edge=1,cell=2)
    domain.topology.create_connectivity(1,2)
    #create dolfinx adjacency list
    ed_to_el = domain.topology.connectivity(1,2)

    #initialize a sparse adjacency matrix
    num_el = len(domain.geometry.dofmap)

    #initialize sparse matrix
    G = lil_matrix((num_el,num_el),dtype=np.int8)
    for i in range(ed_to_el.num_nodes):
        link = ed_to_el.links(i)
        if len(ed_to_el.links(i)) == 2: 
            #add edge to graph (symmetric b/c undirected)
            G[link[0],link[1]] = 1 
        else:
            continue    

    #since G is undirected, it will be symmetric and we could populate the 
    # lower triangular entries with G += G.T, but it is more memory efficient
    # to only store the entries above the diagonal

    # to return the bidirectional (symmetric) graph, uncomment the line below:
    # G += G.T

    return G

def compute_spanning_tree(G):
    '''
    return a spanning tree T from a rigidity graph
    '''
    T = csgraph.minimum_spanning_tree(G)

    return T

def get_element_matrices(domain,form):
    ''' returns a list of scipy sparse matrices
    However, should this be a dictionary with the key being the cell number?
    This would allow for subdomain analysis'''
    assembler = LocalAssembler(form)
    A_list = []
    for i in range(domain.topology.index_map(domain.topology.dim).size_local):
        A_list.append(assembler.assemble_matrix(i))
    
    return A_list

def get_node_to_el_map(domain):
    #create connectivity (node=0,cell=2)
    domain.topology.create_connectivity(0,2)
    #create dolfinx adjacency list
    nod_to_el = domain.topology.connectivity(0,2)
    V = {}
    for i in range(nod_to_el.num_nodes):
        V[i]=nod_to_el.links(i)
    return V 

def assemble_stiffness_matrix(ufl_form):
    #returns a scipy csr_matrix
    A_mat = fem.petsc.assemble_matrix(fem.form(ufl_form))
    A_mat.assemble()
    return csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)

def get_node_to_G_map(domain,fxn_space):
    #gets the stiffness matrix indices for each node corresponding
    #to each function in a vector element/mixed space

    #this map assists with constructing a consistent extension
    dof_maps = []
    #handle scalar fxn space
    if fxn_space.num_sub_spaces != 0:
        for space in range(fxn_space.num_sub_spaces):
            dof_maps.append(get_vtx_to_dofs(domain,fxn_space.sub(space)))
    #handle vector/mixed fxn spaces
    else:
        for i in range(domain.geometry.x.shape[0]):
            # print(i)
            dof_maps.append(i)
            # dof_maps.append(fxn_space.dofmap.cell_dofs(i))
        # dof_to_G=fxn_space.dofmap

    #assemble into a matrix where columns correspond to each fxn component and
    # rows correspond to nodes
    dof_to_G = np.hstack(dof_maps)
        # to get out set of indices used to restrict the global indices of the 
        # stiffness matrix to the relevant indices for fxn select a column of 
        # dof_to_G  (e.g. G[i,:] will be a (num_nodes,) 1d np array)
    return dof_to_G

def get_local_to_global(dof_to_G,fxn_space):
    #utilize fxn space and dof_to_G to return a list of num_elements vectors of length 
    adj_list = fxn_space.dofmap.list
    Ie = []
    for i in range(adj_list.num_nodes):
        Ie.append(dof_to_G[adj_list.links(i)].flatten())
    return Ie

def localAe_to_globalAe(Ae,Ie,n):
    # print('local A:')
    # print(Ae)
    # print('dof vector:')
    # print(Ie)
    # print()
    data = Ae.flatten()
    row = np.repeat(Ie,Ie.shape[0])
    col = np.tile(Ie,Ie.shape[0])
    Ae_global = csr_matrix((data,(row,col)),shape=(n,n))
    return Ae_global
def sym_LU_inv_iter(A,max_iter=1000):
    #returns approximate null vectors of A
    "A: a sparse matrix"
    from scipy.sparse.linalg import splu,svds
    # lu = splu(A,permc_spec="NATURAL")
    # from scipy.linalg import lu
    # p, l, U = lu(A.A)
    lu = splu(A)
    U = lu.U
    L = lu.L
    # u,s,v = np.linalg.svd(A.A)
    # uU,sU,vU = np.linalg.svd(U.A)
    

    #initialize random X matrix of size n x n_z (where nz is nullity dim)

    #perform symmetric inverse iteration to solve A^T @ A @ x(i) = x(i-1)/||x(i-1)|| for x(i)

    #orthogonalize X matrix
    # x = np.random.rand(U.shape[0],1)
    x = np.ones((U.shape[0]))

    from scipy.sparse.linalg import spsolve_triangular,spsolve
    from scipy.sparse.linalg import norm

    # w = spsolve_triangular(U,x,lower=False)
    # x = w/np.linalg.norm(w)
    tol = 1e-8
    for _ in range(max_iter):
        w = spsolve_triangular(L,x/np.linalg.norm(x))
        x = spsolve_triangular(U,w,lower=False)
        x = x/np.linalg.norm(x)

        if np.linalg.norm(A.dot(x)) < tol:
            break
        
        # y1 = spsolve_triangular(L.T,x/np.linalg.norm(x),lower=False)
        # w1 = spsolve_triangular(U.T,y1)
        # y2 = spsolve_triangular(L,w1)
        # x = spsolve_triangular(U,y2,lower=False)
        
        
        # w = spsolve_triangular(A,x/np.linalg.norm(x))
        # w = spsolve(A.T,x/np.linalg.norm(x))
        # x = spsolve(A,w)

    return x

def orthogonalize(U, eps=1e-15):
    """
    Orthogonalizes the matrix U (d x n) using Gram-Schmidt Orthogonalization.
    If the columns of U are linearly dependent with rank(U) = r, the last n-r columns 
    will be 0.
    
    Args:
        U (numpy.array): A d x n matrix with columns that need to be orthogonalized.
        eps (float): Threshold value below which numbers are regarded as 0 (default=1e-15).
    
    Returns:
        (numpy.array): A d x n orthogonal matrix. If the input matrix U's cols were
            not linearly independent, then the last n-r cols are zeros.
    
    Examples:
    ```python
    >>> import numpy as np
    >>> import gram_schmidt as gs
    >>> gs.orthogonalize(np.array([[10., 3.], [7., 8.]]))
    array([[ 0.81923192, -0.57346234],
       [ 0.57346234,  0.81923192]])
    >>> gs.orthogonalize(np.array([[10., 3., 4., 8.], [7., 8., 6., 1.]]))
    array([[ 0.81923192 -0.57346234  0.          0.        ]
       [ 0.57346234  0.81923192  0.          0.        ]])
    ```
    """
    
    n = len(U[0])
    # numpy can readily reference rows using indices, but referencing full rows is a little
    # dirty. So, work with transpose(U)
    V = U.T
    for i in range(n):
        prev_basis = V[0:i]     # orthonormal basis before V[i]
        coeff_vec = np.dot(prev_basis, V[i].T)  # each entry is np.dot(V[j], V[i]) for all j < i
        # subtract projections of V[i] onto already determined basis V[0:i]
        V[i] -= np.dot(coeff_vec, prev_basis).T
        if np.linalg.norm(V[i]) < eps:
            V[i][V[i] < eps] = 0.   # set the small entries to 0
        else:
            V[i] /= np.linalg.norm(V[i])
    return V.T

def main():
    MESH_DIM = 2
    FXN_SPACE_DIM = 1

    if MESH_DIM==1:
        msh = mesh.create_interval(MPI.COMM_WORLD,5,[0,1])
    if MESH_DIM==2:
        N = 1000
        W = .1
        H = .5
        msh = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
        import pyvista
        from dolfinx import plot
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        if False:
            tdim = msh.topology.dim
            topology, cell_types, geometry = plot.create_vtk_mesh(msh, tdim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
            plotter = pyvista.Plotter()
            plotter.add_mesh(grid, show_edges=True,opacity=0.25)
            plotter.view_isometric()
            if not pyvista.OFF_SCREEN:
                plotter.show()

    if MESH_DIM==3:
        msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                        np.array([2.0, 1.0, 1.0])], [3, 3, 3],
                                            mesh.CellType.tetrahedron)
    if FXN_SPACE_DIM==1:
        V = fem.FunctionSpace(msh, ("Lagrange", 1))
    if FXN_SPACE_DIM==2:
        V = fem.VectorFunctionSpace(msh, ("Lagrange", 1),dim=2)
    if FXN_SPACE_DIM==3:
        V = fem.VectorFunctionSpace(msh, ("Lagrange", 1),dim=3)
    if FXN_SPACE_DIM==12:
        Ve = ufl.VectorElement("CG",msh.ufl_cell(),1,dim=3)
        V = fem.FunctionSpace(msh, ufl.MixedElement(4*[Ve]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    ufl_form = ufl.inner(ufl.grad(u), ufl.grad(v)) *ufl.dx

    A = assemble_stiffness_matrix(ufl_form)
    print('stiffness matrix size:')
    print(A.shape)
    get_nullspace(msh,ufl_form,V)

    # # get_nullspace(msh,ufl_form,V)
    # sym_LU_inv_iter(A)
    return


main()