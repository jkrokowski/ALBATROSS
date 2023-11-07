import cffi
from petsc4py.PETSc import ScalarType
from dolfinx import fem
import numpy as np
from scipy.sparse import lil_matrix,csgraph,csr_matrix


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
        return lil_matrix(A_local)

def get_nullspace(domain,form):
    #build list of element matrices using custom local assembler
    A_list = get_element_matrices(domain,form)

    #extract element connectivity from dolfin mesh object
    G = get_rigidity_graph(domain)

    # #get a minimum spanning tree from the graph representation
    # T = compute_spanning_tree(G)
    V = get_node_to_el_map(domain)

    #compute the fretsaw extension (returns a sparse matrix)
    F = compute_fretsaw_extension(A_list,G,V)

    #compute the LU factorization of the fretsaw extension

    #utilize LU factors to perform subspace inverse iteration

    #restrict to first n factors and orthogonalize

def compute_fretsaw_extension(A,G,V):
    '''
    method developed from psuedocode of Shklarski and toledo (2008) Rigidity in FE matrices, Algoithm 1
    
    A: collection of local stiffness matrices of (n x n) shape in scipy sparse matrix format ordered 
        by the cell number with total number of cells k
        ^this is actually not true, A can simply be the assembled global stiffness matrix and we can
        compute the fretsaw extension by F(A) = Q@A@Q^T
    G: rigidity graph for the mesh connecting elements
    '''
    
    T = compute_spanning_tree(G)
    (n,_) = A[0].get_shape()
    k = len(A)
    r = n

    #initialize empty Q(i) matrices (k total matrices)
    #these will be trimmed later
    Q_list = [lil_matrix((n*k,n)) for i in range(k)]


    for j in range(n):
        #initialize to the identity matrix
        # for Q in Q_list:
        #     Q[0,j] = 1  
        Q_list[0][j,j]=1
        #find all elements connected to node j and build "connectivity components"
        #connectivity components are all the portions of the spanning tree T that 
        # have elements connected to node j 
        
        #indices of the elements that are connected to the j-th node:
        idx = V[j]
        
        #get the subgraph induced by the elements incident on the j-th node
        Gj=T[idx,:].tocsc()[:,idx]

        #get the number of connected components and their labels:
        n_comp,labels = csgraph.connected_components(Gj)

        #get the connectivity components of the subgraph Gj
        Gcj = []

        #====
        #populate connectivity components
        #=====

        # for each connectivity component, modify the entries of the 
        # appropriate extension matrices:
        for p in range(n_comp):
            if p==0:
                #set j-th col of Qp to ej
                print()
            else:
                r += 1
                # for all element matrices in V
                    # set the j-th column of Qp to e_r (e.g. a 1 in the r-th )

        #for each extension matrix:
        for Q in Q_list:
            if Q[:,j]
        

                


    return

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
    A_list = {}
    for i in range(domain.topology.index_map(domain.topology.dim).size_local):
        A_list[i]= assembler.assemble_matrix(i)
    
    return A_list

def get_node_to_el_map(domain):
    #create connectivity (edge=1,cell=2)
    domain.topology.create_connectivity(0,2)
    #create dolfinx adjacency list
    nod_to_el = domain.topology.connectivity(0,2)
    V = {}
    for i in range(nod_to_el.num_nodes):
        V[i]=nod_to_el.links(i)
    return V 