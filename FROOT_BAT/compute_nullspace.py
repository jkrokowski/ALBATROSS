import cffi
from petsc4py.PETSc import ScalarType
from dolfinx import fem
import numpy as np
from scipy.sparse import lil_matrix,csgraph,csr_matrix
from FROOT_BAT.utils import get_vtx_to_dofs

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

def get_nullspace(domain,ufl_form):
    #build list of element matrices using custom local assembler
    Ae = get_element_matrices(domain,ufl_form)

    # get local to global index map
    Ie = get_local_to_global()

    # # constructs global stiffness matrix
    # A = assemble_stiffness_matrix(ufl_form)

    #extract element connectivity from dolfin mesh object
    G = get_rigidity_graph(domain)

    #get which elements are connected to each node
    V = get_node_to_el_map(domain)
    
    # for formulations involving natural groupings of indices, get the
    # relationship between the mesh rigidity and the dof grouping
    #   -allows for a "consistent" extension in the fretsaw construction
    #   -this involves making 'cuts' between the same elements for 
    #    different vector/function directions e.g for 2D linear elasticity,
    #    severing the same connections between x- and y-directions 
    get_node_to_G_map(domain,fxn_space)

    #compute the fretsaw extension (returns a sparse matrix)
    F = compute_fretsaw_extension(Ae,Ie,G,V)

    #compute the LU factorization of the fretsaw extension

    #utilize LU factors to perform subspace inverse iteration

    #restrict to first n factors and orthogonalize

def compute_fretsaw_extension(Ae,Ie,G,V):
    '''
    method developed from psuedocode of Shklarski and toledo (2008) Rigidity in FE matrices, Algorithm 1
    
    Ae: list of length (num_elements) of assembled local stiffness matrices of shape (n x n)
    
    Ie: list of length (num_elements) of vectors of (num_element_indices) linking global indices to 
        local stiffness matrix element indices

        Question: is it possible that A can simply be the assembled global stiffness matrix and we can
        compute the fretsaw extension by F(A) = Q@A@Q^T . It may be easier to pass the full stiffness 
        matrix and along with a key to the ordering of the dofs, then build the extension matrix

    G: rigidity graph for the mesh detailing element to element connections
    
    V: node to element connectivity (provided for easy subgraph detection)
    '''
    
    #TODO: need to modify this algorithm to handle the 12dim vector construction

    #get number of indices for each local stiffness matrix
    (ne,_) = Ae[0].get_shape()
    #get number of elements
    n = G.shape
    #get number of local stiffness matrices
    k = len(Ae)
    #initialize last non-zero row of the extension matrices
    r = n

    T = compute_spanning_tree(G)
    
    #initialize extension matrices (k total matrices)
    #these will be trimmed later using the final resultant r value
    Qe = [lil_matrix((n*k,n)) for i in range(k)]

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

        if 0 in ele:
            #TODO: reorder Gcj to have Ae[0] in Gcj[0]
            print("need to make sure Ae[0] is here")
            
        #reorder to make sure "master element" is always in the first connectivity component
        #set the j-th column of Qp to ej for all elements in G0j
        for p in Gcj[0]:
            #zero out the j-th column and eliminate zeros
            nonzeros = Qe[p][:,j].nonzero()[0]
            Qe[p][nonzeros,j] = np.zeros_like(nonzeros)
            #assign ej to column j
            Qe[p][j,j] = 1

        # for each connectivity component after the first one, modify the
        # entries of the appropriate extension matrices:
        for i in range(1,n_comp):
            #increment last non-zero row tracker
            r += 1
            #for all the elements in Gcj[p], set the j-th column of Qp to er
            for p in Gcj[i]:
                # set the j-th column of Qp to e_r (e.g. a 1 in the r-th )
                Qe[p][r,j] = 1
        #for each extension matrix 
        for Q in Qe:
            #if ej is not in the nonzero colums of Ai, then set Qi to ej
            print()

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

def assemble_stiffness_matrix(ufl_form):
    #returns a scipy csr_matrix
    A_mat = fem.assemble_matrix(fem.form(ufl_form))
    A_mat.assemble()
    return csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_mat.size)

def get_node_to_G_map(domain,fxn_space):
    #gets the stiffness matrix indices for each node corresponding
    #to each function in a vector element/mixed space

    #this map assists with constructing a consistent extension
    subspace_dof_maps = []
    for space in range(fxn_space.num_sub_spaces):
        subspace_dof_maps.append(get_vtx_to_dofs(domain,fxn_space.sub(space)))
    #assemble into a matrix where columns correspond to each fxn component and
    # rows correspond to nodes
    dof_to_G = np.hstack(subspace_dof_maps)
        # to get out set of indices used to restrict the global indices of the 
        # stiffness matrix to the relevant indices for fxn select a column of 
        # dof_to_G  (e.g. G[i,:] will be a (num_nodes,) 1d np array)
    return dof_to_G