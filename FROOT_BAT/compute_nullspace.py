import cffi
from petsc4py.PETSc import ScalarType
from dolfinx import fem
import numpy as np
from scipy.sparse import lil_matrix


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

    #compute the fretsaw extension (returns a sparse matrix)
    F = compute_fretsaw_extension(A_list,G)

    #compute the LU factorization of the fretsaw extension

    #utilize LU factors to perform subspace inverse iteration

    #restrict to first n factors and orthogonalize

def compute_fretsaw_extension(A,G):
    '''
    method developed from psuedocode of Shklarski and toledo (2008) Rigidity in FE matrices, Algoithm 1
    A: collection of local stiffness matrices
    G: rigidity graph for the mesh
    '''
    

    #compute fretsaw-forest factorization

    #compute LU factorization with partial pivoting

    #compute nullspace using subspace inverse iteration

    #restrict nullspace to first n coordinates and orthogonalize

    return

def compute_nullspace(F):
    '''
    F: Fretsaw extension of a finite element problem
    
    returns:
    basis for nullspace of the matrix A and F(A)
    '''


def get_rigidity_graph(domain):
    ''''
    returns the ridigity graph structure for a 2d mesh
    needs the mesh from which the edge to cell connectivity can be created

    returns the rigidity graph in the form of an adjacency matrix
    '''
    #create connectivity (edge=1,cell=2)
    domain.topology.create_connectivity(1,2)
    #create dolfinx adjacency list
    e_to_f = domain.topology.connectivity(1,2)

    #initialize a sparse adjacency matrix
    num_el = len(domain.geometry.dofmap)
    G = lil_matrix((num_el,num_el),dtype=np.int8)
    # G = np.zeros((e_to_f.num_nodes,e_to_f.num_nodes))
    for i in range(e_to_f.num_nodes):
        link = e_to_f.links(i)
        if len(e_to_f.links(i)) == 2: 
            #add edge to graph (symmetric b/c undirected)
            G[link[0],link[1]] = 1
        else:
            continue    

    #since G is undirected, it will be symmetric and we could populate the 
    # lower triangular entries with G += G.T, but it is more memory efficient
    # to only store the entries above the diagonal

    return G

def compute_spanning_tree(G):
    '''
    return a spanning tree T from a rigidity graph
    '''

    return

def get_element_matrices(domain,form):
    assembler = LocalAssembler(form)
    A_list = []
    for i in range(domain.topology.index_map(domain.topology.dim).size_local):
        A_list.append(assembler.assemble_matrix(i))
    
    return A_list
    