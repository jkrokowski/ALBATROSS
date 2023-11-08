import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner
from dolfinx import fem, io, mesh, plot
import cffi

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
        return A_local

# msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
#                                   np.array([2.0, 1.0, 1.0])], [3, 3, 3],
#                  mesh.CellType.tetrahedron)
N = 2
W = .1
H = .5
msh = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
# V = fem.FunctionSpace(msh, ("Lagrange", 1))
# V = fem.VectorFunctionSpace(msh, ("Lagrange", 1),dim=2)
Ve = ufl.VectorElement("CG",msh.ufl_cell(),1,dim=3)
V = fem.FunctionSpace(msh, ufl.MixedElement(4*[Ve]))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
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

dof_to_G = get_node_to_G_map(msh,V)
assembler = LocalAssembler(inner(grad(u), grad(v)) *dx)
# assembler = LocalAssembler(inner(grad(u[0]), grad(v[0])) *dx +inner(20*grad(u[1]), grad(v[1])) *dx)

for cell in range(msh.topology.index_map(msh.topology.dim).size_local):
    # if cell == 2:
    #     A = assembler.assemble_matrix(cell)
    #     print(A)
    A = assembler.assemble_matrix(cell)
    print('Local Stiffness matrix #%i' % cell)
    print(A)

#versus the process for global assembly:
A_full = fem.petsc.assemble_matrix(fem.form(inner(grad(u), grad(v)) *dx))
A_full.assemble()
# A_full = A.getValues()
print()

def get_element_matrices(domain,form):
    ''' returns a list of scipy sparse matrices
    However, should this be a dictionary with the key being the cell number?
    This would allow for subdomain analysis'''
    assembler = LocalAssembler(form)
    A_list = {}
    for i in range(domain.topology.index_map(domain.topology.dim).size_local):
        A_list[i]= assembler.assemble_matrix(i)
    
    return A_list

A_list = get_element_matrices(msh,fem.form(inner(grad(u), grad(v)) *dx))

print()