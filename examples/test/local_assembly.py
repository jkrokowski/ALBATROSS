import numpy as np
import ufl
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import ds, dx, grad, inner
from dolfinx import fem, io, mesh, plot
import cffi

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

msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0.0, 0.0, 0.0]),
                                  np.array([2.0, 1.0, 1.0])], [3, 3, 3],
                 mesh.CellType.tetrahedron)

V = fem.FunctionSpace(msh, ("Lagrange", 1))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

assembler = LocalAssembler(inner(grad(u), grad(v)) *dx)

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