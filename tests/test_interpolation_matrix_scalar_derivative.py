import numpy as np
from dolfinx import mesh, fem, geometry, plot, cpp
from mpi4py import MPI
import basix
from petsc4py import PETSc
import pyvista
from ALBATROSS.nonmatching_utils import celltags_to_dofs,get_bbtrees,get_interpolation_matrix,get_collision_celltags,derivative_of_interpolation_matrix_nonmatching_meshes

def plot_meshes(source_mesh, target_mesh):
    """Plot the source and target meshes to visualize their overlap using PyVista."""
    # Convert Dolfinx meshes to PyVista PolyData
    def dolfinx_to_pyvista(dolfin_mesh):
        topology, cell_types, geometry = plot.vtk_mesh(dolfin_mesh, 2)
        return pyvista.UnstructuredGrid(topology, cell_types, geometry)

    pv_source = dolfinx_to_pyvista(source_mesh)
    pv_target = dolfinx_to_pyvista(target_mesh)

    # Create PyVista plotter
    plotter = pyvista.Plotter()
    plotter.add_mesh(pv_source, show_edges=True, color="blue", opacity=0.5, label="Source Mesh")
    plotter.add_mesh(pv_target, show_edges=True, color="red", opacity=0.5, label="Target Mesh")

    # Show the plot
    plotter.add_legend()
    plotter.show()

def get_overlapping_cells(target_mesh,source_mesh):
    '''return the cells of the target mesh that '''
    #get bounding boxes used for collision detection
    target_bbtree,source_bbtree =get_bbtrees([target_mesh,source_mesh])

    collisions_bbtree = geometry.compute_collisions_trees(source_bbtree,target_bbtree)

    # collision_points = geometry.compute_collisions_points(target_bbtree,source_mesh.geometry.x)
    source_cells,target_cells = get_collision_celltags(source_mesh,target_mesh,collisions_bbtree)

    return source_cells,target_cells

def test_interpolation_matrix():
    # Create two overlapping meshes
    N_source = 2
    N_target = 2
    source_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], [N_source, N_source], mesh.CellType.quadrilateral)
    target_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[0.25, 0.25], [1.5, 1.5]], [N_target ,N_target], mesh.CellType.quadrilateral)
    
    plot_meshes(source_mesh,target_mesh)

    # Define function spaces
    source_space = fem.functionspace(source_mesh, ("CG", 1))
    target_space = fem.functionspace(target_mesh, ("CG", 1))
    
    # Create an example function on the source space
    u = fem.Function(source_space)
    u.interpolate(lambda x: x[0]**2 + x[1]**2)  # Example function: u(x, y) = x^2 + y^2

    # Construct interpolation matrix (replace with your implementation)
    interpolation_matrix = get_interpolation_matrix(target_space, source_space)
    interpolation_matrix_derivative = derivative_of_interpolation_matrix_nonmatching_meshes(target_space,source_space)
    
    # Verify the matrix type
    assert isinstance(interpolation_matrix, PETSc.Mat), "Interpolation matrix must be a PETSc matrix."

    # Apply the interpolation matrix
    source_vector = u.x.petsc_vec
    target_vector = PETSc.Vec().createMPI(interpolation_matrix.getSize()[0])
    interpolation_matrix.mult(source_vector, target_vector)

    # Check interpolation results
    v = fem.Function(target_space)
    v.x.petsc_vec.setArray(target_vector.array)

    # Define the exact solution on the target mesh for comparison
    v_exact = fem.Function(target_space)
    v_exact.interpolate(lambda x: x[0]**2 + x[1]**2)
    
    #use the FEniCS nonmatching mesh interpolation feature to validate against
    #the error should be identical to this method and the exact solution allows us
    # to quantify the total error from interpolation
    v_interp = fem.Function(target_space)
    cell_map_v = target_mesh.topology.index_map(target_mesh.topology.dim)
    num_cells_on_proc = cell_map_v.size_local + cell_map_v.num_ghosts
    cells_v = np.arange(num_cells_on_proc,dtype=np.int32)
    v_interp.interpolate_nonmatching(u, cells_v,interpolation_data=fem.create_interpolation_data(target_space, source_space, cells_v))
    
    #get the celltags of the overlapping region
    source_cells,target_cells = get_overlapping_cells(target_mesh,source_mesh)

    #get the dofs associated with the cells of the overlapping region
    target_dofs = celltags_to_dofs(target_space,target_cells)

    #TODO: need to handle the case for cells with "hanging dofs"
    target_nz = np.where(v.x.array!=0)

    # Compute the error between the fenics interpolation method 
    #   and the method with the explicitly constructed interpolation matrix
    error = np.linalg.norm(v.x.array - v_interp.x.array)
    assert error < 1e-12, f"Interpolation error is too large: {error}"

    interpolation_error =  np.linalg.norm(v_exact.x.array[target_dofs]-v.x.array[target_dofs])
    print(f"Interpolation error (w/ hanging dofs) : {interpolation_error}")

    interpolation_error =  np.linalg.norm(v_exact.x.array[target_nz]-v.x.array[target_nz])
    print(f"Interpolation error (w/o hanging dofs): {interpolation_error}")

    plot_meshes(source_mesh,target_mesh)

    #
    # dP = grad_phi \dot Ja_inv 
    
    ct      = cpp.mesh.to_string(source_mesh.topology.cell_type)
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.CellType[ct], source_mesh.ufl_element().degree, basix.LagrangeVariant.equispaced)

    #return the basis function values at the reference points for all points and basis function indices at the scalar component
    #TODO: this is likely where I would modify my function to handle non-scalar spaces (e.g. last index)
    # basis_matrix = element.tabulate(0, x_ref)[0,:,:,0]
    # grad_phi = 

if __name__ == "__main__":
    test_interpolation_matrix()
