import numpy as np
from basix.ufl import mixed_element,element
from dolfinx import mesh, fem, geometry, plot
from mpi4py import MPI
from petsc4py import PETSc
import pyvista
from ALBATROSS.nonmatching_utils import celltags_to_dofs,get_bbtrees,get_interpolation_matrix,get_collision_celltags

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
    source_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[0, 0], [1, 1]], [10, 10], mesh.CellType.triangle)
    target_mesh = mesh.create_rectangle(MPI.COMM_WORLD, [[0.5, 0.5], [1.5, 1.5]], [8,8], mesh.CellType.triangle)
    
    # Define function spaces
    ele = element('CG',source_mesh.topology.cell_name(),1,shape=(3,))
    source_space = fem.functionspace(source_mesh,mixed_element(4*[ele]))
    target_space = fem.functionspace(source_mesh,mixed_element(4*[ele]))
    
    # Create an example function on the source space
    u = fem.Function(source_space)
    u.interpolate(lambda x: (x[0]**2 + x[1]**2,
                                   x[0]**3 + x[1]**3))

    # Construct interpolation matrix (replace with your implementation)
    interpolation_matrix = get_interpolation_matrix(target_space, source_space)

    # Verify the matrix type
    assert isinstance(interpolation_matrix, PETSc.Mat), "Interpolation matrix must be a PETSc matrix."

    # Apply the interpolation matrix
    source_vector = u.vector
    target_vector = PETSc.Vec().createMPI(interpolation_matrix.getSize()[0])
    interpolation_matrix.mult(source_vector, target_vector)

    # Check interpolation results
    v = fem.Function(target_space)
    v.vector.setArray(target_vector.array)

    # Define the exact solution on the target mesh for comparison
    v_exact = fem.Function(target_space)
    v_exact.interpolate(lambda x: (x[0]**2 + x[1]**2,
                                   x[0]**3 + x[1]**3))
    
    #use the FEniCS nonmatching mesh interpolation feature to validate against
    #the error should be identical to this method and the exact solution allows us
    # to quantify the total error from interpolation
    v_interp = fem.Function(target_space)
    v_interp.interpolate(u, 
                         nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
                                                    v_interp.function_space.mesh,
                                                    v_interp.function_space.element,
                                                    u.function_space.mesh))
    #get the celltags of the overlapping region
    source_cells,target_cells = get_overlapping_cells(target_mesh,source_mesh)

    #get the dofs associated with the cells of the overlapping region
    target_dofs = celltags_to_dofs(target_space,target_cells)

    #TODO: need to handle the case for cells with "hanging dofs"
    target_nz = np.where(v.vector.array!=0)

    # Compute the error between the fenics interpolation method 
    #   and the method with the explicitly constructed interpolation matrix
    error = np.linalg.norm(v.vector.array - v_interp.vector.array)
    assert error < 1e-12, f"Interpolation error is too large: {error}"

    interpolation_error =  np.linalg.norm(v_exact.vector.array[target_dofs]-v.vector.array[target_dofs])
    print(f"Interpolation error (w/ hanging dofs) : {interpolation_error}")

    interpolation_error =  np.linalg.norm(v_exact.vector.array[target_nz]-v.vector.array[target_nz])
    print(f"Interpolation error (w/o hanging dofs): {interpolation_error}")

    plot_meshes(source_mesh,target_mesh)

if __name__ == "__main__":
    test_interpolation_matrix()
