from dolfinx import mesh,plot
from mpi4py import MPI
import numpy as np
from scipy.sparse import lil_matrix
W = 1
H = 1
N = 4

domain = mesh.create_rectangle( MPI.COMM_SELF,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
if False:
     import pyvista
     tdim = domain.topology.dim
     topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
     plotter = pyvista.Plotter()
     plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     plotter.view_isometric()
     if not pyvista.OFF_SCREEN:
          plotter.show()
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

G = get_rigidity_graph(domain)

def get_element_matrices(domain,form):
    assembler = LocalAssembler(form)
    A_list = []
    for i in range(domain.topology.index_map(domain.topology.dim).size_local):
        A_list.append(assembler.assemble_matrix(i))
    
    return A_list