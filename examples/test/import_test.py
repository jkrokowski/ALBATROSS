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
# def get_rigidity_graph(domain):
#     ''''
#     returns the ridigity graph structure for a 2d mesh
#     needs the mesh from which the edge to cell connectivity can be created

#     returns the rigidity graph in the form of an adjacency matrix
#     '''
#     #create connectivity (edge=1,cell=2)
#     domain.topology.create_connectivity(1,2)
#     #create dolfinx adjacency list
#     e_to_f = domain.topology.connectivity(1,2)

#     #initialize a sparse adjacency matrix
#     num_el = len(domain.geometry.dofmap)
#     G = lil_matrix((num_el,num_el),dtype=np.int8)
#     # G = np.zeros((e_to_f.num_nodes,e_to_f.num_nodes))
#     for i in range(e_to_f.num_nodes):
#         link = e_to_f.links(i)
#         if len(e_to_f.links(i)) == 2: 
#             #add edge to graph (symmetric b/c undirected)
#             G[link[0],link[1]] = 1
#         else:
#             continue    

#     #since G is undirected, it will be symmetric and we could populate the 
#     # lower triangular entries with G += G.T, but it is more memory efficient
#     # to only store the entries above the diagonal

#     return G
def get_rigidity_graph(domain,direction='directed'):
    ''''
    returns the ridigity graph structure for a 2d mesh
    needs the mesh from which the edge to cell connectivity can be created

    returns the rigidity graph in the form of an adjacency list
    '''
    #create connectivity (edge=1,cell=2)
    domain.topology.create_connectivity(1,2)
    #create dolfinx adjacency list
    e_to_f = domain.topology.connectivity(1,2)

    #initialize a sparse adjacency matrix
    num_el = len(domain.geometry.dofmap)
    G = lil_matrix((num_el,num_el),dtype=np.int8)
    # G = {}
    # G = np.zeros((e_to_f.num_nodes,e_to_f.num_nodes))
    for i in range(e_to_f.num_nodes):
        link = e_to_f.links(i)
        if len(e_to_f.links(i)) == 2: 
            #add edge to graph (symmetric b/c undirected)
            G[link[0],link[1]] = 1
        #     if link[0] not in G:
        #         G[link[0]] = [link[1]]
        #     else:
        #         G[link[0]].append(link[1])
        #     if direction=='undirected':
        #         if link[1] not in G:
        #             G[link[1]] = [link[0]]
        #         else:
        #             G[link[1]].append(link[0])     
        else:
            continue    

    #since G is undirected, it will be symmetric and we could populate the 
    # lower triangular entries with G += G.T, but it is more memory efficient
    # to only store the entries above the diagonal
    # G += G.T
    return G

G = get_rigidity_graph(domain)
from scipy.sparse import csgraph

#returns csr matrix of adjacency matrix representing min. spanning tree
T = csgraph.minimum_spanning_tree(G)

# #create connectivity (edge=1,cell=2)
# domain.topology.create_connectivity(0,2)
# #create dolfinx adjacency list
# nod_to_el = domain.topology.connectivity(0,2)

def get_nod_to_el_map(domain):
    #create connectivity (edge=1,cell=2)
    domain.topology.create_connectivity(0,2)
    #create dolfinx adjacency list
    nod_to_el = domain.topology.connectivity(0,2)
    V = {}
    for i in range(nod_to_el.num_nodes):
        V[i]=nod_to_el.links(i)
    return V 

V = get_nod_to_el_map(domain)


print()
# def compute_spanning_tree(G):
#     '''
#     return a spanning tree T from a rigidity graph G using a simplified
#     version of Kruskal's Algorithm (since our graph is unweighted we
#     don't need to sort based on weights, saving  computational time )

#     G: rigidity graph in adjacency list form

#     T: a spanning tree
#     '''
#     V = list(G.keys())
#     # A utility function to find set of an element i 
#     # (truly uses path compression technique) 
#     def find(parent, i): 
#         if parent[i] != i: 
  
#             # Reassignment of node's parent 
#             # to root node as 
#             # path compression requires 
#             parent[i] = find(parent, parent[i]) 
#         return parent[i] 
  
#     # A function that does union of two sets of x and y 
#     # (uses union by rank) 
#     def union(parent, rank, x, y): 
  
#         # Attach smaller rank tree under root of 
#         # high rank tree (Union by Rank) 
#         if rank[x] < rank[y]: 
#             parent[x] = y 
#         elif rank[x] > rank[y]: 
#             parent[y] = x 
  
#         # If ranks are same, then make one as root 
#         # and increment its rank by one 
#         else: 
#             parent[y] = x 
#             rank[x] += 1

#         return parent,rank

#     # This will store the resultant MST 
#     result = [] 

#     # An index variable, used for sorted edges 
#     i = 0

#     # An index variable, used for result[] 
#     e = 0

#     # # Sort all the edges in 
#     # # non-decreasing order of their 
#     # # weight 
#     # self.graph = sorted(self.graph, 
#     #                     key=lambda item: item[2]) 

#     parent = [] 
#     rank = [] 

#     # Create V subsets with single elements 
#     for node in V: 
#         parent.append(node) 
#         rank.append(0) 
#     def adj_list_to_edges(G):
#         edge_list = []
#         for i in range(len(G)):
#             for j in G[i]:
#                 edge_list.append([i,j])
#         return edge_list
#     # Number of edges to be taken is less than to V-1 
#     while e < len(V): 

#         # Pick the smallest edge and increment 
#         # the index for next iteration 
#         edge_list= adj_list_to_edges(G)
#         u, v= edge_list[i]
#         i = i + 1
#         x = find(parent, u) 
#         y = find(parent, v) 

#         # If including this edge doesn't 
#         # cause cycle, then include it in result 
#         # and increment the index of result 
#         # for next edge 
#         if x != y: 
#             e = e + 1
#             result.append([u, v]) 
#             parent,rank=union(parent, rank, x, y) 
#         # Else discard the edge 

#     minimumCost = 0
#     print("Edges in the constructed MST") 
#     for u, v, weight in result: 
#         minimumCost += weight 
#         print("%d -- %d == %d" % (u, v, weight)) 
#     print("Minimum Spanning Tree", minimumCost) 
    
#     return

# T = compute_spanning_tree(G)

# print()