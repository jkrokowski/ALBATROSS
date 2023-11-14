import dolfinx.cpp.mesh
import numpy as np
import pyvista
from dolfinx import plot
import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI
import scipy.io
import meshio

from dolfinx.geometry import BoundingBoxTree,compute_collisions,compute_colliding_cells
from ufl import TestFunction,TrialFunction,inner,dx
from dolfinx.fem.petsc import assemble_matrix,assemble_vector,apply_lifting,set_bc,create_vector
from dolfinx.fem import form
from petsc4py import PETSc

def get_vtx_to_dofs(domain,V):
     '''
     solution from https://fenicsproject.discourse.group/t/application-of-point-forces-mapping-vertex-indices-to-corresponding-dofs/9646
     --------------
     input: subspace to find DOFs in
     output: map of DOFs related to their corresponding vertices
     '''
     V0, V0_to_V = V.collapse()
     dof_layout = V0.dofmap.dof_layout

     num_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
     vertex_to_par_dof_map = np.zeros(num_vertices, dtype=np.int32)
     num_cells = domain.topology.index_map(
          domain.topology.dim).size_local + domain.topology.index_map(
          domain.topology.dim).num_ghosts
     c_to_v = domain.topology.connectivity(domain.topology.dim, 0)
     for cell in range(num_cells):
          vertices = c_to_v.links(cell)
          dofs = V0.dofmap.cell_dofs(cell)
          for i, vertex in enumerate(vertices):
               vertex_to_par_dof_map[vertex] = dofs[dof_layout.entity_dofs(0, i)]

     geometry_indices = dolfinx.cpp.mesh.entities_to_geometry(
          domain, 0, np.arange(num_vertices, dtype=np.int32), False)
     bs = V0.dofmap.bs
     vtx_to_dof = np.zeros((num_vertices,bs), dtype=np.int32)
     for vertex, geom_index in enumerate(geometry_indices):
          par_dof = vertex_to_par_dof_map[vertex]
          for b in range(bs):
               vtx_to_dof[vertex, b] = V0_to_V[par_dof*bs+b]
     # vtx_to_dof = np.reshape(vtx_to_dof, (-1,1))

     return vtx_to_dof

def plot_xdmf_mesh(msh,add_nodes=False):
     #plot mesh
     pyvista.global_theme.background = [255, 255, 255, 255]
     pyvista.global_theme.font.color = 'black'
     tdim = msh.topology.dim
     topology, cell_types, geom = plot.create_vtk_mesh(msh, tdim)
     grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
     plotter = pyvista.Plotter()
     plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     if add_nodes==True:
          plotter.add_mesh(grid, style='points')
     plotter.view_isometric()
     if not pyvista.OFF_SCREEN:
          plotter.show()

def beam_interval_mesh_3D(pts,ne,meshname):
    '''
    pts = list of nx (x,y,z) locations of a beam nodes (np)
    ne = list of number of elements for each segment between nodes (np-1)
    meshname = name of mesh
    '''
    filename = 'output/'+meshname+'.xdmf'

    gdim = 3
    tdim = 1

    gmsh.initialize()

    #construct line in 3D space
    gmsh.model.add(meshname)
    gmsh.model.setCurrent(meshname)
    
    pt_tags = []
    for pt in pts:
        pt_tag = gmsh.model.geo.addPoint(pt[0],pt[1],pt[2])
        pt_tags.append(pt_tag)
    line_tags = []
    for i,n in enumerate(ne):
        line_tag = gmsh.model.geo.addLine(pt_tags[i],pt_tags[i+1])
        line_tags.append(line_tag)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag, int(n + 1))

    # Synchronize model representation with gmsh model
    gmsh.model.geo.synchronize()

    # add physical marker
    gmsh.model.add_physical_group(tdim,line_tags)

    #generate the mesh and optionally write the gmsh mesh file
    gmsh.model.mesh.generate(gdim)
    # gmsh.write(filename)

    #use meshio to convert msh file to xdmf
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
    msh.name = meshname
    # cell_markers.name = f"{msh.name}_cells"
    # facet_markers.name = f"{msh.name}_facets"

    # close gmsh API
    gmsh.finalize()

    #write xdmf mesh file
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)

    #return mesh
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        return xdmf.read_mesh(name=meshname)
    

def get_pts_and_cells(domain,points):
     '''
     ARGS:
          point = tuple of (x,y,z) locations to return displacements and rotations
     '''
     bb_tree = BoundingBoxTree(domain,domain.topology.dim)
     points = np.array(points)

     cells = []
     points_on_proc = []
     # Find cells whose bounding-box collide with the the points
     cell_candidates = compute_collisions(bb_tree, points)
     # Choose one of the cells that contains the point
     colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
     for i, point in enumerate(points):
          if len(colliding_cells.links(i))>0:
               points_on_proc.append(point)
               cells.append(colliding_cells.links(i)[0])

     points_on_proc = np.array(points_on_proc,dtype=np.float64)
     # points_on_proc = np.array(points_on_proc,dtype=np.float64)

     return points_on_proc,cells
     # disp = self.uh.sub(0).eval(points_on_proc,cells)
     # rot = self.uh.sub(1).eval(points_on_proc,cells)

def mat_to_mesh(filename,aux_data=None, plot_xc = False ):
     mat = scipy.io.loadmat(filename)
     data = []
     for item in aux_data:
          data.append(mat[item])

     elems = mat['vabs_2d_mesh_elements']
     nodes = mat['vabs_2d_mesh_nodes']
     print('Number of nodes:')
     print(len(nodes))
     print('Number of Elements:')
     print(len(elems))
     elems -=1

     cells = {'triangle':elems[:,0:3]}
     meshio.write_points_cells('file.xdmf',nodes,cells,file_format='xdmf')

     with XDMFFile(MPI.COMM_WORLD, 'file.xdmf', "r") as xdmf:
          msh = xdmf.read_mesh(name='Grid')

     if plot_xc:

          msh.topology.create_connectivity(msh.topology.dim-1, 0)

          plotter = pyvista.Plotter()
          num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
          topology, cell_types, x = plot.create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

          grid = pyvista.UnstructuredGrid(topology, cell_types, x)
          plotter.add_mesh(grid,show_edges=True)
          plotter.show_axes()
          plotter.add_points(np.array((0,0,0)))
          plotter.show()
          
     if aux_data is not None:
          return msh,data
     else:
          return msh
     

"""
    Project function does not work the same between legacy FEniCS and FEniCSx,
    so the following project function must be defined based on this forum post:
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-function-dolfinx/4142/6
    and inspired by Ru Xiang's Shell module project function
    https://github.com/RuruX/shell_analysis_fenicsx/blob/b842670f4e7fbdd6528090fc6061e300a74bf892/shell_analysis_fenicsx/utils.py#L22
    """

def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = inner(Pv, w) * dx
    L = inner(v, w) * dx

    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)