import dolfinx.cpp.mesh
import numpy as np
import pyvista
from dolfinx import plot
from dolfinx.io import XDMFFile
from mpi4py import MPI
import scipy.io
from scipy.sparse import csc_matrix,csr_matrix
import meshio
from scipy.spatial import cKDTree
from dolfinx.geometry import bb_tree,compute_collisions_points,compute_colliding_cells
from ufl import TestFunction,TrialFunction,inner,dx,as_vector
from dolfinx.fem.petsc import assemble_matrix,assemble_vector,apply_lifting,set_bc
from dolfinx import fem
from petsc4py import PETSc


def proj_expr(expr,V):
     return

def _orthonormalize_rbm(self,fxn,verbose=False):
     V = fxn.function_space
     x = self.x
     dx  = self.dx

     #Rigid Body Modes expression (3D)
     rbms = [
          fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((1.0,0.0,0.0))),V.element.interpolation_points()),
          fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((0.0,1.0,0.0))),V.element.interpolation_points()),
          fem.Expression(fem.Constant(self.msh,PETSc.ScalarType((0.0,0.0,1.0))),V.element.interpolation_points()),
          fem.Expression(as_vector([0,-x[1],x[0]]),V.element.interpolation_points())#,
          # fem.Expression(ufl.as_vector([x[1],0,0]),V.element.interpolation_points()),
          # fem.Expression(ufl.as_vector([-x[0],0,0]),V.element.interpolation_points())
     ]

     # List of functions to orthogonalise
     vx = fem.Function(V)
     vy = fem.Function(V)
     vz = fem.Function(V)
     vrx = fem.Function(V)
     # vry = fem.Function(V)
     # vrz = fem.Function(V)
     vx.interpolate(rbms[0])
     vy.interpolate(rbms[1])
     vz.interpolate(rbms[2])
     vrx.interpolate(rbms[3])
     # vry.interpolate(rbms[4])
     # vrz.interpolate(rbms[5])

     # v = list((vx,vy,vz,vrx,vry,vrz))
     v = list((vx,vy,vz,vrx))

     # GS Projection
     def proj(u, v):
          res = fem.assemble_scalar(fem.form(inner(u, v)*dx))/fem.assemble_scalar(fem.form(inner(u, u)*dx)) * u.vector.array
          return res

     # GS orthogonalisation
     def ortho(v):
          xi = [None]*len(v)
          xi[0] = v[0]
          for j in range(1, len(xi)):
               xi[j] = fem.Function(V)
               xi[j].vector.array = v[j].vector.array - sum(proj(xi[i], v[j]) for i in range(j))
          return xi
     
     xi = ortho(v)

     # Orthonormalised vector basis
     e = [fem.Function(V) for i in range(len(v))]
     for i,xi_ in enumerate(xi):
          e[i].vector.array = xi_.vector.array/fem.assemble_scalar(fem.form(inner(xi_, xi_)*dx))**0.5
     
     new_fxn = fem.Function(V)
     new_fxn.vector.array  = fxn.vector.array - sum(proj(e_, fxn) for e_ in e)

     if verbose is True:
          print("orthonormalisation test:")
          for i in range(len(xi)):
               for j in range(i+1):
                    print(f"inner(e[{i}], e[{j}])*dx {fem.assemble_scalar(fem.form(inner(e[i], e[j])*dx))}")

          print(f"u norm {fxn.vector.norm(2)}, u_star norm {new_fxn.vector.norm(2)}")
          print(f"orthogonalisation of u_star with rigid body modes test:")
          for j in range(len(v)):
               print(f"(rbms[{j}], u_star) = {fem.assemble_scalar(fem.form(inner(new_fxn, v[j])*dx))}")

     return new_fxn
        
def get_vtx_to_dofs(domain,V):
     '''
     solution from https://fenicsproject.discourse.group/t/application-of-point-forces-mapping-vertex-indices-to-corresponding-dofs/9646
     --------------
     input: (uncollapsed) subspace to find DOFs in
     output: map of DOFs related to their corresponding vertices 
               (shape: num vertices x num dofs per vertex)
     '''
     V0, V0_to_V = V.collapse()
     
     num_vertices_per_cell = dolfinx.cpp.mesh.cell_num_entities(
          domain.topology.cell_type, 0
     )

     dof_layout2 = np.empty((num_vertices_per_cell,), dtype=np.int32)
     for i in range(num_vertices_per_cell):
          var = V.dofmap.dof_layout.entity_dofs(0, i)
          assert len(var) == 1
          dof_layout2[i] = var[0]

     num_vertices = (
          domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
     )

     c_to_v = domain.topology.connectivity(domain.topology.dim, 0)
     assert (
          c_to_v.offsets[1:] - c_to_v.offsets[:-1] == c_to_v.offsets[1]
     ).all(), "Single cell type supported"
     
     #construct 
     vertex_to_dof_map = np.empty(num_vertices, dtype=np.int32)
     vertex_to_dof_map[c_to_v.array] = V0.dofmap.list[:, dof_layout2].reshape(-1)
     
     geometry_indices = dolfinx.cpp.mesh.entities_to_geometry(
          domain._cpp_object, 0, np.arange(num_vertices, dtype=np.int32), False)
     bs = V0.dofmap.bs
     vtx_to_dof = np.zeros((num_vertices,bs), dtype=np.int32)
     for vertex, geom_index in enumerate(geometry_indices):
          par_dof = vertex_to_dof_map[vertex]
          for b in range(bs):
               vtx_to_dof[vertex, b] = V0_to_V[par_dof*bs+b]

     return vtx_to_dof

def plot_xdmf_mesh(msh,surface=True,add_nodes=False):
     pyvista.global_theme.background = [255, 255, 255, 255]
     pyvista.global_theme.font.color = 'black'
     plotter = pyvista.Plotter()
     #plot mesh
     if type(msh) is not list:
          tdim = msh.topology.dim
          topology, cell_types, geom = plot.vtk_mesh(msh, tdim)
          grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
          if surface:
               plotter.add_mesh(grid,show_edges=True,opacity=0.25)
          if not surface:
               plotter.add_mesh(grid,color='k',show_edges=True)
          if add_nodes:
               plotter.add_mesh(grid, style='points',color='k')
          # plotter.view_isometric()
          plotter.view_xy()
          plotter.show_bounds()
          plotter.add_axes()
          if not pyvista.OFF_SCREEN:
               plotter.show()
     else:
          for m in msh:
               tdim = m.topology.dim
               topology, cell_types, geom = plot.vtk_mesh(m, tdim)
               grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
               # plotter.add_mesh(grid,show_edges=True,opacity=0.25)
               if surface:
                    plotter.add_mesh(grid,show_edges=True,opacity=0.25)
               else:
                    plotter.add_mesh(grid,color='k',show_edges=True)
               if add_nodes:
                    plotter.add_mesh(grid, style='points',color='k')
          # plotter.view_isometric()
          plotter.view_xy()
          plotter.show_bounds()
          plotter.add_axes()
          if not pyvista.OFF_SCREEN:
               plotter.show() 
    
def get_pts_and_cells(domain,points):
     '''
     ARGS:
          point = tuple of (x,y,z) locations to return displacements and rotations
     '''
     bounding_box_tree = bb_tree(domain,domain.topology.dim)
     points = np.array(points)

     cells = []
     points_on_proc = []
     # Find cells whose bounding-box collide with the the points
     cell_candidates = compute_collisions_points(bounding_box_tree, points)
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

def mat_to_mesh(filename,aux_data=None, plot_xs = False ):
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

     if plot_xs:

          msh.topology.create_connectivity(msh.topology.dim-1, 0)

          plotter = pyvista.Plotter()
          num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
          topology, cell_types, x = plot.vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

          grid = pyvista.UnstructuredGrid(topology, cell_types, x)
          plotter.add_mesh(grid,show_edges=True)
          plotter.show_axes()

          if True:
               # Add labels to points on the yz plane (where x == 0)
               points = grid.points
               # mask = points[:, 0] == 0
               data=points - np.tile([[0,np.min(points[:,1]),0]],(points.shape[0],1))
               m_to_in = 39.37
               plotter.add_point_labels(points, (m_to_in*data).tolist())

               # plotter.camera_position = [(-1.5, 1.5, 3.0), (0.05, 0.6, 1.2), (0.2, 0.9, -0.25)]
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
    A = assemble_matrix(fem.form(a), bcs)
    A.assemble()
    b = assemble_vector(fem.form(L))
    apply_lifting(b, [fem.form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

def sparseify(mat,sparse_format='csr',lim=None):
     if lim is None:
          lim = 2*np.finfo(float).eps # this is the numpy precision
     # lim = 1e-10 #choosing a slightly more conservative value 
     mat.real[abs(mat.real) < lim] = 0.0
     if sparse_format == 'csc':
          return csc_matrix(mat)
     elif sparse_format == 'csr':
          return csr_matrix(mat)
     

def gmsh_to_xdmf(mesh, cell_type, prune_z=False):
     cells = mesh.get_cells_type(cell_type)
     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
     points = mesh.points[:,:2] if prune_z else mesh.points
     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
     return out_mesh

def xy2D_to_yz3D(mesh):
     #TODO: implement this guy
     return



def order_boundary_nodes(coords):
    N = len(coords)
    ordered = [0]  # start with first node
    used = set(ordered)

    tree = cKDTree(coords)
    for _ in range(1, N):
        last = coords[ordered[-1]]
        dists, idxs = tree.query(last, k=N)
        next_idx = next(i for i in idxs if i not in used)
        ordered.append(next_idx)
        used.add(next_idx)

    return np.array(ordered)

def detect_corners(P, angle_deg_min=30.0, sagitta_min=0.05, k_list=(1,2)):
    P = np.asarray(P, float)
    N = len(P)
    is_corner = np.zeros(N, dtype=bool)
    score = np.zeros(N)

    def roll(a, s):  # closed loop neighbor indexing
        return np.roll(a, s, axis=0)

    for k in k_list:
        p_prev = roll(P, +k)
        p_next = roll(P, -k)

        v1 = P - p_prev
        v2 = p_next - P

        # Turning angle
        cross = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
        dot   = (v1*v2).sum(axis=1)
        theta = np.arctan2(cross, dot)         # radians
        ang   = np.abs(np.degrees(theta))      # degrees

        # Sagitta normalized by chord
        chord = p_next - p_prev
        L = np.linalg.norm(chord, axis=1)
        # distance point->chord (area*2/chord)
        # area = 0.5*| (p_i - p_prev) x chord |
        area2 = np.abs((P[:,0]-p_prev[:,0])*chord[:,1] - (P[:,1]-p_prev[:,1])*chord[:,0])
        d = area2 / L.clip(min=1e-15)
        s = 2*d / L.clip(min=1e-15)

        # Edge-length sanity
        eok = (np.linalg.norm(v1,axis=1) > 1e-12) & (np.linalg.norm(v2,axis=1) > 1e-12)

        # Corner score: combine angle + sagitta
        cur_score = (ang/angle_deg_min) * (s/sagitta_min)
        mask = (ang >= angle_deg_min) & (s >= sagitta_min) & eok

        # Keep best across scales
        improve = cur_score > score
        is_corner = (is_corner & (~improve)) | (mask & improve)
        score = np.where(improve, cur_score, score)

    # non-max suppression in a small neighborhood
    for i in range(N):
        if not is_corner[i]: continue
        nb = [(i-1)%N,(i+1)%N]
        for j in nb:
            if is_corner[j] and score[j] < score[i]:
                is_corner[j] = False
    return np.where(is_corner)[0], score