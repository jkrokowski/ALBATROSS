import numpy as np
from dolfinx import cpp,mesh,fem,geometry
from dolfinx.fem.petsc import assemble_vector,assemble_matrix,create_vector
from petsc4py import PETSc
import basix
from mpi4py import MPI
import ufl
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,LineString, Point, MultiLineString
from shapely.ops import polygonize, unary_union
import gmsh
from dolfinx.io import gmshio


class Collision:
    '''
    Collection of information about each overlapping section
    '''
    def __init__(self,bb_tree_collisions,mesh_indices,celltags,pts):
        self.bb_tree_collisions = bb_tree_collisions
        self.mesh_indices = mesh_indices
        self.celltags = celltags
        self.pts = pts
        
    def add_pen_vec(self,pen_vec):
        self.pen_vec = pen_vec

    def add_overlap_areas(self,A_plus,A_minus,A_avg):
        self.A_plus,self.A_minus,self.A_avg  = A_plus,A_minus,A_avg

class Separation:
    '''
    An object to store information about the lack of an overlap
    (basically just an empty spare matrix to be added to the nested PETSc system)'''
    def __init__(self):
        self.mat = PETSc.Mat()

class Region:
    '''
    A distinct domain defined by a mesh
    '''
    def __init__(self,msh,fxn_space=None,element_metadata=None,bc_arg=None):
        
        #mesh for the region
        self.msh = msh

        #fxn space is either created now based on the problem, or already 
        # given and "linked".
        if element_metadata is not None:
            self.fxn_space = fem.functionspace(self.msh,element_metadata)
        else:
            self.fxn_space = fxn_space

        if bc_arg is not None:
            self.bc = bc_arg

        #dof to vertex map
        self.dof_to_vertex_map = np.tile(np.arange(self.msh.geometry.x.shape[0]),self.fxn_space.value_size)
        indices_to=[]
        for i in range(self.fxn_space.num_sub_spaces):
            _,map_to = self.fxn_space.sub(i).collapse()
            indices_to.extend(map_to)
        self.dof_to_vertex_map = self.dof_to_vertex_map[np.argsort(indices_to)]
 
    def add_plotting_info(self,plotting_dict): 
        self.plotting = plotting_dict

class CoupledProblem:
    '''solve a coupled problem with non matching, overlapping meshes'''
    def __init__(self,meshes,element_metadata,form_construction,bc_args,pen=1e5):
        assert(len(meshes)==len(bc_args))

        self.regions = {i:Region(msh,element_metadata=element_metadata,bc_arg=bc) for i,(msh,bc) in enumerate(zip(meshes,bc_args))} 
        self.meshes = {i:msh for i,msh in enumerate(meshes)}
        self.form_construction = form_construction
        self.bcs = {i:bc for i,bc in enumerate(bc_args)}
        self.pen = pen

        self.num_meshes = len(meshes)

        #compute collisions between all meshes
        self._find_overlap()

        #set up each individual mesh 
        self._assemble_systems()

    def _find_overlap(self):
        '''
        Construct collision objects 
        '''
        bb_trees = get_bbtrees(list(self.meshes.values()))
        self.bb_trees = {i:bbtree for i,bbtree in zip(self.meshes.keys(),bb_trees)}

        #compute all collisions
        # TODO: some collision detection computational time can be saved by avoiding 
        # the collision detection on the inverse mesh combination with a non-overlapping section         
        collisions = {}
        separations = {}
        adjacency = np.zeros((self.num_meshes,self.num_meshes),dtype=int)
        for i in self.meshes.keys():
            collisions_i = {}
            separations_i = {}
            for j in self.meshes.keys():
                if i==j:
                    continue
                #get collisions 
                collisions_bbtree_ij = geometry.compute_collisions_trees(self.bb_trees[i], self.bb_trees[j])
                
                if collisions_bbtree_ij.size != 0:
                    adjacency[i,j] = 1
                    collision_points_ij = geometry.compute_collisions_points(self.bb_trees[j],self.meshes[i].geometry.x)
                    celltags_i,celltags_j = get_collision_celltags(self.meshes[i],self.meshes[j],collisions_bbtree_ij)

                    penalty_dofs = pts_to_dofs(self.regions[i].fxn_space,collision_points_ij)
                    
                    #information about a collision of mesh i on mesh j
                    collision_ij = Collision(collisions_bbtree_ij,
                                             collision_points_ij,
                                             (celltags_i,celltags_j),
                                             penalty_dofs)

                    pen_vec = self._build_penalty_vector(self.regions[i],collision_ij)
                    
                    collision_ij.add_pen_vec(pen_vec)

                    collisions_i[j]=collision_ij

                elif collisions_bbtree_ij.size == 0:
                    separations_i[j]=Separation()
                    
            #add all collisions to dictionary list        
            collisions[i] = collisions_i
            separations[i] = separations_i

        self.collisions = collisions
        self.separations = separations
        self.adjacency = adjacency
    
    
    def _build_penalty_vector(self,region_i,collision_ij):
        '''
        Build the vector of penalty terms per dof
        This is a PETSc Vector that can be directly multiplied by 
        '''
        #initialize empty PETSc vector
        pen_vec = PETSc.Vec().create()
        vec_size = region_i.fxn_space.dofmap.index_map.size_global * region_i.fxn_space.num_sub_spaces
        pen_vec.setSizes(vec_size)
        pen_vec.setFromOptions()

        #compute areas of each element in the overlapping subdomain using a DG0 space
        DG0 = fem.functionspace(region_i.msh,("DG",0))
        v = ufl.TestFunction(DG0)
        # dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_id = 1, subdomain_data=collision_ij.celltags)
        dx_overlap = ufl.Measure("dx", domain=region_i.msh, subdomain_data=collision_ij.celltags[0])
        cell_area_form = fem.form(v*dx_overlap(1))
        cell_areas = fem.assemble_vector(cell_area_form)

        #create connectivity between cells and vertices (if not already created)
        region_i.msh.topology.create_connectivity(0,2)
        pen_values = np.zeros((len(collision_ij.penalty_dofs),),dtype=float)
        #return a list of penalty values for each dof
        for i,dof in enumerate(collision_ij.penalty_dofs):
            #get the corresponding vertex for a specific dof
            vtx = region_i.dof_to_vertex_map[dof]

            #get the cells connected to the penalty dof
            cells = region_i.msh.topology.connectivity(0,2).links(vtx)
            
            #add up area of all cells that are incident to the penalty dof
            #  adjust penalty proportionately to the supported area
            pen_values[i] = self.pen * np.sum(cell_areas.array[cells])

        #populate the PETSc vector with the values at the proper indices
        for idx,val in zip(collision_ij.penalty_dofs,pen_values):
            pen_vec.setValue(idx,val)
        
        return pen_vec

    def _assemble_systems(self):
        #compile systems for each individual mesh
        systems = []
        for region in self.regions.values():
            forms = self.form_construction(region.msh,region.fxn_space,region.bc)
            region.forms = forms
            system = get_petsc_system(forms[0],forms[1],forms[2])
            region.system = system
            systems.append(system)
        self.systems = systems

    def _construct_solution_fxns(self):
        for region in self.regions.values():
            region.fxn = fem.Function(region.fxn_space)

    def _construct_coupled_system(self):
        '''
        Given collisions and regions, 
        set up the coupled system with the penalty terms
        '''

        # for each collision, compute the interpolation matrices and add the penalty terms to the corresponding dofs
        for idx,val in np.ndenumerate(self.adjacency):
            if val == 1:
                #get the interpolation matrix
                self.collisions[idx[0]][idx[1]].inter_mat = get_interpolation_matrix(self.regions[idx[1]].fxn_space,
                                                                                     self.regions[idx[0]].fxn_space)
                #copy the dimensions of the interpolation matrix:
                self.collisions[idx[0]][idx[1]].pen_mat = self.collisions[idx[0]][idx[1]].inter_mat.duplicate()
                #prepopulate the penalty matrix term with the interpolation matrix
                # self.collisions[idx[0]][idx[1]].pen_mat.copy(self.collisions[idx[0]][idx[1]].inter_mat.duplicate())

            elif val == 0 and idx[0] != idx[1]:
                self.separations[idx[0]][idx[1]].mat.createAIJ([self.regions[idx[1]].system[0].getSize()[0],
                                                                self.regions[idx[0]].system[0].getSize()[1]])
                self.separations[idx[0]][idx[1]].mat.assemble()

        #populate an array of the same size as the adjacency matrix of the petsc matrices
        #  using a *stupidly* incomprehensible list "comprehension" 
        # this adds the unadultered system to the diaongals, the interpolation matrices where there is a collision
        # and the assembled empty matrices where there is a "separation"
        A_list = [ [self.regions[i].system[0] if i==j
                    else self.separations[i][j].mat if self.adjacency[i][j] == 0 and i!=j
                    else self.collisions[i][j].pen_mat 
                        for i in range(self.num_meshes)]
                     for j in range(self.num_meshes) ]
        
        # add the penalty to the relevant block of A_list
        for idx,val in np.ndenumerate(self.adjacency):
            # if idx[0]==idx[1]:
            if val==1:
                pen_term = PETSc.Mat().createAIJ(A_list[idx[0]][idx[0]].getSize())
                pen_term.assemble()
                diag = pen_term.getDiagonal()
                #set diagonal values to the pre-computed penalty vector values
                for val in self.collisions[idx[0]][idx[1]].penalty_dofs:
                    diag[val] = self.collisions[idx[0]][idx[1]].pen_vec[val]
                pen_term.setDiagonal(diag)
                # penalty_term = PETSc.Mat().createAIJ(I_mat.getSize())
                # I_mat.multTranspose(self.collisions[idx[0]][idx[1]].pen_vec,penalty_term)

                #add penalty term to diagonal block
                A_list[idx[0]][idx[0]].axpy(1.0,pen_term)

                #add penalty term to off diagonal block (scale the interpolation matrix)
                # A_list[idx[0]][idx[1]].scale(-self.alpha)

                #TODO: add the off-diagonal block which is the matrix mult of pen_term and the interpolation matrix
                # PETSc.MatMatMult(pen_term,self.collisions[idx[0]][idx[1]].inter_mat,A_list[idx[0]][idx[1]])
                A_list[idx[0]][idx[1]] = pen_term.matMult(self.collisions[idx[1]][idx[0]].inter_mat)
                A_list[idx[0]][idx[1]].assemble()
                A_list[idx[0]][idx[1]].scale(-1.0)
                # A_list[idx[0]][idx[1]] = A_list[idx[0]][idx[1]].matMult() self.collisions[idx[0]][idx[1]].pen_vec


        #create a list of the RHS vectors
        b_list = [region.system[1] for region in self.regions.values()]

        A = PETSc.Mat()
        A.createNest(A_list)

        b = PETSc.Vec()
        b.createNest(b_list)
        b.setUp()
        
        self.A = A
        self.b = b


    def _solve_coupled_system(self):
        '''
        Solve the assembled coupled system'''

        self._construct_solution_fxns()

        self.uh = PETSc.Vec()
        self.uh.createNest([region.fxn.vector for region in self.regions.values()])
        self.uh.setUp()

        #solve linear problem
        ksp = PETSc.KSP().create()
        ksp.setType(PETSc.KSP.Type.CG)
        ksp.setTolerances(rtol=1e-18)
        ksp.setOperators(self.A)
        ksp.setFromOptions()
        ksp.solve(self.b,self.uh)

        for region in self.regions.values():
            region.fxn.vector.ghostUpdate()

        self.solution = [region.fxn for region in self.regions.values()]


    def solve(self):

        self._construct_coupled_system()

        self._solve_coupled_system()

        return self.solution
    
def mesh_from_polygon(polygon, mesh_name="polygon_mesh", mesh_res=0.1):
    """
    Generate a GMSH mesh from a Shapely Polygon or MultiPolygon.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or MultiPolygon
        The input geometry to mesh.
    mesh_name : str
        A name for the gmsh model.
    mesh_res : float
        Target mesh element size (resolution).

    Returns
    -------
    dolfinx.mesh.Mesh
        The generated mesh.
    """
    if polygon.is_empty:
        raise ValueError("Provided polygon is empty.")

    gmsh.initialize()
    gmsh.model.add(mesh_name)
    point_id = 1
    curve_id = 1
    loop_id = 1
    surface_tags = []

    def estimate_mesh_resolution(poly, elements_across=30):
        """Estimate a good GMSH mesh resolution based on geometry size."""
        xmin, ymin = np.array(poly.exterior.coords[:-1]).min(axis=0)
        xmax, ymax = np.array(poly.exterior.coords[:-1]).max(axis=0)
        L = min(xmax - xmin, ymax - ymin)
        return L / elements_across
    
    def add_polygon(poly):
        nonlocal point_id, curve_id, loop_id
        point_map = {}

        # Outer boundary
        coords = list(poly.exterior.coords[:-1])  # omit duplicate endpoint
        outer_pts = []
        for x, y in coords:
            pid = gmsh.model.geo.addPoint(x, y, 0, mesh_res, point_id)
            point_map[(x, y)] = pid
            outer_pts.append(pid)
            point_id += 1
        outer_lines = []
        for i in range(len(outer_pts)):
            a = outer_pts[i]
            b = outer_pts[(i + 1) % len(outer_pts)]
            lid = gmsh.model.geo.addLine(a, b)
            outer_lines.append(lid)
            curve_id += 1
        outer_loop = gmsh.model.geo.addCurveLoop(outer_lines,reorient=True)

        # Inner holes
        inner_loops = []
        for interior in poly.interiors:
            coords = list(interior.coords[:-1])
            hole_pts = []
            for x, y in coords:
                pid = gmsh.model.geo.addPoint(x, y, 0, mesh_res, point_id)
                # pid = gmsh.model.geo.addPoint(x, y, 0, tag=point_id)
                point_map[(x, y)] = pid
                hole_pts.append(pid)
                point_id += 1
            hole_lines = []
            for i in range(len(hole_pts)):
                a = hole_pts[i]
                b = hole_pts[(i + 1) % len(hole_pts)]
                lid = gmsh.model.geo.addLine(a, b)
                hole_lines.append(lid)
                curve_id += 1
            hole_loop = gmsh.model.geo.addCurveLoop(hole_lines)
            inner_loops.append(hole_loop)

        surface = gmsh.model.geo.addPlaneSurface([outer_loop] + inner_loops)
        gmsh.model.geo.mesh.setRecombine(2, surface,angle=75)
        gmsh.model.addPhysicalGroup(2, [surface], tag=1)
        surface_tags.append(surface)

        return outer_lines
    mesh_size = estimate_mesh_resolution(polygon)

    # Support MultiPolygon
    if polygon.geom_type == "Polygon":
        lines = add_polygon(polygon)
    elif polygon.geom_type == "MultiPolygon":
        for poly in polygon.geoms:
            add_polygon(poly)
    else:
        raise TypeError(f"Unsupported geometry type: {type(polygon)}")

    gmsh.model.geo.synchronize()

    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", lines)  # or NodesList
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size*3)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", mesh_size*5)

    # gmsh.option.setNumber("Mesh.Optimize", 1)
    # gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size*3) 
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    # gmsh.option.setNumber("Mesh.Algorithm", 8)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.SmoothNormals", 1)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm",3)  # Blossom
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    # gmsh.option.setNumber("Mesh.Smoothing", 1)  # number of smoothing steps
    gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()

    mesh, _, _ = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    mesh.name = 'intersection'
    gmsh.finalize()
    return mesh


def compute_union_polygon(mesh_A, facet_tags_A, mesh_B, facet_tags_B, tag_val=1, gmsh_res=0.0):
    """
    Constructs a new mesh T^C over the overlapping region using boundary facet information
    from meshes A and B. Includes:
        - All vertices from tagged boundary facets
        - All intersection points between boundary lines of A and B

    Parameters
    ----------
    mesh_A, mesh_B : dolfinx.mesh.Mesh
        Input meshes.
    facet_tags_A, facet_tags_B : dolfinx.mesh.meshtags
        Boundary facet tags with value=tag_val on overlap boundaries.
    tag_val : int
        Tag marking boundary facets in the tags.
    gmsh_res : float
        Target resolution for gmsh meshing.

    Returns
    -------
    dolfinx.mesh.Mesh
        A new mesh over the overlap region.
    """

    # Build topology
    mesh_A.topology.create_connectivity(mesh_A.topology.dim - 1, 0)
    mesh_B.topology.create_connectivity(mesh_B.topology.dim - 1, 0)

    f2v_A = mesh_A.topology.connectivity(mesh_A.topology.dim - 1, 0)
    f2v_B = mesh_B.topology.connectivity(mesh_B.topology.dim - 1, 0)

    coords_A = mesh_A.geometry.x
    coords_B = mesh_B.geometry.x

    # Build LineStrings for A
    boundary_lines_A = []
    for facet in facet_tags_A.find(tag_val):
        verts = f2v_A.links(facet)
        pts = [coords_A[v,:2] for v in verts]
        boundary_lines_A.append(LineString(pts))

    # Build LineStrings for B
    boundary_lines_B = []
    for facet in facet_tags_B.find(tag_val):
        verts = f2v_B.links(facet)
        pts = [coords_B[v,:2] for v in verts]
        boundary_lines_B.append(LineString(pts))

    # Construct closed polygons
    poly_A = unary_union(polygonize(MultiLineString(boundary_lines_A)))
    poly_B = unary_union(polygonize(MultiLineString(boundary_lines_B)))
    
    #compute the polygon
    # poly_C = poly_A.intersection(poly_B)
    poly_C = poly_A.intersection(poly_B).simplify(1e-16)

    return poly_C


def get_overlap_boundary_facets(mesh, tags, tag_values=(1,2)):
    """
    Returns the boundary facets (edges) of the union of cells tagged with 1 or 2.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    tags : dolfinx.mesh.meshtags
        Cell tags where values 1 or 2 mark overlapping region.
    tag_values : tuple
        The tag values considered "in the overlap".

    Returns
    -------
    boundary_facets : np.ndarray
        Array of facet indices on the boundary of the overlapping region.
    """

    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    mesh.topology.create_connectivity(mesh.topology.dim - 1,  mesh.topology.dim)

    c2f = mesh.topology.connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    f2c = mesh.topology.connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    # Get the cells marked as overlapping
    overlap_cells = np.where(np.isin(tags.values, tag_values))[0]

    # Collect all facets belonging to these cells
    candidate_facets = []
    for cell in overlap_cells:
        for facet in c2f.links(cell):
            adj_cells = f2c.links(facet)
            #this finds all facets that border a cell NOT in the overlap region,
            # this is an "internal boundary facet"
            if np.any([c not in overlap_cells for c in adj_cells]):
                candidate_facets.append(facet)
            #we also need to find all facets that only belond to one cell
            if len(adj_cells) ==1:
                candidate_facets.append(facet)

    return np.array(candidate_facets, dtype=np.int32)


def mark_cells(msh, cell_index,partial=False,pts=None):
    num_cells = msh.topology.index_map(
        msh.topology.dim).size_local + msh.topology.index_map(
        msh.topology.dim).num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    values = np.full(cells.shape, 0, dtype=np.int32)
    if partial:
        #TODO: for some reason this bit of code tags the boundary cells as 1 and the interior cells as 2 (BOO!) need to do the opposite!
        values[cell_index] = np.full(len(cell_index), 2, dtype=np.int32)
        #find cells that are straddling the boundary
        boundary_cell_indices = cell_index[[np.isin(msh.topology.connectivity(2,0).links(cell_index[n]),pts).all() for n in range(len(cell_index))]]
        values[boundary_cell_indices] = np.full(len(boundary_cell_indices), 1, dtype=np.int32)

    else:
        values[cell_index] = np.full(len(cell_index), 1, dtype=np.int32)

    cell_tag = mesh.meshtags(msh, msh.topology.dim, cells, values)
    return cell_tag


def extract_cell_geometry(input_mesh, cell: int):
    mesh_nodes = cpp.mesh.entities_to_geometry(
        input_mesh._cpp_object, input_mesh.topology.dim, np.array([cell], dtype=np.int32), False)[0]

    return input_mesh.geometry.x[mesh_nodes]


def celltags_to_dofs(V,celltags):
    cells = celltags.indices[celltags.values==1]
    indices = np.unique(V.dofmap.list[cells,:].flatten())
    # vertices = np.unique(V.mesh.geometry.dofmap[cells,:].flatten())
    # indices=[]
    # for i in range(V.num_sub_spaces):
    #     _,dofmap = V.sub(i).collapse()
    #     indices.extend([dofmap[j] for j in vertices])
    return indices

def get_points_from_cells(mesh, cell_indices):
    """
    Get the unique points (vertices) that belong to a given set of cells.
    
    Parameters:
    - mesh: dolfinx.Mesh object
    - cell_indices: list or array of cell indices

    Returns:
    - np.ndarray of unique vertex coordinates
    """
    # Get connectivity from cells to vertices
    cell_to_vertex = mesh.topology.connectivity(2, 0)
    
    # Get the unique vertex indices from the selected cells
    vertex_indices = np.unique(np.concatenate([cell_to_vertex.links(cell) for cell in cell_indices]))

    # Retrieve the coordinates of these vertices
    return mesh.geometry.x[vertex_indices]

def get_collision_celltags(mesh0,mesh1,collisions,tol=1e-14,partial=False,pts=None):
    '''
    return celltags object for the collision between two meshes
    celltags are marked:
        0: not part of overlap region
        1: entirely contained in overlap region
    if 'partial' flag is True, 
        2: partially contained in overlap (overwritten from 1)
    '''
    cells0 = []
    cells1 = []
    for i, (cell0, cell1) in enumerate(collisions):
        geom0 = extract_cell_geometry(mesh0, cell0)
        geom1 = extract_cell_geometry(mesh1, cell1)
        distance = geometry.compute_distance_gjk(geom0, geom1)
        if np.linalg.norm(distance) <= tol:
            cells0.append(cell0)
            cells1.append(cell1)
    #np.unique instead of asarray
    if partial:
        celltags0 = mark_cells(mesh0, np.unique(np.asarray(cells0, dtype=np.int32)),partial=partial,pts=pts[0])
        celltags1 = mark_cells(mesh1, np.unique(np.asarray(cells1, dtype=np.int32)),partial=partial,pts=pts[1])
    else:
        celltags0 = mark_cells(mesh0, np.unique(np.asarray(cells0, dtype=np.int32)))
        celltags1 = mark_cells(mesh1, np.unique(np.asarray(cells1, dtype=np.int32)))

    return celltags0,celltags1


def get_bbtrees(meshes):
    bb_trees = []
    for msh in meshes:
        num_cells = msh.topology.index_map(msh.topology.dim).size_local + \
                msh.topology.index_map(msh.topology.dim).num_ghosts
        
        bb_tree=geometry.bb_tree(msh, msh.topology.dim, np.arange(num_cells, dtype=np.int32))
        bb_trees.append(bb_tree)

    return bb_trees

def pts_to_dofs(V,collision_pts):
    vertices=[]
    for i in range(collision_pts.num_nodes):
        if len(collision_pts.links(i)) > 0:
            vertices.append(i)
    indices=[]
    for i in range(V.num_sub_spaces):
        _,dofmap = V.sub(i).collapse()
        indices.extend([dofmap[j] for j in vertices])
    return indices


def get_petsc_system(a,L,bc=None):
    if bc is not None:
        A = assemble_matrix(fem.form(a),bcs=[bc])
    else: 
        A = assemble_matrix(fem.form(a))
    A.assemble()
    b=create_vector(fem.form(L))
    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b,fem.form(L))
    if bc is not None:
        fem.apply_lifting(b,[fem.form(a)],bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(b,[bc])
    return A,b


def interpolation_matrix_nonmatching_meshes(V_1,V_0): # Function spaces from nonmatching meshes
    '''
    V1: fxn space to be interpolated TO
    V0: fxn space to be interpolated FROM
    '''
    msh_0 = V_0.mesh
    msh_0.topology.dim
    msh_1 = V_1.mesh
    x_0   = V_0.tabulate_dof_coordinates()
    x_1   = V_1.tabulate_dof_coordinates()

    bb_tree         = geometry.bb_tree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, x_1)
    cells           = []
    points_on_proc  = []
    index_points    = []
    colliding_cells = geometry.compute_colliding_cells(msh_0, cell_candidates, x_1)

    for i, point in enumerate(x_1):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
            index_points.append(i)
            
    index_points_   = np.array(index_points)
    points_on_proc_ = np.array(points_on_proc, dtype=np.float64)
    cells_          = np.array(cells)

    ct      = cpp.mesh.to_string(msh_0.topology.cell_type)
    element = basix.create_element(basix.finite_element.string_to_family(
        "Lagrange", ct), basix.CellType[ct], V_0.ufl_element().degree, basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), 2))

    for i in range(0, len(cells_)):
        # geom_dofs  = msh_0.geometry.dofmap.links(cells_[i])
        geom_dofs  = list(msh_0.geometry.dofmap[cells_[i]])
        x_ref[i,:] = msh_0.geometry.cmap.pull_back(np.array([points_on_proc_[i,:]]), msh_0.geometry.x[geom_dofs])

    basis_matrix = element.tabulate(0, x_ref)[0,:,:,0]

    cell_dofs         = np.zeros((len(x_1), len(basis_matrix[0,:])))
    basis_matrix_full = np.zeros((len(x_1), len(basis_matrix[0,:])))


    for nn in range(0,len(cells_)):
        cell_dofs[index_points_[nn],:] = V_0.dofmap.cell_dofs(cells_[nn])
        basis_matrix_full[index_points_[nn],:] = basis_matrix[nn,:]

    cell_dofs_ = cell_dofs.astype(int) ###### REDUCE HERE

    # [JEF] I = np.zeros((len(x_1), len(x_0)), dtype=complex)
    # make a petsc matrix here instead of np- 
    # for Josh: probably more efficient ways to do this 
    I = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    I.setSizes((len(x_1), len(x_0)))
    I.setUp()
    for i in range(0,len(x_1)):
        for j in range(0,len(basis_matrix[0,:])):
            # [JEF] I[i,cell_dofs_[i,j]] = basis_matrix_full[i,j]
            I.setValue(i,cell_dofs_[i,j],basis_matrix_full[i,j])

    return I


def permute_and_expand_matrix(V_to,V_from,M_scalar,mixed=False):
    '''return assembled PETSc matrix that interpolates from one mesh to another'''
    # if there is a mixed space, e.g. num_sub_spaces != value_size, need to loop through subspaces of subspaces
    indices_to = []
    indices_from = []
    # if mixed is True:
    #     sub_indices_to = []
    #     sub_indices_from = []

    for i in range(V_to.num_sub_spaces):
        if mixed is False:
            _,map_to = V_to.sub(i).collapse()
            _,map_from = V_from.sub(i).collapse()
            indices_to.extend(map_to)
            indices_from.extend(map_from)

        elif mixed is True:
            for j in range(V_to.sub(i).num_sub_spaces):
                _,map_to = V_to.sub(i).sub(j).collapse()
                _,map_from = V_from.sub(i).sub(j).collapse()
                
                indices_to.extend(map_to)
                indices_from.extend(map_from)

        # #sort sub-subspaces if using a mixed formulation
        # if mixed is True:
            
        #     for j in range(V_to.sub(i).num_sub_spaces):
        #         _, sub_map_to = V_to.sub(i).sub(j).collapse()
        #         _, sub_map_from = V_from.sub(i).sub(j).collapse()
                
        #         sub_indices_to.extend(sub_map_to)
        #         sub_indices_from.extend(sub_map_from)

        #     # Sort sub_indices_to based on sub_map_to order
        #     #   this returns the map from subspace to the sub-subspace
        #     sub_sort_to = np.argsort(sub_indices_to)
        #     sub_sort_from = np.argsort(sub_indices_from)

        #     map_to = list(np.array(map_to)[sub_sort_to])
        #     map_from = list(np.array(map_from)[sub_sort_from])
                
        

    #provide the permutations to sort these indices from the block diagonal form
    sort_to = np.argsort(indices_to)
    sort_from = np.argsort(indices_from)

    #construct block diagonal matrix (with num_blocks = num_subspaces), then permute
    dim0,dim1 = M_scalar.getSize()
    M0 = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    M0.setSizes((dim0, dim1))
    M0.setUp()
    
    #list comprehension to construct blocks for the nested PETSc matrix
    M_list = [[M_scalar if i==j
               else M0
                for i in range(V_to.value_shape[0])]
               for j in range(V_to.value_shape[0])]

    #construct nested petsc matrix from the list of interpolation matrices
    M_nest = PETSc.Mat(comm=MPI.COMM_WORLD)
    M_nest.createNest(M_list)
    M_nest.assemble()

    #convert nested matrix to normal PETSc matrix to allow for permutation
    M_unpermuted = M_nest.convert('aij')

    row_perm = PETSc.IS().createGeneral(list(sort_to))
    col_perm = PETSc.IS().createGeneral(list(sort_from))    
    
    M = M_unpermuted.permute(row_perm,col_perm)
    M.assemble()

    return M

def get_interpolation_matrix(V_1,V_0,mixed=False):
    '''
    returns the interpolation matrix from one functionspace on a mesh to another
    V_1: functionspace to interpolate to
    V_0: functionspace to interpolate from
    '''
    # In order to properly handle vector fxn spaces and mixed function spaces, 
    #   we need to us the dofmaps and the interpolation matrix from the scalar fxn space
    if mixed is True:
        M01 = interpolation_matrix_nonmatching_meshes(V_1.sub(0).sub(0).collapse()[0],
                                                      V_0.sub(0).sub(0).collapse()[0])
    else: 
        M01 = interpolation_matrix_nonmatching_meshes(V_1,V_0)

    #assemble the PETSc matrix
    M01.assemble()

    #expand the interpolation matrix based on the ordering of the subspace
    if len(V_1.value_shape) != 0:
        M01_expanded = permute_and_expand_matrix(V_1,V_0,M01,mixed=mixed)
        return M01_expanded
    else:
        return M01
    
def get_nn_interpolation_matrix(pts1,pts2):
    '''return a nearest-neighbor interpolation matrix from pts1 to pts2'''
    interpolation_matrix = np.zeros((pts2.shape[0],pts1.shape[0]))
    tree = cKDTree(pts2)
    dists, indices = tree.query(pts1)
    row_indices = list(indices)
    col_indices = [i for i in range(pts1.shape[0])]
    interpolation_matrix[row_indices,col_indices] = 1

    return interpolation_matrix



def solve_coupled_system(A1,A2,b1,b2,uh1,uh2,M12,M21,subdofs1,subdofs2,alpha,w=1):
    '''construct, assemble and solve a linear interpolation based coupled system'''
    
    # #relative weights of each solution
    # w1 = 1
    # w2 = 1*w
    

    #Select overlapping section dofs for penalty parameter by constructing near-identity matrices
    #construct empty matrices
    I1 = PETSc.Mat().createAIJ(A1.getSize())
    I2 = PETSc.Mat().createAIJ(A2.getSize())
    I1.assemble()
    I2.assemble()
    #extract and then set overlapping section dofs to 1
    diag1 = I1.getDiagonal()
    diag2 = I2.getDiagonal()
    for val1 in subdofs1:
        diag1[val1] = 1.0
    for val2 in subdofs2:
        diag2[val2] = 1.0
    I1.setDiagonal(diag1)
    I2.setDiagonal(diag2)

    #add penalty term to diagonal blocks
    A1.axpy(alpha,I1)
    A2.axpy((w**2)*alpha,I2)
    #add penalty term to off-diagonal blocks
    M21.scale(-alpha)
    M12.scale(-(w**2)*alpha)
    
    A = PETSc.Mat()
    A.createNest([[A1, M21],
                  [M12, A2]])

    b = PETSc.Vec()
    b.createNest([b1,b2])
    b.setUp()

    uh = PETSc.Vec()
    uh.createNest([uh1.vector,uh2.vector])
    uh.setUp()

    #solve linear problem
    ksp = PETSc.KSP().create()
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.setTolerances(rtol=1e-18)
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b,uh)

    uh1.vector.ghostUpdate()
    uh2.vector.ghostUpdate()

