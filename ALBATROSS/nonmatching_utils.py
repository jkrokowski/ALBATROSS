import numpy as np
from dolfinx import cpp,mesh,fem,geometry
from dolfinx.fem.petsc import assemble_vector,assemble_matrix,create_vector
from petsc4py import PETSc
import basix
from mpi4py import MPI

def mark_cells(msh, cell_index):
    num_cells = msh.topology.index_map(
        msh.topology.dim).size_local + msh.topology.index_map(
        msh.topology.dim).num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    values = np.full(cells.shape, 0, dtype=np.int32)
    values[cell_index] = np.full(len(cell_index), 1, dtype=np.int32)
    cell_tag = mesh.meshtags(msh, msh.topology.dim, cells, values)
    return cell_tag


def extract_cell_geometry(input_mesh, cell: int):
    mesh_nodes = cpp.mesh.entities_to_geometry(
        input_mesh._cpp_object, input_mesh.topology.dim, np.array([cell], dtype=np.int32), False)[0]

    return input_mesh.geometry.x[mesh_nodes]


def celltags_to_dofs(V,cell_tags):
    cells = cell_tags.indices[cell_tags.values==1]
    vertices = np.unique(V.mesh.geometry.dofmap[cells,:].flatten())
    indices=[]
    for i in range(V.num_sub_spaces):
        _,dofmap = V.sub(i).collapse()
        indices.extend([dofmap[j] for j in vertices])
    return indices


def get_collision_celltags(mesh0,mesh1,collisions,tol=1e-14):
    '''return celltags on '''
    cells0 = []
    cells1 = []
    for i, (cell0, cell1) in enumerate(collisions):
        geom0 = extract_cell_geometry(mesh0, cell0)
        geom1 = extract_cell_geometry(mesh1, cell1)
        distance = geometry.compute_distance_gjk(geom0, geom1)
        if np.linalg.norm(distance) <= tol:
            cells0.append(cell0)
            cells1.append(cell1)
    celltags0 = mark_cells(mesh0, np.asarray(cells0, dtype=np.int32))
    celltags1 = mark_cells(mesh1, np.asarray(cells1, dtype=np.int32))

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
        "Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree, basix.LagrangeVariant.equispaced)

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


def permute_and_expand_matrix(V_to,V_from,M_scalar):
    '''return assembled PETSc matrix that interpolated from one mesh to another'''
    indices_to = []
    indices_from = []
    for i in range(V_to.num_sub_spaces):
        _,map_to = V_to.sub(i).collapse()
        _,map_from = V_from.sub(i).collapse()
        indices_to.extend(map_to)
        indices_from.extend(map_from)

    #provide the permutations to sort these indices from the block diagonal form
    sort_to = np.argsort(indices_to)
    sort_from = np.argsort(indices_from)

    #construct block diagonal matrix (with num_blocks = num_subspaces), then permute
    dim0,dim1 = M_scalar.getSize()
    M0 = PETSc.Mat().create(comm=MPI.COMM_WORLD)
    M0.setSizes((dim0, dim1))
    M0.setUp()

    M12 = M0
    M21 = M0

    M11 = M_scalar
    M22 = M_scalar

    #need to figure out how to customize this nesting structure based on the number of subspaces
    M_nest = PETSc.Mat(comm=MPI.COMM_WORLD)
    M_nest.createNest([[M11, M12],
                  [M21, M22]])
    M_nest.assemble()

    #convert nested matrix to normal PETSc matrix to allow for permutation
    M_unpermuted = M_nest.convert('aij')

    row_perm = PETSc.IS().createGeneral(list(sort_to))
    col_perm = PETSc.IS().createGeneral(list(sort_from))    
    
    M = M_unpermuted.permute(row_perm,col_perm)
    M.assemble()

    return M

def get_interpolation_matrix(V_1,V_0):
    M01 = interpolation_matrix_nonmatching_meshes(V_1,V_0)
    M01.assemble()
    M01_expanded = permute_and_expand_matrix(V_1,V_0,M01)

    return M01_expanded


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

