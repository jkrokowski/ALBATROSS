from scipy.sparse import csr_matrix,vstack
from petsc4py import PETSc
from ALBATROSS.utils import sparseify
   
def convert_petsc_to_numpy(mat,sparse='False'):
    mataij = mat.convert('aij')
    mat_sparse = csr_matrix(mataij.getValuesCSR()[::-1], shape=mataij.size)
    mat_np = mat_sparse.toarray()
    
    if sparse is True:
        return mat,mat_sparse
    else:
        return mat_np

def sparse_mat_from_loflofvec(l_of_l_of_vec):
    flat_list = [vec for row in l_of_l_of_vec for vec in row]
    sparse_list = []
    for vec in flat_list:
        sparse_list.append(sparseify(vec.array))
    sparse_mat = vstack(sparse_list)
    return sparse_mat
    
def AT_C_B(A: PETSc.Mat, M: PETSc.Mat, B: PETSc.Mat) -> PETSc.Mat:
    """
    Compute A M B using PETSc matrix multiplies.
    A: PETSc.Mat of size (nC x nA)
    B: PETSc.Mat of size (nC x nB)
    M: PETSc.Mat of size (nC x nC)
    Returns: PETSc.Mat of size (nA x nB)
    """
    # Step 1: MB = M * B
    MB = M.matMult(B)

    # Step 2: Aᵀ * (M * B)
    result = A.transposeMatMult(MB)
    return result

def A_C_B(A: PETSc.Mat, M: PETSc.Mat, B: PETSc.Mat) -> PETSc.Mat:
    """
    Compute A M B using PETSc matrix multiplies.
    A: PETSc.Mat of size (nC x nA)
    B: PETSc.Mat of size (nC x nB)
    M: PETSc.Mat of size (nC x nC)
    Returns: PETSc.Mat of size (nA x nB)
    """
    # Step 1: MB = M * B
    MB = M.matMult(B)
    # Step 2: A* (M * B)
    result = A.matMult(MB)
    return result