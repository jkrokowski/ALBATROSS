from petsc4py import PETSc 
from dolfinx import cpp, la 



def  assembleLinearSystemBackground(a_f, L_f, M):
    """
    Assemble the linear system on the background mesh, with
    variational forms defined on the foreground mesh.
    
    Parameters
    ----------
    a_f: LHS PETSc matrix
    L_f: RHS PETSc matrix
    M: extraction petsc matrix 
    
    Returns
    -------  
    A_b: PETSc matrix on the background mesh
    b_b: PETSc vector on the background mesh
    """

    A_b = AT_R_A(M, a_f)
    b_b = AT_x(M, L_f)
    return A_b, b_b

def AT_R_A(A, R):
    """
    Compute "A^T*R*A". A,R are "petsc4py.PETSc.Mat".

    Parameters
    -----------
    A : petsc4py.PETSc.Mat
    R : petsc4py.PETSc.Mat

    Returns
    ------ 
    ATRA : petsc4py.PETSc.Mat
    """
    AT = A.transpose()
    ATR = AT.matMult(R)
    ATT = A.transpose()
    ATRA = ATR.matMult(ATT)
    return ATRA


def AT_x(A, x):
    """
    Compute b = A^T*x.
    Parameters
    ----------
    A : petsc4py.PETSc.Mat
    x : petsc4py.PETSc.Vec
    Returns
    -------
    b_PETSc : petsc4py.PETSc.Vec
    """
    
    b_PETSc = A.createVecRight()
    A.multTranspose(x, b_PETSc)
    return b_PETSc

def transferToForeground(u_f, u_b, M):
    """
    Transfer the solution vector from the background to the forground
    mesh.
    
    Parameters
    ----------
    u_f: Dolfin function on the foreground mesh
    u_b: PETSc vector of soln on the background mesh 
    M: extraction matrix from background to foreground.
    """
    u_petsc = cpp.la.petsc.create_vector_wrap(u_f.x)
    #u_petsc = la.create_petsc_vector_wrap(u_f.x)
    M.mult(u_b, u_petsc)
    u_f.x.scatter_forward()






def solveKSP(A,b,u,method='gmres', PC='jacobi',
            remove_zero_diagonal=False,rtol=1E-8, 
            atol=1E-9, max_it=1000000, bfr_tol=1E-9,
            monitor=True,gmr_res=3000,bfr_b=True):
    """
    solve linear system A*u=b
    A: PETSC Matrix
    b: PETSC Vector
    u: PETSC Vector
    """
    #print('in solve ksp')
    #u.setUp()
    if method == None:
        method='gmres'
    if PC == None:
        PC='jacobi'

    if method == 'mumps':

        ksp = PETSc.KSP().create() 
        ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)
        '''
        if remove_zero_diagonal and bfr_tol is not None:
            if bfr_b:
                A,b = trimNodes(A,b=b,bfr_tol=bfr_tol)
            else:
                A,_ = trimNodes(A,bfr_tol=bfr_tol)
        '''

        opts = PETSc.Options("mat_mumps_")
        # icntl_24: controls the detection of “null pivot rows", 1 for on, 0 for off
        # without basis function removal, this needs to be 1 to avoid crashing 
        opts["icntl_24"] = 1
        # cntl_3: is used to determine null pivot rows
        opts["cntl_3"] = 1e-12           

        # I don't entirely know what these options do 
        #opts["icntl_1"] = 1 # controls output of error info? 
        #opts["icntl_2"] = 1
        #opts["icntl_13"] = 1
        #opts["icntl_10"] = 5
        #opts["icntl_6"] = 7


        #A_new.assemble()
        #ksp.setOperators(A_new)
        A.assemble()
        ksp.setOperators(A)
        ksp.setType('preonly')
        pc=ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        ksp.setUp()

        ksp.solve(b,u)
        #u.x.scatter_forward()

        #ksp_d = PETScKrylovSolver(ksp)
        #ksp_d.solve(PETScVector(u),PETScVector(b))
        return 


    ksp = PETSc.KSP().create() 
    ksp.setTolerances(rtol=rtol, atol = atol, max_it= max_it)

    if method == 'gmres': 
        ksp.setType(PETSc.KSP.Type.FGMRES)
    elif method == 'gcr':
        ksp.setType(PETSc.KSP.Type.GCR)
    elif method == 'cg':
        ksp.setType(PETSc.KSP.Type.CG)


    '''
    if remove_zero_diagonal and bfr_tol is not None:
        A,b = trimNodes(A,b=b, bfr_tol=bfr_tol)
    '''

    if PC == 'jacobi':
        A.assemble()
        ksp.setOperators(A)
        #ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("jacobi")
        ksp.setUp()
        ksp.setGMRESRestart(300)

    elif PC == 'ASM':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("asm")
        pc.setASMOverlap(1)
        ksp.setUp()
        localKSP = pc.getASMSubKSP()[0]
        localKSP.setType(PETSc.KSP.Type.FGMRES)
        localKSP.getPC().setType("lu")
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ICC':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("icc")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC== 'ILU':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("euclid")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

    elif PC == 'ILUT':
        A.assemble()
        ksp.setOperators(A)
        ksp.setFromOptions()
        pc = ksp.getPC()
        pc.setType("hypre")
        pc.setHYPREType("pilut")
        ksp.setUp()
        ksp.setGMRESRestart(gmr_res)

        '''
        setHYPREType(self, hypretype):
        setHYPREDiscreteCurl(self, Mat mat):
        setHYPREDiscreteGradient(self, Mat mat):
        setHYPRESetAlphaPoissonMatrix(self, Mat mat):
        setHYPRESetBetaPoissonMatrix(self, Mat mat=None):
        setHYPRESetEdgeConstantVectors(self, Vec ozz, Vec zoz, Vec zzo=None):                
        '''
    print('ready to solve')
    #print(dir(ksp))
    #monitorObj = ksp.getMonitor()
    #print(monitorObj)
    #print(dir(monitorObj))

    '''
    ksp.parameters['monitor_convergence'] = True
    ksp.parameters['absolute_tolerance'] = atol
    ksp.parameters['relative_tolerance'] = rtol
    ksp.parameters['maximum_iterations'] = max_it
    ksp.parameters['nonzero_initial_guess'] = True
    ksp.parameters['error_on_nonconvergence'] = False
    '''
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["ksp_view"] = None
    ksp.setFromOptions()
    ksp.solve(b,u)
    print('solving')
    #u.x.scatter_forward()

    '''
    ksp_d = PETScKrylovSolver(ksp)
    if monitor:
        ksp_d.parameters['monitor_convergence'] = True
    ksp_d.parameters['absolute_tolerance'] = atol
    ksp_d.parameters['relative_tolerance'] = rtol
    ksp_d.parameters['maximum_iterations'] = max_it
    ksp_d.parameters['nonzero_initial_guess'] = True
    ksp_d.parameters['error_on_nonconvergence'] = False
    ksp_d.solve(PETScVector(u),PETScVector(b))
    '''

    history = ksp.getConvergenceHistory()
    if monitor:
        print('Converged in', ksp.getIterationNumber(), 'iterations.')
        print('Convergence history:', history)

def estimateConditionNumber(A,b,u,rtol=1E-8, atol=1E-9, max_it=100000, PC=None):

    ksp = PETSc.KSP().create() 
    ksp.setComputeSingularValues(True)
    ksp.setTolerances(rtol=rtol, atol = atol, max_it=max_it)
    ksp.setType(PETSc.KSP.Type.GMRES)

    A.assemble()
    ksp.setOperators(A)
    #PETSc.Options().setValue("ksp_monitor","")
    PETSc.Options().setValue("ksp_view_singularvalues","")
    ksp.setFromOptions()  
    pc = ksp.getPC()
    if PC is not None:
        pc.setType(PC)
    else: 
        pc.setType("none")
    ksp.setUp()
    ksp.setGMRESRestart(1000)
    ksp.solve(b, u)

    smax,smin = ksp.computeExtremeSingularValues()

    return smax,smin