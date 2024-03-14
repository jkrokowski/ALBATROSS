'''
Implementation of 2D/3D Poisson's problem with unfitted mesh
and Nitsche's method to weakly apply the BCs
'''

from mpi4py import MPI

from dolfinx import mesh, fem, io, cpp, geometry
from dolfinx.fem import petsc
from dolfinx.cpp import fem as c_fem, io as c_io
import os
import ufl
from petsc4py import PETSc
import numpy as np
import basix
from timeit import default_timer
import linAlgHelp
comm = MPI.COMM_WORLD

from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem import locate_dofs_topological,dirichletbc,VectorFunctionSpace,Function,TensorFunctionSpace
from ufl import tr,sym,grad,Constant,Identity,TrialFunction,TestFunction,inner,dx
from dolfinx import geometry,mesh,plot
from mpi4py import MPI
import numpy as np
import pyvista
from petsc4py.PETSc import ScalarType

def generateUnfittedMeshes(corner_b = [[0,0],[20,5]], corner_f1 = [[1.1,1.1],[10.1,3.1]],corner_f2 = [[9.15,2.15],[19.15,4.15]],\
                        N_b = 20, N_f1=10, N_f2 = 10,\
                        angle_1 =10, angle_2 = 10):
    """
    Generate simple unfitted foreground and background meshes, with the 
    forground/background mesh rotated to make the boundary cells
    cut by the background/foreground cell edges.
    
    Parameters
    ----------
    corner_b, corner_f1, corner_f2: corner coords of meshes, pre rotation
    N_b, N_f1, N_f2: numbers of elements on each edge
    angle_1, angle_2: rotation angle for fg meshes (about corner 1)
    
    Returns
    -------
    mesh_f, mesh_b
    """
    mesh_b =  mesh.create_rectangle(comm, [np.array(corner_b[0]), np.array(corner_b[1])],
                            [N_b, N_b], mesh.CellType.triangle)
    mesh_f1 =  mesh.create_rectangle(comm, [np.array(corner_f1[0]), np.array(corner_f1[1])],
                            [N_f1, N_f1], mesh.CellType.triangle)
    mesh_f2 =  mesh.create_rectangle(comm, [np.array(corner_f2[0]), np.array(corner_f2[1])],
                            [N_f2, N_f2], mesh.CellType.triangle)
    #mesh_f1.rotate(angle_1, 2,corner_f1[0])
    #mesh_f2.rotate(angle_2, 2,corner_f2[0])
    return mesh_b, mesh_f1, mesh_f2

def interiorResidual(u,v,f,dx_,):
    '''
    Linear elasticity problem interior residual
    '''
    def eps(v):
        return sym(grad(v))

    # E = Constant(domain,1e5)
    # nu = Constant(domain,0.3)
    E = 1e5
    nu = 0.3
    model = "plane_stress"

    mu = E/2/(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    if model == "plane_stress":
        lmbda = 2*mu*lmbda/(lmbda+2*mu)

    def sigma(v):
        return lmbda*tr(eps(v))*Identity(2) + 2.0*mu*eps(v)


    return inner(sigma(u), eps(v))*dx_ \
            - ufl.inner(f, v)*dx_
    


def boundaryResidual(u,v,u_exact,ds_,domain,
                        sym=True,
                        beta_value=10,
                        overPenalize=False,
                        h=None):

    '''
    Nitsches method for poisson problem 
    Formulation from Github:
    https://github.com/MiroK/fenics-nitsche/blob/master/poisson/poisson_circle_dirichlet.py
    
    Note RE h: 'real' Nitsches method will use the size of the cut portion of the background cell, 
    I (jennifer E fromm) usually use the size of the fg mesh element or average fg mesh element size 
    instead for ease of computation

    '''
    beta =beta_value
    if sym:
        sgn = 1.0
    else:
        sgn = -1.0
    n = ufl.FacetNormal(domain)
    if h is not None:
        h_E = h
    else:
        size = ufl.JacobianDeterminant(domain)
        h_E = size**(0.5)
    const = - ufl.inner(ufl.dot(ufl.grad(u), n), v)*ds_ 
    retval = const + sgn*ufl.inner(u_exact-u, ufl.dot(ufl.grad(v), n))*ds_ 
    penalty = beta*h_E**(-1)*ufl.inner(u-u_exact, v)*ds_
    if (overPenalize or sym):
        retval += penalty
    return penalty #retval
    

print(">>> Generating mesh...")
N_b=20
N_f1=10
N_f2=12

h1 = 2/N_f1
h2 = 2/N_f2
mesh_b, mesh_f1, mesh_f2= generateUnfittedMeshes(N_b=N_b, N_f1=N_f1, N_f2=N_f2)

# define polynomial order
k = 1
# define exact solutions 
def u_exact_fun(x): 
    return  np.sin(0.1*(x[1] + x[0]+ 0.1))*np.cos(0.1*(x[1] + x[0]- 0.1))
def u_exact_ufl(x): 
    return ufl.sin(0.1*(x[1] + x[0]+ 0.1))*ufl.cos(0.1*(x[1] + x[0]- 0.1))

#first mesh 
V1 = VectorFunctionSpace(mesh_f1, ("CG", k))
u1 = fem.Function(V1)
v1 = ufl.TestFunction(V1)
u_ex_disp1 = fem.Function(V1)

u_ex_disp1.interpolate(u_exact_fun)

x1 = ufl.SpatialCoordinate(mesh_f1)
u_ex1 = u_exact_ufl(x1)
rho_g = 1e-3
f1 = Constant(mesh_f1,ScalarType((0, -rho_g)))
dx_1 = ufl.Measure('dx',domain=mesh_f1, metadata={'quadrature_degree': 2*k})
ds_1 = ufl.Measure('ds',domain=mesh_f1, metadata={'quadrature_degree': 2*k})
res_interior1 = interiorResidual(u1, v1, f1, dx_1)
res_boundary1 = boundaryResidual(u1, v1, u_ex1,ds_1, mesh_f1,h=h1)
res1 = res_interior1+ res_boundary1
J1 = ufl.derivative(res1,u1)

res1_form = fem.form(res1)
res1_petsc = fem.petsc.assemble_vector(res1_form)
#for i in range(res1_petsc.getSize()):
#    print(res1_petsc.getValue(i))


J1_form = fem.form(J1)
J1_petsc = fem.petsc.assemble_matrix(J1_form)
J1_petsc.assemble()
res1_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
res1_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


#for i in range(J1_petsc.getSize()[0]):
#    for j in range(J1_petsc.getSize()[1]):
#        print(J1_petsc.getValue(i,j))

#second mesh 
V2 = fem.FunctionSpace(mesh_f2,('CG',k))
u2 = fem.Function(V2)
v2 = ufl.TestFunction(V2)
u_ex_disp2 = fem.Function(V2)

u_ex_disp2.interpolate(u_exact_fun)

x2 = ufl.SpatialCoordinate(mesh_f2)
u_ex2 = u_exact_ufl(x2) 
f2 = -ufl.div(ufl.grad(u_ex2))
dx_2 = ufl.Measure('dx',domain=mesh_f2, metadata={'quadrature_degree': 2*k})
ds_2 = ufl.Measure('ds',domain=mesh_f2, metadata={'quadrature_degree': 2*k})
res_interior2 = interiorResidual(u2, v2, f2, dx_2)
res_boundary2 = boundaryResidual(u2, v2, u_ex2,ds_2, mesh_f2,h=h2)
res2 = res_interior2+ res_boundary2
J2 = ufl.derivative(res2,u2)

res2_form = fem.form(res2)
res2_petsc = fem.petsc.assemble_vector(res2_form)
J2_form = fem.form(J2)
J2_petsc = fem.petsc.assemble_matrix(J2_form)
J2_petsc.assemble()
res2_petsc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
res2_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# define bg space:
Vb = fem.FunctionSpace(mesh_b,('CG',k))


# make extraction operators- not supported by this method anymore
#M1 = c_fem.PETScDMCollection.create_transfer_matrix(Vb, V1)
#M2 = c_fem.PETScDMCollection.create_transfer_matrix(Vb, V2)

# found this code from online that does somthing similar (i think)
#https://fenicsproject.discourse.group/t/interpolation-matrix-with-non-matching-meshes/12204/13
def interpolation_matrix_nonmatching_meshes(V_1,V_0): # Function spaces from nonmatching meshes
    msh_0 = V_0.mesh
    msh_0.topology.dim
    msh_1 = V_1.mesh
    x_0   = V_0.tabulate_dof_coordinates()
    x_1   = V_1.tabulate_dof_coordinates()

    bb_tree         = geometry.BoundingBoxTree(msh_0, msh_0.topology.dim)
    cell_candidates = geometry.compute_collisions(bb_tree, x_1)
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
        "Lagrange", ct), basix.cell.string_to_type(ct), V_0.ufl_element().degree(), basix.LagrangeVariant.equispaced)

    x_ref = np.zeros((len(cells_), 2))

    for i in range(0, len(cells_)):
        geom_dofs  = msh_0.geometry.dofmap.links(cells_[i])
        x_ref[i,:] = msh_0.geometry.cmap.pull_back([points_on_proc_[i,:]], msh_0.geometry.x[geom_dofs])

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

M1 = interpolation_matrix_nonmatching_meshes(V1,Vb)
M2 = interpolation_matrix_nonmatching_meshes(V2,Vb)
M1.assemble()
M2.assemble()

A1,b1 = linAlgHelp.assembleLinearSystemBackground(J1_petsc,-res1_petsc,M1)
A2,b2 = linAlgHelp.assembleLinearSystemBackground(J2_petsc,-res2_petsc,M2)


# add the two matrices
A1.axpy(1.0,A2)
b1.axpy(1.0,b2)

x = A1.createVecLeft()

# solve on bg mesh
# note: cannot use direct solver w/ dense matrices 
linAlgHelp.solveKSP(A1,b1,x,monitor=False,method='mumps')
#linAlgHelp.solveKSP(A1,b1,x,rtol=1E-18, atol=1E-19)

#for i in range(x.getSize()):
#    if x.getValue(i) > 0:
#        print(x.getValue(i))


linAlgHelp.transferToForeground(u1, x, M1)
linAlgHelp.transferToForeground(u2, x, M2)



L2_error = fem.form(ufl.inner(u1 - u_ex1 ,u1 - u_ex1) * dx_1)
error_local = fem.assemble_scalar(L2_error)
error_L2_1  = np.sqrt(mesh_f1.comm.allreduce(error_local, op=MPI.SUM))

H10_error = fem.form(ufl.inner(ufl.grad(u1 - u_ex1), ufl.grad(u1 - u_ex1)) *dx_1 )
error_local = fem.assemble_scalar(H10_error)
error_H10_1  = np.sqrt(mesh_f1.comm.allreduce(error_local, op=MPI.SUM))

L2_error = fem.form(ufl.inner(u2 - u_ex2, u2 - u_ex2) * dx_2)
error_local = fem.assemble_scalar(L2_error)
error_L2_2  = np.sqrt(mesh_f2.comm.allreduce(error_local, op=MPI.SUM))

H10_error = fem.form(ufl.inner(ufl.grad(u2 - u_ex2), ufl.grad(u2 - u_ex2)) *dx_2)
error_local = fem.assemble_scalar(H10_error)
error_H10_2  = np.sqrt(mesh_f2.comm.allreduce(error_local, op=MPI.SUM))

net_L2 = error_L2_1 + error_L2_2
net_H10 = error_H10_1 + error_H10_2


if mesh_f1.comm.rank == 0:
    print(f"Error_L2: {net_L2}")
    print(f"Error_H10: {net_H10}")
    print(f"Error_L2 (fg mesh 1): {error_L2_1}")
    print(f"Error_H10 (fg mesh 1): {error_H10_1}")
    print(f"Error_L2 (fg mesh 2): {error_L2_2}")
    print(f"Error_H10 (fg mesh 2): {error_H10_2}")

def outputXDMF(f,V,folder,name):
    ''' 
    function to interpolate a ufl object onto a function space, and then 
    plot the function on the function space's domain to visualize as an 
    xdmf file 
    '''
    domain = V.mesh
    f_expr = fem.Expression(f, V.element.interpolation_points())
    f_fun= fem.Function(V)
    f_fun.interpolate(f_expr)
    xdmf = io.XDMFFile(domain.comm, folder +name + ".xdmf", "w")
    xdmf.write_mesh(domain)
    xdmf.write_function(f_fun)

folder = 'poisson/'

visOutput = True
if visOutput:
    outputXDMF(u_ex_disp1,V1,folder,'u1_exact')
    outputXDMF(u1,V1,folder,'u1_soln')
    outputXDMF(u_ex_disp2,V2,folder,'u2_exact')
    outputXDMF(u2,V2,folder,'u2_soln')