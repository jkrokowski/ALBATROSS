# Cross Sectional Analysis using the Principle of Minimum Total Complementary Energy
# ================

from mpi4py import MPI
import dolfinx.cpp.mesh
from dolfinx import mesh,plot
from dolfinx.fem import locate_dofs_topological,Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace,locate_dofs_geometrical
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import (sym,FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
import matplotlib.pylab as plt
import time
from slepc4py import SLEPc

from dolfinx import geometry
t0 = time.time()
# Create 2d mesh and define function space
N = 23
W = 1
H = 1
# domain = mesh.create_unit_square(MPI.COMM_WORLD,N,N, mesh.CellType.quadrilateral)
domain = mesh.create_rectangle( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N], cell_type=mesh.CellType.quadrilateral)
# domain2 = mesh.create_interval( MPI.COMM_WORLD,np.array([[0,0],[W, H]]),[N,N])
# Create 3d mesh (to be used to calculate perturbed displacement field)
# L,W,H = 10,1,1
# domain2 = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, H])],
#                   [1,N,N], cell_type=mesh.CellType.hexahedron)
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
if False:
     tdim = domain.topology.dim
     topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
     plotter = pyvista.Plotter()
     plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     plotter.view_isometric()
     if not pyvista.OFF_SCREEN:
          plotter.show()
     # tdim = domain2.topology.dim
     # topology, cell_types, geometry = plot.create_vtk_mesh(domain2, tdim)
     # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
     # plotter = pyvista.Plotter()
     # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     # if not pyvista.OFF_SCREEN:
     #      plotter.show()

# Construct Displacment Coefficient mixed function space
Ve = VectorElement("CG",domain.ufl_cell(),1,dim=3)
#TODO:check on whether TensorFxnSpace is more suitable for this
V = FunctionSpace(domain, MixedElement([Ve,Ve,Ve,Ve]))

#displacement and test functions
u = TrialFunction(V)
v = TestFunction(V)

#displacement coefficient trial functions
ubar,uhat,utilde,ubreve=split(u)

#displacement coefficient test functions
vbar,vhat,vtilde,vbreve=split(v)

#geometric dimension
d = 3
#indices
i,j,k,l=indices(4)
a,B = indices(2)
#integration measures
dx = Measure("dx",domain=domain)
ds = Measure("ds",domain=domain)
#spatial coordinate and facet normals
x = SpatialCoordinate(domain)
n = FacetNormal(domain)

#material parameters
E = 100 #70e9
nu = 0.2
_lam = (E*nu)/((1+nu)*(1-2*nu))
mu = E/(2*(1+nu))

#d x d indentity matrix
delta = Identity(d)

# C_ufl = (E/(1+nu))*(0.5*(delta_ik*delta_jl + delta_jk*delta_il) + (nu/(1-2*nu))*delta_ij*delta_kl)
# C = as_tensor((E/(1+nu))*(0.5*(delta[i,k]*delta[j,l] \
#                 + delta[j,k]*delta[i,l]) \
#                     + (nu/(1-(2*nu)))*delta[i,j]*delta[k,l])  ,(i,j,k,l))

C = as_tensor(_lam*(delta[i,j]*delta[k,l]) \
                + mu*(delta[i,k]*delta[j,l]+ delta[i,l]*delta[j,k])  ,(i,j,k,l))

#sub-tensors of stiffness tensor
Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
Ci1kB = as_tensor([[[C[i, 0, k, l] for l in [1,2]]
              for k in range(d)] for i in range(d)])
Ciak1 = as_tensor([[[C[i, j, k, 0] for k in range(d)]
              for j in [1,2]] for i in range(d)])
CiakB = as_tensor([[[[C[i, j, k, l] for l in [1,2]]
              for k in range(d)] for j in [1,2]] 
              for i in range(d)])
     
#partial derivatives of displacement:
ubar_B = grad(ubar)
uhat_B = grad(uhat)
utilde_B = grad(utilde)
ubreve_B = grad(ubreve)

#partial derivatives of shape fxn:
vbar_a = grad(vbar)
vhat_a = grad(vhat)
vtilde_a = grad(vtilde)
vbreve_a = grad(vbreve)

#traction free boundary conditions
Tbar = Ciak1[i,a,k]*uhat[k]*n[a]*vbar[i]*ds \
          + CiakB[i,a,k,B]*ubar_B[k,B]*n[a]*vbar[i]*ds 
That = 2*Ciak1[i,a,k]*utilde[k]*n[a]*vhat[i]*ds \
          + CiakB[i,a,k,B]*uhat_B[k,B]*n[a]*vhat[i]*ds
Ttilde = 3*Ciak1[i,a,k]*ubreve[k]*n[a]*vtilde[i]*ds \
          + CiakB[i,a,k,B]*utilde_B[k,B]*n[a]*vtilde[i]*ds 
Tbreve = CiakB[i,a,k,B]*ubreve_B[k,B]*n[a]*vbreve[i]*ds 

# Tbar = Ciak1[i,a,k]*uhat[k]*n[a]*vbar[i]*ds \
#           + CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*ds 
# That = 2*Ciak1[i,a,k]*utilde[k]*n[a]*vhat[i]*ds \
#           + CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*ds
# Ttilde = 3*Ciak1[i,a,k]*ubreve[k]*n[a]*vtilde[i]*ds \
#           + CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*ds 
# Tbreve = CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*ds 

# equation 1,2,3
L1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
     + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
     - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
     - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \
     # + Tbar

# # equation 4,5,6
L2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
     + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
     - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
     - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \
     # + That

# equation 7,8,9
L3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
     - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
     - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\
     # + Ttilde

#equation 10,11,12
L4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\
     # + Tbreve

Residual = L1+L2+L3+L4

LHS = lhs(Residual)
# RHS = rhs(Residual)
RHS = Constant(domain,0.0)*v[0]*ds

#assemble system matrices
A = assemble_matrix(form(Residual))
A.assemble()

b=create_vector(form(RHS))
with b.localForm() as b_loc:
      b_loc.set(0)
assemble_vector(b,form(RHS))

# import slepc4py,sys
# slepc4py.init(sys.argv)

t1 = time.time()

# Prob = slepc4py.SLEPc.SVD()
# Prob.create()
# Prob.setOperators(A)
# # Prob.setProblemType(slepc4py.SLEPc.SVD.ProblemType.STANDARD)
# # Prob.setType(slepc4py.SLEPc.SVD.ProblemType.LAPACK)
# Prob.setType("lapack")
# # Prob.setTolerances(1E-8, 100)
# # Prob.setWhichSingularTriplets(Prob.Which.SMALLEST)
# Prob.solve()
# #========
Prob = SLEPc.SVD().create()
Prob.setOperators(A)
# Prob.setProblemType(slepc4py.SLEPc.SVD.ProblemType.STANDARD)
# Prob.setType(slepc4py.SLEPc.SVD.ProblemType.LAPACK)
Prob.setType("trlanczos")
Prob.setDimensions(12,200,PETSc.DECIDE)
Prob.setTRLanczosExplicitMatrix(False)
Prob.setTRLanczosOneSide(True)
Prob.setWhichSingularTriplets(Prob.Which.SMALLEST)
Prob.setTolerances(1e-8, 1000)
Prob.setTRLanczosLocking(False)
# Prob.setImplicitTranspose(True)
Prob.setFromOptions()
Prob.solve()     

uvec = PETSc.Vec()
uvec.create(comm=PETSc.COMM_WORLD)
uvec.setSizes(12*(N+1)**2)
uvec.setUp()
# vvec = PETSc.Vec()
# vvec.create(comm=PETSc.COMM_WORLD)

# vvec = PETSc.Vec.create()
print("square of %ix%i" % (N,N))
print(Prob.getConverged())
print(Prob.getIterationNumber())
Prob.getSingularTriplet(0,uvec)
# nconv = Prob.getConverged()
# niters = Prob.getIterationNumber()
# print("Number of eigenvalues successfully computed: ", nconv)
# print("Iterations used", niters)
# #========
t2 = time.time()
print("SLEPc time:")
print(t2-t1)
print()
# nullspace = null_space(Anp)
# print(nullspace.shape)
# print("stiffness matrix:")
# print(Anp)
# print(A.getSize())
# print(b.getSize())
m,n1=A.getSize()
Anp = A.getValues(range(m),range(n1))
Usvd,sv,Vsvd = np.linalg.svd(Anp)

t3 = time.time()

# from scipy.linalg import null_space

# m,n1=A.getSize()
# Anp = A.getValues(range(m),range(n1))
# sols_scipy = null_space(Anp)

t4 = time.time()

# from scipy.sparse.linalg import svds
# from scipy.sparse import csr_matrix

# # m,n1=A.getSize()
# # Anp = A.getValues(range(m),range(n1))
# # Amat =as_backend_type(A.mat()
# # assert isinstance(A, PETSc.Mat)
# # ai,aj,av =A.getValuesCSR()
# # Acsr = csr_matrix((av,ai,aj),A.size)
# Acsr = csr_matrix(A.getValuesCSR()[::-1], shape=A.size)

# sols_sps= svds(Acsr,k=12,which='SM',maxiter=100,solver='lobpcg')
# sols_sps= svds(Acsr,k=12,which='SM',solver='arpack')

t5 = time.time()

m,n1=A.getSize()
Anp = A.getValues(range(m),range(n1))
q,r = np.linalg.qr(Anp,mode='complete')

t6 = time.time()

print('problem setup')
print(t1-t0)
print('slepc solve')
print(t2-t1)
print('numpy solve')
print(t3-t2)
print('scipy nullspace')
print(t4-t3)
print('scipy sparse')
print(t5-t4)
print('numpy qr')
print(t6-t5)

# sols = Vsvd[:,-12:]
# sols = Vsvd[-12:,:].T
sols = q[:,-12:]
# sols = Usvd[:,-12:]
# sols = Usvd[-12:,:].T

# plt.spy(Anp)
# plt.show()

#add in dirichlet BC to prevent rigid body translations?
#add in neumann BC to prevent rigid body rotations?

# sols = q[-12:,:].T
#==================================================#
#======== GET MAPS FROM VERTICES TO DOFS ==========#
#==================================================#
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

#get maps of vertices to displacemnnt coefficients DOFs 
ubar_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(0))
uhat_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(1))
utilde_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(2))
ubreve_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(3))

#==================================================#
#==============DECOUPLE THE SOLUTIONS==============# 
#===(SEPARATE RIGID BODY MODES FROM ELASTIC MODES)=#
#==================================================#

#GET UBAR AND UHAT RELATED MODES FROM SVD
ubar_modes = sols[ubar_vtx_to_dof,:]
uhat_modes = sols[uhat_vtx_to_dof,:]

#INITIALIZE DECOUPLING MATRIX (12X12)
mat = np.zeros((12,12))

#CONSTRUCT FUNCTION FOR UBAR AND UHAT SOLUTIONS GIVEN EACH MODE
UBAR = V.sub(0).collapse()[0]
UHAT = V.sub(1).collapse()[0]
ubar_mode = Function(UBAR)
uhat_mode = Function(UBAR)

#compute area
A = assemble_scalar(form(1.0*dx))

#compute average y and z locations for each cell
yavg = assemble_scalar(form(x[0]*dx))/A
zavg = assemble_scalar(form(x[1]*dx))/A
print(yavg)
print(zavg)

#TODO: fix the 0.5's to be the average x 
#LOOP THROUGH MAT'S COLUMN (EACH MODE IS A COLUMN OF MAT):
for mode in range(mat.shape[1]):
     #construct function from mode
     ubar_mode.vector.array = ubar_modes[:,:,mode].flatten()
     uhat_mode.vector.array = uhat_modes[:,:,mode].flatten()

     #plot ubar modes
     if False:
          tdim = domain.topology.dim
          topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
          grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
          plotter = pyvista.Plotter()
          plotter.add_mesh(grid, show_edges=True,opacity=0.25)
          u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(UBAR)
          u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
          u_grid.point_data["u"] = ubar_mode.x.array.reshape((geometry.shape[0], 3))
          u_grid.set_active_scalars("u")
          plotter.add_mesh(u_grid.warp_by_scalar("u",factor=1), show_edges=True)
          if not pyvista.OFF_SCREEN:
               plotter.show()

     #FIRST THREE ROWS : AVERAGE UBAR_i VALUE FOR THAT MODE
     # mat[0:3,mode]=np.average(np.average(ubar_modes,axis=0),axis=1)
     mat[0,mode]=assemble_scalar(form(ubar_mode[0]*dx))/A
     mat[1,mode]=assemble_scalar(form(ubar_mode[1]*dx))/A
     mat[2,mode]=assemble_scalar(form(ubar_mode[2]*dx))/A
     
     #SECOND THREE ROWS : AVERAGE ROTATION (COMPUTED USING UBAR x Xi, WHERE X1=0, X2,XY=Y,Z)
     mat[3,mode]=assemble_scalar(form(((ubar_mode[2]*(x[0]-yavg)-ubar_mode[1]*(x[1]-zavg))*dx)))
     mat[4,mode]=assemble_scalar(form(((ubar_mode[0]*(x[1]-zavg))*dx)))
     mat[5,mode]=assemble_scalar(form(((-ubar_mode[0]*(x[0]-yavg))*dx)))

     # #CONSTRUCT STRESSES FOR LAST SIX ROWS

     #compute strains at x1=0
     gradubar=grad(ubar_mode)
     eps = as_tensor([[uhat_mode[0],uhat_mode[1],uhat_mode[2]],
                      [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
                      [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
     
     # construct strain and stress tensors based on u_sol
     sigma = as_tensor(C[i,j,k,l]*eps[k,l],(i,j))

     #relevant components of stress tensor
     sigma11 = sigma[0,0]
     sigma12 = sigma[0,1]
     sigma13 = sigma[0,2]

     #integrate stresses over cross-section at "root" of beam and construct xc load vector
     P1 = assemble_scalar(form(sigma11*dx))
     V2 = assemble_scalar(form(sigma12*dx))
     V3 = assemble_scalar(form(sigma13*dx))
     T1 = assemble_scalar(form(((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
     M2 = assemble_scalar(form((x[1]-zavg)*sigma11*dx))
     M3 = assemble_scalar(form(-(x[0]-yavg)*sigma11*dx))


     #THIRD THREE ROWS: AVERAGE FORCE (COMPUTED WITH UBAR AND UHAT)
     mat[6,mode]=P1
     mat[7,mode]=V2
     mat[8,mode]=V3   

     #FOURTH THREE ROWS: AVERAGE MOMENTS (COMPUTED WITH UBAR AND UHAT)
     mat[9,mode]=T1
     mat[10,mode]=M2
     mat[11,mode]=M3

#CONSTRUCT DECOUPLED MODES BY LEFT MULTIPLYING BY THE INVERSE OF "mat"
# sols_decoup = sols@np.linalg.inv(mat)
# from scipy import sparse
# sparse_mat = sparse.csr_matrix(mat)
sols_decoup = sols@np.linalg.inv(mat)
#==================================================#
#============ Define 3d fields  ===================#
#==================================================#

#construct coefficient fields over 2D mesh
U2d = FunctionSpace(domain,Ve)
ubar_field = Function(U2d)
uhat_field = Function(U2d)
utilde_field = Function(U2d)
ubreve_field = Function(U2d)

#==================================================#
#======== LOOP FOR BUILDING LOAD MATRICES =========#
#==================================================#

Cstack = np.vstack((np.zeros((6,6)),np.eye(6)))

from itertools import combinations
idx_ops = [0,1,2,3,4,5]
comb_list = list(combinations(idx_ops,2))
Ccomb = np.zeros((6,len(comb_list)))
for idx,ind in enumerate(comb_list):
     np.put(Ccomb[:,idx],ind,1)
Ccomb = np.vstack((np.zeros_like(Ccomb),Ccomb))

Ctotal = np.hstack((Cstack,Ccomb))

K1 = np.zeros((6,6))
K2 = np.zeros((6,6))

#START LOOP HERE and loop through unit vectors for K1 and diagonal entries of K2
# then combinations of ci=1 where there is more than one nonzero entry
for idx,c in enumerate(Ctotal.T):
     # use right singular vectors (nullspace) to find corresponding
     # coefficient solutions given arbitrary coefficient vector c
     c = np.reshape(c,(-1,1))
     u_coeff=sols_decoup@c

     #use previous dof maps to populate arrays of the individual dofs that depend on the solution coefficients
     ubar_coeff = u_coeff.flatten()[ubar_vtx_to_dof]
     uhat_coeff = u_coeff.flatten()[uhat_vtx_to_dof]
     # utilde_coeff = u_coeff.flatten()[utilde_vtx_to_dof]
     # ubreve_coeff = u_coeff.flatten()[ubreve_vtx_to_dof]

     #populate functions with coefficient function values
     ubar_field.vector.array = ubar_coeff.flatten()
     uhat_field.vector.array = uhat_coeff.flatten()
     # utilde_field.vector.array = utilde_coeff.flatten()
     # ubreve_field.vector.array = ubreve_coeff.flatten()
     
     #compute strains at x1=0
     gradubar=grad(ubar_field)
     eps = as_tensor([[uhat_field[0],uhat_field[1],uhat_field[2]],
                      [gradubar[0,0],gradubar[1,0],gradubar[2,0]],
                      [gradubar[0,1],gradubar[1,1],gradubar[2,1]]])
     
     # construct strain and stress tensors based on u_sol
     sigma = as_tensor(C[i,j,k,l]*eps[k,l],(i,j))

     #relevant components of stress tensor
     sigma11 = sigma[0,0]
     sigma12 = sigma[0,1]
     sigma13 = sigma[0,2]

     #integrate stresses over cross-section at "root" of beam and construct xc load vector
     P1 = assemble_scalar(form(sigma11*dx))
     V2 = assemble_scalar(form(sigma12*dx))
     V3 = assemble_scalar(form(sigma13*dx))
     T1 = assemble_scalar(form(((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
     M2 = assemble_scalar(form((x[1]-zavg)*sigma11*dx))
     M3 = assemble_scalar(form(-(x[0]-yavg)*sigma11*dx))

     #assemble loads into load vector
     P = np.array([P1,V2,V3,T1,M2,M3])

     #compute complementary energy given coefficient vector
     Uc = assemble_scalar(form(sigma[i,j]*eps[i,j]*dx))
     
     if idx<=5:
          K1[:,idx]= P
          K2[idx,idx] = Uc
     else:
          idx1 = comb_list[idx-6][0]
          idx2 = comb_list[idx-6][1]
          Kxx = 0.5 * ( Uc - K2[idx1,idx1] - K2[idx2,idx2])
          K2[idx1,idx2] = Kxx
          K2[idx2,idx1] = Kxx



     # #======= PLOT DISPLACEMENT FROM PERTURBATION =========#
     # tdim = domain.topology.dim
     # topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
     # grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
     # plotter = pyvista.Plotter()
     # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     # u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(UBAR)
     # u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
     # u_grid.point_data["u"] = ubar_field.x.array.reshape((geometry.shape[0], 3))
     # u_grid.set_active_scalars("u")
     # plotter.add_mesh(u_grid.warp_by_scalar("u",factor=1), show_edges=True)
     # # plotter.view_vector((-0.25,-1,0.5))
     # if not pyvista.OFF_SCREEN:
     #      plotter.show()
#compute Flexibility matrix
K1_inv = np.linalg.inv(K1)

S = K1_inv.T@K2@K1_inv
print(S)
print(np.linalg.inv(S))
t1 = time.time()
print(t1-t0)

# #======= PLOT DISPLACEMENT FROM PERTURBATION =========#
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(UBAR)
# ubar_plot = Function(UBAR)
# ubar_modes_decoup = sols_decoup[ubar_vtx_to_dof,:]
# ubar_plot.vector.array = ubar_modes_decoup[:,:,7].flatten()
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# # u_grid.point_data["u"] = ubar_modes_decoup[:,:,7]
# u_grid.point_data["u"] = ubar_plot.x.array.reshape((geometry.shape[0], 3))
# u_grid.set_active_scalars("u")
# plotter.add_mesh(u_grid.warp_by_scalar("u",factor=1), show_edges=True)
# # plotter.view_vector((-0.25,-1,0.5))
# if not pyvista.OFF_SCREEN:
#     plotter.show()
