# Cross Sectional Analysis using the Principle of Minimum Total Complementary Energy
# ================

from mpi4py import MPI
import dolfinx.cpp.mesh
from dolfinx import mesh,plot
from dolfinx.mesh import meshtags
from dolfinx.fem import locate_dofs_topological,Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace,locate_dofs_geometrical
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import (sym,FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh

tdim = 2
from dolfinx import geometry
# Create 2d mesh and define function space
N = 3
W = .1
H = .1

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
#read in mesh
xcName = "square_2iso_quads"
fileName = "output/"+ xcName + ".xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#plot mesh:
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

#right
right_marker=0
right_facets = ct.find(right_marker)
right_mt = meshtags(domain, tdim, right_facets, right_marker)
#left
left_marker=1
left_facets = ct.find(left_marker)
left_mt = meshtags(domain, tdim, left_facets, left_marker)

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

# Construct Displacment Coefficient mixed function space
Ve = VectorElement("CG",domain.ufl_cell(),1,dim=3)
#TODO:check on whether TensorFxnSpace is more suitable for this
V = FunctionSpace(domain, MixedElement(4*[Ve]))

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

#integration measures
dx = Measure("dx",domain=domain,subdomain_data=ct)
ds = Measure("ds",domain=domain)
#spatial coordinate and facet normals
x = SpatialCoordinate(domain)
n = FacetNormal(domain)

#material parameters
E1 = 100 #70e9
alpha = .1
E2 = alpha*E1
E_mat = [E1,E2]
nu = 0.2

#material region assignment
region_to_mat = [0,1]

def construct_C(E,nu):   
     _lam = (E*nu)/((1+nu)*(1-2*nu))
     mu = E/(2*(1+nu))
     
     i,j,k,l=indices(4)

     #elasticity tensor construction
     delta = Identity(d)
     C = as_tensor(_lam*(delta[i,j]*delta[k,l]) \
                    + mu*(delta[i,k]*delta[j,l]+ delta[i,l]*delta[j,k])  ,(i,j,k,l))
     return C

C1 = construct_C(E1,nu)
C2 = construct_C(E2,nu)

C_list = [C1,C2]

Residual = 0
for i,C in enumerate(C_list):
     #integration measure
     dx_c = dx(i)
     #indices
     i,j,k,l=indices(4)
     a,B = indices(2)
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

     # equation 1,2,3
     L1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
          + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
          - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
          - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \

     # # equation 4,5,6
     L2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
          + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
          - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
          - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \

     # equation 7,8,9
     L3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
          - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
          - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\

     #equation 10,11,12
     L4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\

     Residual += (L1+L2+L3+L4)

LHS = lhs(Residual)
# RHS = rhs(Residual)
RHS = Constant(domain,0.0)*v[0]*ds

#assemble system matrices
A = assemble_matrix(form(Residual))
A.assemble()
m,n1=A.getSize()
Anp = A.getValues(range(m),range(n1))

#compute modes
Usvd,sv,Vsvd = np.linalg.svd(Anp)
sols = Vsvd[-12:,:].T

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

#LOOP THROUGH MAT'S COLUMN (EACH MODE IS A COLUMN OF MAT):
for mode in range(mat.shape[1]):
     #construct function from mode
     ubar_mode.vector.array = ubar_modes[:,:,mode].flatten()
     uhat_mode.vector.array = uhat_modes[:,:,mode].flatten()

     #FIRST THREE ROWS : AVERAGE UBAR_i VALUE FOR THAT MODE
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
sols_decoup = sols@np.linalg.inv(mat)

#==================================================#
#============ Define solution fields  =============#
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

     #populate functions with coefficient function values
     ubar_field.vector.array = ubar_coeff.flatten()
     uhat_field.vector.array = uhat_coeff.flatten()
    
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
     T1 = assemble_scalar(form(-((x[0]-yavg)*sigma13 - (x[1]-zavg)*sigma12)*dx))
     M2 = assemble_scalar(form(-(x[1]-zavg)*sigma11*dx))
     M3 = assemble_scalar(form((x[0]-yavg)*sigma11*dx))

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

#compute Flexibility matrix
K1_inv = np.linalg.inv(K1)

S = K1_inv.T@K2@K1_inv
# print(S)

K = np.linalg.inv(S)

print(K)