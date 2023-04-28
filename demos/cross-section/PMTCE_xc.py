# Cross Sectional Analysis using the Principle of Minimum Total Complementary Energy
# ================

from mpi4py import MPI
import dolfinx.cpp.mesh
from dolfinx import mesh,plot
from dolfinx.fem import Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace,locate_dofs_geometrical
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import (sym,FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
from scipy.linalg import null_space
import matplotlib.pylab as plt

# Create mesh and define function space
N = 3
domain = mesh.create_unit_square(MPI.COMM_WORLD,N,N, mesh.CellType.quadrilateral)
L,W,H = 1,1,1
domain2 = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, W, W])],
                  [N,N,N], cell_type=mesh.CellType.hexahedron)
Ve = VectorElement("CG",domain.ufl_cell(),1,dim=3)
#TODO:check on whether TensorFxnSpace is more suitable for this
V = FunctionSpace(domain, MixedElement([Ve,Ve,Ve,Ve]))

#displacement and test functions
u = TrialFunction(V)
v = TestFunction(V)

#displacement coefficient functions
ubar,uhat,utilde,ubreve=split(u)

#displacement coefficient functions
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
RHS = Constant(domain,0.0)*v[1]*dx

#assemble system matrices
A = assemble_matrix(form(Residual))
A.assemble()
b=create_vector(form(RHS))
with b.localForm() as b_loc:
      b_loc.set(0)
assemble_vector(b,form(RHS))

print(A.getSize())
print(b.getSize())
m,n1=A.getSize()

Anp = A.getValues(range(m),range(n1))
# print("rank:")
# print(np.linalg.matrix_rank(Anp))

# nullspace = null_space(Anp)
# print(nullspace.shape)

# print("stiffness matrix:")
# print(Anp)
Usvd,sv,Vsvd = np.linalg.svd(Anp)
# nullspace = A.getNullSpace()
# print(nullspace.test(A))

sols = Vsvd[:,-12:]
Anp_diag = np.diag(Anp)

# plt.spy(Anp)
# plt.show()

#add in dirichlet BC to prevent rigid body modes?

# use right singular vectors (nullspace) to find corresponding
# coefficient solutions given arbitrary coefficient vector c
c = np.vstack([np.zeros((6,1)),np.ones((6,1))])
u_coeff=sols@c
def get_vtx_to_dofs(domain,V):
     '''
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
#get maps of vertice to displacemnnt coefficients DOFs 
ubar_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(0))
uhat_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(1))
utilde_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(2))
ubreve_vtx_to_dof = get_vtx_to_dofs(domain,V.sub(3))

ubar_coeff = u_coeff.flatten()[ubar_vtx_to_dof]
uhat_coeff = u_coeff.flatten()[uhat_vtx_to_dof]
utilde_coeff = u_coeff.flatten()[utilde_vtx_to_dof]
ubreve_coeff = u_coeff.flatten()[ubreve_vtx_to_dof]

#construct displacement fxn space
Ue = VectorElement("CG",domain.ufl_cell(),1,dim=3)
U = FunctionSpace(domain,Ue)
U.dofmap
u_sol = Function(U)

def get_disp(ubar,uhat,utilde,ubreve,x1):
     '''
     Args:
     x: location along beam axis
     ubar,uhat,utilde,ubreve: coeff. fxns

     returns:
     displacement of nodes in (NumNodes)x3 matrix
     '''
     return ubar+uhat*x1+utilde*x1**2 + ubreve*x1**3

# map each node solution to u_sol given u_nod_coeff 
u_sol.vector.array = get_disp(ubar_coeff,uhat_coeff,utilde_coeff,ubreve_coeff,0).flatten()

# construct stress tensor based on u_sol
# X = SpatialCoordinate(domain2)
sigma = C[i,j,k,l]*grad(u_sol)


#integrate stresses over cross-section and construct xc load vector

#compute K1 from Load vector and arbitrary coefficient vector

#compute K2 by solving matrix equation c^T*K2*c = L^t*K1^-T*K2*K1^-1*L




# # Solution Function
# uh = Function(V)

# # Create Krylov solver
# solver = PETSc.KSP().create(A.getComm())
# solver.setOperators(A)

# # Create vector that spans the null space
# nullspace = PETSc.NullSpace().create(constant=True,comm=MPI.COMM_WORLD)
# A.setNullSpace(nullspace)

# # orthogonalize b with respect to the nullspace ensures that 
# # b does not contain any component in the nullspace
# nullspace.remove(b)

# # Finally we are able to solve our linear system :
# solver.solve(b,uh.vector)


#======= PLOT WARPING FUNCTION =========#
tdim = domain.topology.dim
topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True,opacity=0.25)
u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(U)
u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
print()
u_grid.point_data["u"] = u_sol.x.array.reshape((geometry.shape[0], 3))
u_grid.set_active_scalars("u")
plotter.add_mesh(u_grid.warp_by_scalar("u",factor=.1), show_edges=True)
plotter.view_vector((-0.25,-1,0.5))
if not pyvista.OFF_SCREEN:
    plotter.show()

# # #plot warping
# tdim = domain.topology.dim
# topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
# grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True,opacity=0.25)
# u_topology, u_cell_types, u_geometry = plot.create_vtk_mesh(V)
# u_grid = pyvista.UnstructuredGrid(u_topology, u_cell_types, u_geometry)
# u_grid.point_data["w"] = w.x.array.real
# u_grid.set_active_scalars("w")
# plotter.add_mesh(u_grid.warp_by_scalar("w",factor=2), show_edges=True)
# plotter.view_vector((-0.25,-1,0.5))
if not pyvista.OFF_SCREEN:
    plotter.show()

# #======= PLOT GRAD OF WARPING FUNCTION =========#
# grad_plotter = pyvista.Plotter()
# grad_plotter.add_mesh(grid, style="wireframe", color="k")

# values = np.zeros((geometry.shape[0], 3), dtype=np.float64)
# values[:, :len(grad_u)] = grad_u.x.array.real.reshape((geometry.shape[0], len(grad_u)))

# function_grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
# function_grid["grad_u"] = values
# glyphs = function_grid.glyph(orient="grad_u", factor=1)

# grad_plotter.add_mesh(glyphs)
# grad_plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     grad_plotter.show()