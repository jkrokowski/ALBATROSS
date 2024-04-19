# # Nonlinear beam model in finite rotations
# In this numerical tour, we show how to formulate and solve a 3D nonlinear beam model in large displacements and  rotations. We however consider here slender structures for which local strains will remain small. We therefore adopt an infinitesimal strain linear elastic model. The main difficulty here is related to the fact that finite rotations cannot be described using a simple rotation vector as in the infinitesimal rotation case but must be handled through rotation matrices. 
#from:
# https://comet-fenics.readthedocs.io/en/latest/demo/finite_rotation_beam/finite_rotation_nonlinear_beam.html
from dolfinx import fem,plot,io
import numpy as np
import matplotlib.pyplot as plt
from ufl import sqrt,dot,grad,derivative,Measure,as_vector,diag, Jacobian, MixedElement,TrialFunction,TestFunction,split,shape,VectorElement,pi
from rotation_parametrization import ExponentialMap
import meshio
from mpi4py import MPI
import pyvista
import os
import sys

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

# Mesh
length = 10.0
N = 40 # number of elements

points = np.zeros((N+1, 3))
points[:, 0] = np.linspace(0, length, N+1)
cells = np.array([[i, i+1] for i in range(N)])
meshio.write(os.path.join(dirpath,"nonlinear_beam.xdmf"), meshio.Mesh(points, {"line": cells}))

fileName =  "nonlinear_beam.xdmf"
filePath=os.path.join(dirpath,fileName)
with io.XDMFFile(MPI.COMM_WORLD,filePath,"r") as infile:
    domain = infile.read_mesh(name='Grid')

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'    
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
p.add_mesh(grid, show_edges=True)
p.show_grid()
# p.view_xy()
p.view_isometric()
p.show_axes()
p.show()

# ubc = dolfinx.fem.Function()
# with ubc.vector.localForm() as uloc:
#      uloc.set(0.)

# fixed_dof_num = 0
# locate_BC = locate_dofs_topological(W,tdim,fixed_dof_num)

# bcs = dirichletbc(ubc,locate_BC)

# We now define the geometric and material properties of the beam cross-section as well as the loading corresponding to those investigated in [[IBR95]](#References). The loading consists of an externally applied concentrated bending moment $\boldsymbol{m}(t)=-M_\text{max}t\be_y$ and concentrated load $\boldsymbol{f}(t)=F_\text{max} t \be_y$ applied in 400 load steps. Finally, we specify the method resolution relying either on the `total` rotation vector or on the `incremental` rotation vector.

# Geometrical properties
radius = fem.Constant(domain,0.2)
S = pi * radius ** 2
I = pi * radius ** 4 / 4

# Stiffness moduli
ES = fem.Constant(domain,1e4)
GS = fem.Constant(domain,1e4)
GS_2 = GS
GS_3 = GS
EI = fem.Constant(domain,1e2)
EI_2 = EI
EI_3 = EI
GJ = fem.Constant(domain,1e2)

# Loading parameters
M_max = fem.Constant(domain,100 * 2* np.pi)
F_max = fem.Constant(domain,50.)
Tlist = np.linspace(0, 1.0, 501)
load = fem.Constant(domain,0)
#TODO: need to interpolate into function for dolfinx
# load = Expression("t", t=0, degree=0)
# class MyExpression:
#     def __init__(self):
#         self.t = 0.0

#     def eval(self, x):
#         # Added some spatial variation here. Expression is sin(t)*x
#         return np.full(x.shape[1], np.sin(self.t)*x[0])

# Resolution method {"total", "incremental"}
method = "incremental"

# We now define a mixed function space consisting of a $P_2$ displacement 
# vector and $P_1$ rotation parameter vector. We will also need a DG-0 
# function space for keeping track of the previous rotation matrix as well
# as the previous curvature strain for implementing the `incremental` approach.
# We also keep track of the total displacement vector.
Ue = VectorElement("CG", domain.ufl_cell(), 2, dim=3)
Te = VectorElement("CG", domain.ufl_cell(), 1, dim=3)
V = fem.FunctionSpace(domain, MixedElement([Ue, Te]))

v_ = TestFunction(V)
u_, theta_ = split(v_)
dv = TrialFunction(V)
v = fem.Function(V, name="Generalized displacement")
u, theta = split(v)

VR = fem.TensorFunctionSpace(domain, ("DG",0), shape=(3, 3))
R_old = fem.Function(VR, name="Previous rotation matrix")
R_old.interpolate(fem.Constant(domain,((1, 0, 0), (0, 1, 0), (0, 0, 1))))

V0 = fem.VectorFunctionSpace(domain, ("DG",0), dim=3)
curv_old = fem.Function(V0, name="Previous curvature strain")


Vu = V.sub(0).collapse()
total_displ = fem.Function(Vu, name="Previous total displacement")

# V1 = fem.FunctionSpace(domain,("DG",0))
# load = fem.Function(V1,name='load')

# We now define the rotation parametrization and the corresponding rotation 
# and curvature matrices obtained from the vector rotation parameter $\btheta$. 
# We then use the mesh `Jacobian` (which we flatten to be of shape=(3,)) to 
# compute the beam axis unit tangent vector `t0` in the reference configuration. 
# We then define the `tgrad` function in order to compute the curvilinear 
# derivative in the beam axis direction.
rot_param = ExponentialMap()
R = rot_param.rotation_matrix(theta)
H = rot_param.curvature_matrix(theta)

Jac = Jacobian(domain)
gdim = domain.geometry().dim()
Jac = as_vector([Jac[i, 0] for i in range(gdim)])
t0 = Jac/sqrt(dot(Jac, Jac))

def tgrad(u):
    return dot(grad(u), t0)


# The strain measures are now defined, depending on the chosen resolution method. We also define the constitutive matrices.
if method == "total":
    defo = dot(R.T, t0 + tgrad(u)) - t0
    curv = dot(H.T, tgrad(theta))
elif method == "incremental":
    R_new = R * R_old
    defo = dot(R_new.T, t0 + tgrad(total_displ + u)) - t0
    curv = curv_old + dot(R_old.T * H.T, tgrad(theta))
    
C_N = diag(as_vector([ES, GS_2, GS_3]))
C_M = diag(as_vector([GJ, EI_2, EI_3]))


# We first define a uniform quadrature degree of 4 for integrating the various nonlinear forms.
# We are now in position to define the beam elastic energy as well as the nonlinear residual form expressing balance between the internal and external works. The corresponding tangent form is also derived for the Newton-Raphson solver.
metadata = {"quadrature_degree": 4}
#TODO: fix subdomain/constraints
ds = Measure("ds", domain=domain, subdomain_data=facets, metadata=metadata)
dx = Measure("dx", domain=domain, metadata=metadata)

elastic_energy = 0.5 * (dot(defo, dot(C_N, defo)) + dot(curv, dot(C_M, curv))) * dx

residual = derivative(elastic_energy, v, v_)
residual += load * (M_max* dot(H, theta_)[1] - F_max * u_[1]) * ds(2)

tangent_form = derivative(residual, v, dv)

#TODO: apply bcs in dolfinx style
# We finish by defining the clamped boundary conditions and the nonlinear newton solver.
bcs = DirichletBC(V, dolfinx.fem.Constant(domain,(0,)*6), left_end)

problem = NonlinearVariationalProblem(residual, v, bcs, tangent_form)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters["newton_solver"]
prm["linear_solver"] = "mumps"
tol = 1e-6
prm["absolute_tolerance"] = tol
prm["relative_tolerance"] = tol


# During the load stepping loop, total displacement and rotation vectors will be saved to `.xdmf` format every few increments. We also plot the trajectory of the extremal point in the $X-Z$ plane. Note that depending on the resolution method, the total displacement is given by $\bu$ for the `total` method or by incrementing it with $\bu$ for the `incremental` method. For the latter case, we also update the previous rotation matrix and curvature. Note also that for this approach, a good initial guess is the zero vector, rather than the previous increment. We therefore zero the solution vector with `v.vector().zero()` which will be used as an initial guess for the next increment.
uh = np.zeros((len(Tlist), 3))

out_file = io.XDMFFile("helical_beam.xdmf")
out_file.parameters["functions_share_mesh"] = True
out_file.parameters["flush_output"] = True

for (i, t) in enumerate(Tlist[1:]):
    load.value = t

    solver.solve()
    displ = v.sub(0, True)

    if method == "total":
        total_displ.vector()[:] = displ.vector()[:]
    if method == "incremental":
        total_displ.vector()[:] += displ.vector()[:]
        #TODO: fix project/assign
        R_old.assign(project(R * R_old, VR))
        curv_old.assign(project(curv, V0))
        v.vector().zero()

    uh[i+1, :] = total_displ((length, 0, 0))

    rotation_vector = v.sub(1, True)
    rotation_vector.rename("Rotation vector", "")   

    #TODO: fix the output format
    if i % 10 == 0:
        out_file.write(rotation_vector, t)
        out_file.write(total_displ, t)

        plt.plot(length+uh[:i+2, 0], uh[:i+2, 2], linewidth=1)
        plt.xlim(-length/2, length)
        plt.gca().set_aspect("equal")
        plt.show()
        
out_file.close()


# ## References
# 
# [BAU03] Bauchau, O. A., & Trainelli, L. (2003). The vectorial parameterization of rotation. Nonlinear dynamics, 32(1), 71-92.
# 
# [IBR95] Ibrahimbegović, A., Frey, F., & Kožar, I. (1995). Computational aspects of vector‐like parametrization of three‐dimensional finite rotations. International Journal for Numerical Methods in Engineering, 38(21), 3653-3673.
# 
# [MAS20] Magisano, D., Leonetti, L., Madeo, A., & Garcea, G. (2020). A large rotation finite element analysis of 3D beams by incremental rotation vector and exact strain measure with all the desirable features. Computer Methods in Applied Mechanics and Engineering, 361, 112811.
