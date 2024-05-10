# # Nonlinear beam model in finite rotations
# In this numerical tour, we show how to formulate and solve a 3D nonlinear beam model in large displacements and  rotations. We however consider here slender structures for which local strains will remain small. We therefore adopt an infinitesimal strain linear elastic model. The main difficulty here is related to the fact that finite rotations cannot be described using a simple rotation vector as in the infinitesimal rotation case but must be handled through rotation matrices. 
#from:
# https://comet-fenics.readthedocs.io/en/latest/demo/finite_rotation_beam/finite_rotation_nonlinear_beam.html
from dolfinx import fem,plot,io,nls,mesh,geometry
import numpy as np
from ufl import sqrt,dot,grad,derivative,Measure,as_vector,diag, Jacobian, MixedElement,TrialFunction,TestFunction,split,VectorElement,pi
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
startpt = [0,0,0]
endpt = [length,0,0]

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
Tlist = np.linspace(0, 1.0, 500)
class MyExpression:
    def __init__(self):
        self.t = 0.0

    def eval(self, x):
        # Added some spatial variation here. Expression is sin(t)*x
        return np.full(x.shape[1],self.t)
    
load_expr = MyExpression()

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
v = fem.Function(V, name="Generalized_displacement")
u, theta = split(v)

VR = fem.TensorFunctionSpace(domain, ("DG",0), shape=(3, 3))
R_old = fem.Function(VR, name="Previous_rotation_matrix")
R_old.interpolate(fem.Expression(fem.Constant(domain,((1., 0., 0.),
                                                      (0., 1., 0.), 
                                                      (0., 0., 1.))),
                                         VR.element.interpolation_points()))

V0 = fem.VectorFunctionSpace(domain, ("DG",0), dim=3)
curv_old = fem.Function(V0, name="Previous_curvature_strain")

Vu,Vu_to_V = V.sub(0).collapse()
total_displ = fem.Function(Vu, name="Previous_total_displacement")

Vr,Vr_to_V = V.sub(1).collapse()
rotation_vector = fem.Function(Vr,name='rotation_vector')

#add DG0 space for updating load step
V1 = fem.FunctionSpace(domain,("DG",0))
load = fem.Function(V1,name='load')

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
gdim = domain.geometry.dim
Jac = as_vector([Jac[i, 0] for i in range(gdim)])
t0 = Jac/sqrt(dot(Jac, Jac))

def tgrad(u):
    return dot(grad(u), t0)

# The strain measures are now defined, depending on the chosen resolution method. 
# We also define the constitutive matrices.
if method == "total":
    defo = dot(R.T, t0 + tgrad(u)) - t0
    curv = dot(H.T, tgrad(theta))
elif method == "incremental":
    R_new = R * R_old
    defo = dot(R_new.T, t0 + tgrad(total_displ + u)) - t0
    curv = curv_old + dot(R_old.T * H.T, tgrad(theta))
    
# C_N = diag(as_vector([ES, GS_2, GS_3]))
# C_M = diag(as_vector([GJ, EI_2, EI_3]))
C = diag(as_vector([ES, GS_2, GS_3,GJ, EI_2, EI_3]))

# We first define a uniform quadrature degree of 4 for integrating the various 
# nonlinear forms.
# We are now in position to define the beam elastic energy as well as the 
# nonlinear residual form expressing balance between the internal and external
# works. The corresponding tangent form is also derived for the Newton-Raphson solver.
metadata = {"quadrature_degree": 4}

def locate_dofs_endpt(x):
    return np.logical_and.reduce((np.isclose(x[0],endpt[0]),
                                    np.isclose(x[1],endpt[1]),
                                    np.isclose(x[2],endpt[2])))

fdim= domain.topology.dim-1
end_pt_node = mesh.locate_entities_boundary(domain,fdim,locate_dofs_endpt)
facet_tag = mesh.meshtags(domain,fdim,end_pt_node,np.full(len(end_pt_node),2,dtype=np.int32))
ds = Measure("ds", domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = Measure("dx", domain=domain, metadata=metadata)

gen_eps = as_vector([defo[0],defo[1],defo[2],curv[0],curv[1],curv[2]])

elastic_energy = 0.5 * (dot(gen_eps, dot(C, gen_eps))) * dx
# elastic_energy = 0.5 * (dot(defo, dot(C_N, defo)) + dot(curv, dot(C_M, curv))) * dx

residual = derivative(elastic_energy, v, v_)
residual += load * (M_max* dot(H, theta_)[1] - F_max * u_[1]) * ds(2)

tangent_form = derivative(residual, v, dv)

# We finish by defining the clamped boundary conditions and the nonlinear newton solver.
ubc = fem.Function(V)
with ubc.vector.localForm() as uloc:
    uloc.set(0.)
pt = startpt
def locate_dofs_startpt(x):
    return np.logical_and.reduce((np.isclose(x[0],pt[0]),
                                    np.isclose(x[1],pt[1]),
                                    np.isclose(x[2],pt[2])))    
start_pt_facets1 = fem.locate_dofs_geometrical((V.sub(0),Vu),locate_dofs_startpt)
start_pt_facets2 = fem.locate_dofs_geometrical((V.sub(1),V.sub(1).collapse()[0]),locate_dofs_startpt)

# start_pt_facets = fem.locate_dofs_geometrical(V,locate_dofs)
bc1 = fem.dirichletbc(ubc, start_pt_facets1,V.sub(0))
bc2 = fem.dirichletbc(ubc, start_pt_facets2,V.sub(1))
bcs = [bc1,bc2]

# problem = fem.petsc.NonlinearProblem(residual, v, [bcs], tangent_form)
problem = fem.petsc.NonlinearProblem(residual, v, bcs,tangent_form)
solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD,problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.atol = 1e-6
solver.report = True
solver.max_it = 1000

# During the load stepping loop, total displacement and rotation vectors will
# be saved to `.xdmf` format every few increments. We also plot the trajectory
# of the extremal point in the $X-Z$ plane. Note that depending on the 
# resolution method, the total displacement is given by $\bu$ for the `total`
# method or by incrementing it with $\bu$ for the `incremental` method. For 
# the latter case, we also update the previous rotation matrix and curvature. 
# Note also that for this approach, a good initial guess is the zero vector,
# rather than the previous increment. We therefore zero the solution vector 
# with `v.vector().zero()` which will be used as an initial guess for the 
# next increment.
uh = np.zeros((len(Tlist), 3))

# store points and cells for evaluating endpoint displacement  
# convert point in array with one element
points_list_array = np.array([endpt, ])
# for each point, compute a colliding cells and append to the lists
points_on_proc = []
cells = []
bbtree = geometry.BoundingBoxTree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions(bbtree, points_list_array)  # get candidates
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_list_array)  # get actual
for _i, point in enumerate(points_list_array):
    if len(colliding_cells.links(_i)) > 0:
        cc = colliding_cells.links(_i)[0]
        points_on_proc.append(point)
        cells.append(cc)
# convert to numpy array
points_on_proc = np.array(points_on_proc)
cells = np.array(cells)

for (i, t) in enumerate(Tlist[1:]):
    #update loadstep
    load_expr.t = t
    load.interpolate(load_expr.eval)

    solver.solve(v)

    if method == "total":
        total_displ.vector.array = v.vector.array[Vu_to_V]
    if method == "incremental":
        total_displ.vector.array += v.vector.array[Vu_to_V]
        R_old.interpolate(fem.Expression(R*R_old,
                                         VR.element.interpolation_points()))
        curv_old.interpolate(fem.Expression(curv,
                                         V0.element.interpolation_points()))
    
    uh[i+1, :] = total_displ.eval(points_on_proc,cells)

    rotation_vector.vector.array = v.vector.array[Vr_to_V]

    if i % 10 == 0:
        p = pyvista.Plotter(window_size=[800, 800])
        topology,cell_types,x = plot.create_vtk_mesh(Vu)
        grid = pyvista.UnstructuredGrid(topology, cell_types, x)
        
        grid.point_data['u'] = total_displ.x.array.reshape((x.shape[0],3))
        actor0 = p.add_mesh(grid, show_edges=True)
        warped = grid.warp_by_vector("u", factor=1)
        actor_1 = p.add_mesh(warped, show_edges=True)

        p.view_isometric()
        p.show_axes()
        p.show()

    if method=='incremental':
        v.vector.array = np.zeros_like(v.vector.array)       

# ## References
# 
# [BAU03] Bauchau, O. A., & Trainelli, L. (2003). The vectorial parameterization of rotation. Nonlinear dynamics, 32(1), 71-92.
# 
# [IBR95] Ibrahimbegović, A., Frey, F., & Kožar, I. (1995). Computational aspects of vector‐like parametrization of three‐dimensional finite rotations. International Journal for Numerical Methods in Engineering, 38(21), 3653-3673.
# 
# [MAS20] Magisano, D., Leonetti, L., Madeo, A., & Garcea, G. (2020). A large rotation finite element analysis of 3D beams by incremental rotation vector and exact strain measure with all the desirable features. Computer Methods in Applied Mechanics and Engineering, 361, 112811.
