# static 1D Cantilever Beam with a rectangular cross section
# this example uses Euler-Bernoulli ("classical") beam theory
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import basix
from dolfinx.fem import (Function, FunctionSpace, dirichletbc,
                         locate_dofs_topological,Constant,Expression)
from dolfinx.io import VTKFile
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary,create_interval
from ufl import (SpatialCoordinate,inner, TestFunction, TrialFunction, div, grad, dx,derivative)

########## GEOMETRIC INPUT ####################
E = 70e3
L = 1.0
b = 0.1
h = 0.05
rho= 2.7e-3
g = 9.81
#NOTE: floats must be converted to dolfin constants on domain below

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
NUM_ELEM = 10
domain = create_interval(MPI.COMM_WORLD, NUM_ELEM, [0, L])

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
x = SpatialCoordinate(domain)

thick = Constant(domain,h)
width = Constant(domain,b)
E = Constant(domain,E)
rho = Constant(domain,rho)
g = Constant(domain,g)

A = thick*width
EI = (E*width*thick**3)/12

#distributed load value (due to weight)
q=-rho*A*g

#################################################################
########### EXACT SOLUTIONS FOR VARIOUS BCS #####################
#################################################################
#pinned-pinned
w_pp = q/EI * ( (x[0]**4)/24 -(L*x[0]**3)/12 +((L**3)*x[0])/24 )

#fixed-fixed
w_ff = q/EI * ( (x[0]**4)/24 -(L*x[0]**3)/12 +((L**2)*x[0]**2)/24 )

#fixed-pinned
w_fp = q/EI * ( (x[0]**4)/24 -(5*L*x[0]**3)/48 +((L**2)*x[0]**2)/16)

#cantilever
w_cl = q/EI * ( (x[0]**4)/24 -(L*x[0]**3)/6 +((L**2)*x[0]**2)/4 )
rot_cl = grad(w_cl)
moment_cl = grad(rot_cl)
shear_cl = grad(moment_cl)

# rot_cl = q/EI * ((x[0]**3)/6 -(L*x[0]**2)/2 +((L**2)*x[0])/2 )
#################################################################
########### COMPUTE STATIC SOLUTION #############################
#################################################################

#define Moment expression
def M(u):
    return EI*div(grad(u))

# Create Hermite order 3 on a interval (for more informations see:
#    https://defelement.com/elements/examples/interval-Hermite-3.html )
beam_element = basix.ufl_wrapper.create_element(basix.ElementFamily.Hermite, basix.CellType.interval, 3)

#finite element function space on domain, with trial and test fxns
# W = FunctionSpace(domain,("HER", 3))
W = FunctionSpace(domain,beam_element)
print("Number of DOFs: %d" % W.dofmap.index_map.size_global)
print("Number of elements (intervals): %d" % NUM_ELEM)
print("Number of nodes: %d" % (NUM_ELEM+1))
u_ = TestFunction(W)
v = TrialFunction(W)

#bilinear form (LHS)
k_form = inner(div(grad(v)),M(u_))*dx

#linear form construction (RHS)
l_form = q*u_*dx

#APPLY BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

#locate endpoints
startpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], 0))
endpt=locate_entities_boundary(domain,0,lambda x : np.isclose(x[0], L))

#locate DOFs from endpoints
startdof=locate_dofs_topological(W,0,startpt)
enddof=locate_dofs_topological(W,0,endpt)

#fix displacement of start point and rotation as well
fixed_disp = dirichletbc(ubc,[startdof[0]])
fixed_rot = dirichletbc(ubc,[startdof[1]])

#SOLVE VARIATIONAL PROBLEM
#initialize function in functionspace for beam properties
u = Function(W)

# solve variational problem
problem = LinearProblem(k_form, l_form, u=u, bcs=[fixed_disp,fixed_rot])
uh=problem.solve()
uh.name = "Displacement and Rotation "


#################################################################
########### SAVE AND VISUALIZE RESULTS ##########################
#################################################################

#save output (if uh is directly visualized in Paraview, the plot will look odd,
#  as the rotation DOFs are included in this  )
with VTKFile(domain.comm, "output/output.pvd", "w") as vtk:
    vtk.write([uh._cpp_object])
    # vtk.write([theta._cpp_object])
    # vtk.write([moment._cpp_object])
    # vtk.write([shear._cpp_object])

# print("Maximum displacement: %e" % np.min(uh.x.array))

#NOTE: The solution uh contains both the rotation and the displacement solutions
#The rotation and displacment solutions can be separated as follows:
#TODO: there is likely a much easier way to separate these DOFs and do so in a 

disp = np.empty(0)
rot = np.empty(0)
for i,x in enumerate(uh.x.array):
    if i % 2 != 0:
        rot = np.append(rot,x)
    else:
        disp = np.append(disp,x)

#evaluate derivatives and interpolate to higher order function space
T = FunctionSpace(domain,("CG",1))

#interpolate exact ufl expression onto high-order function space
disp_expr = Expression(w_cl,T.element.interpolation_points())
disp_exact = Function(T)
disp_exact.interpolate(disp_expr)

rot_expr = Expression(rot_cl,T.element.interpolation_points())
rot_exact = Function(T)
rot_exact.interpolate(rot_expr)

mom_expr = Expression(moment_cl,T.element.interpolation_points())
mom_exact = Function(T)
mom_exact.interpolate(mom_expr)

shr_expr = Expression(shear_cl,T.element.interpolation_points())
shr_exact = Function(T)
shr_exact.interpolate(shr_exact)

#extract numpy arrays for plotting
exact_disp = disp_exact.x.array
exact_rot = rot_exact.x.array
exact_mom = mom_exact.x.array
exact_shr = shr_exact.x.array

x_exact = np.linspace(0,1,exact_disp.shape[0])

x_fem = np.linspace(0,1,disp.shape[0])

figure, axis = plt.subplots(2, 2)

print("Maximum magnitude displacement (cantilever exact solution) is: %e" % np.min(exact_disp))
print("Maximum magnitude displacement (cantilever FEM solution) is: %e" % np.min(disp))

####PLOTTING####
#Displacement
axis[0,0].plot(x_exact,exact_disp,label='exact')
axis[0,0].plot(x_fem, disp,label='FEM')
axis[0,0].set_title("Displacement")
axis[0,0].legend()

#Rotation
axis[0,1].plot(x_exact,exact_rot,label='exact')
axis[0,1].plot(x_fem, rot,label='FEM')
axis[0,1].set_title("Rotation")
axis[0,1].legend()

#Moment
axis[1,0].plot(x_exact,exact_mom,label='exact')
# axis[1,0].plot(x_fem, rot,label='FEM')
axis[1,0].set_title("Moment")
axis[1,0].legend()

#Shear
axis[1,1].plot(x_exact,exact_shr,label='exact')
# axis[1,1].plot(x_fem, rot,label='FEM')
axis[1,1].set_title("Shear")
axis[1,1].legend()

plt.show()