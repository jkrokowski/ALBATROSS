# static 1D Beam in 3D space example based on Jeremy Bleyer's implementation here:
# https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html
# 
# The beam cross-section in this example is:
#   -homogenous
#   -isotropic
#   -rectangular
#   -constant along the beam axis
#   
# this example consists of a kinked beam under two classes of loading:
#   -distributed loads:
#       -gravity
#       -centrifugal
#   -point loads:
#       -point forces (applied at a node)
#       -torques/moments (applied at a node)

from dolfinx.fem import (VectorFunctionSpace,Function,FunctionSpace,
                        dirichletbc,locate_dofs_geometrical,
                        locate_dofs_topological,Constant,
                        form)
from dolfinx.io import XDMFFile,gmshio,VTKFile
from dolfinx.fem.petsc import (LinearProblem,assemble_matrix,assemble_vector, 
                                apply_lifting,set_bc,create_vector)
from ufl import (Jacobian, diag, as_vector, inner, sqrt,cross,dot,
                VectorElement, TestFunction, TrialFunction,split,grad,Measure,dx)
import gmsh
from mpi4py import MPI
import numpy as np
import pyvista
from dolfinx import plot
from petsc4py import PETSc

plot_with_pyvista = True

DOLFIN_EPS = 3E-16

#################################################################
########### CONSTRUCT BEAM MESH #################################
#################################################################
gmsh.initialize()

# model and mesh parameters
gdim = 3
tdim = 1
R = 10.0
lc = 1e-1 #TODO: find if there is an easier way to set the number of elements

#construct line in 3D space
gmsh.model.add("Beam")
gmsh.model.setCurrent("Beam")
p1 = gmsh.model.occ.addPoint(0,0,0)
p2 = gmsh.model.occ.addPoint(0.2*R, 0.05*R, 0.1*R)
p3 = gmsh.model.occ.addPoint(R, -0.025*R, -0.05*R)
line1 = gmsh.model.occ.addLine(p1,p2)
line2 = gmsh.model.occ.addLine(p2,p3)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# add physical marker
gmsh.model.add_physical_group(tdim,[line1,line2])
# gmsh.model.add_physical_group(0,[p2])

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.0005*R)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.005*R)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write("output/beam_mesh.msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
msh.name = 'beam_mesh'
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"output/beam_mesh.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

#read in xdmf mesh from generation process
fileName = "output/beam_mesh.xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    domain = xdmf.read_mesh(name="beam_mesh")

#confirm that we have an interval mesh in 3D:
print('cell_type   :', domain.ufl_cell())

#################################################################
##### ENTER MATERIAL PARAMETERS AND CONSTITUTIVE MODEL ##########
#################################################################
thick = Constant(domain,0.3)
width = thick/3
E = Constant(domain,70e3)
nu = Constant(domain,0.3)
G = E/2/(1+nu)
rho = Constant(domain,2.7e-3)
g = Constant(domain,9.81)

#rotor parameters
r = Constant(domain,R)
def RPM_to_Hz(rpm):
    return rpm/60
omega = Constant(domain,RPM_to_Hz(1000))

S = thick*width
ES = E*S
EI1 = E*width*thick**3/12
EI2 = E*width**3*thick/12
GJ = G*0.26*thick*width**3
kappa = Constant(domain,5./6.)
GS1 = kappa*G*S
GS2 = kappa*G*S

#construct constitutive matrix
Q = diag(as_vector([ES, GS1, GS2, GJ, EI1, EI2]))

#################################################################
########### DEFINE AND CONSTRUCT VARIATIONAL FORM ###############
#################################################################

# Compute transformation Jacobian between reference interval and elements
def tangent(domain):
    t = Jacobian(domain)
    return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))

t = tangent(domain)

#compute section local axis
ez = as_vector([0, 0, 1])
a1 = cross(t, ez)
a1 /= sqrt(dot(a1, a1))
a2 = cross(t, a1)
a2 /= sqrt(dot(a2, a2))

#construct mixed element function space
Ue = VectorElement("CG", domain.ufl_cell(), 1, dim=3)
W = FunctionSpace(domain, Ue*Ue)

u = TrialFunction(W)
du = TestFunction(W)
(w, theta) = split(u)
(dw, dtheta) = split(du)

def tgrad(u_):
    return dot(grad(u_), t)
def generalized_strains(u_):
    (w_, theta_) = split(u_)
    return as_vector([dot(tgrad(w_), t),
                      dot(tgrad(w_), a1)-dot(theta_, a2),
                      dot(tgrad(w_), a2)+dot(theta_, a1),
                      dot(tgrad(theta_), t),
                      dot(tgrad(theta_), a1),
                      dot(tgrad(theta_), a2)])
def generalized_stresses(u_):
    return dot(Q, generalized_strains(u_))

Sig = generalized_stresses(du)
Eps =  generalized_strains(u)

#modify quadrature scheme for shear components
dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})

#### LHS construction (bilinear form) ####
a_form = sum([Sig[i]*Eps[i]*dx for i in [0, 3, 4, 5]]) + (Sig[1]*Eps[1]+Sig[2]*Eps[2])*dx_shear

#### RHS construction (linear form) ####
#weight per unit length
q = rho*S*g
gravity =-q*w[2]*dx

#centrifugal force per unit length
cf = rho*S*omega**2
centrifugal = cf*w[0]*dx

# gravity + centrifugal + fh_force
L_form = gravity + centrifugal 

#DEFINE BOUNDARY CONDITIONS
#initialize function for boundary condition application
ubc = Function(W)
with ubc.vector.localForm() as uloc:
     uloc.set(0.)

fixed_dof_num = 0
locate_BC = locate_dofs_topological(W,tdim,fixed_dof_num)

bcs = dirichletbc(ubc,locate_BC)

#################################################################
########### CONSTRUCT AND SOLVE LINEAR SYSTEM ###################
#################################################################
#If no point loads or point moments are required, the following
# two lines can be used to solve the system in a more compact manner:
# problem = LinearProblem(a_form, L_form, bcs=[bcs])
# uh=problem.solve()

#ASSEMBLE THE LINEAR SYSTEM
#assemble LHS
A = assemble_matrix(form(a_form), bcs=[bcs])
A.assemble()

#assemble RHS
b=create_vector(form(L_form))
with b.localForm() as b_loc:
            b_loc.set(0)
assemble_vector(b,form(L_form))

# APPLY dirchelet bc: these steps are directly pulled from the 
# petsc.py LinearProblem().solve() method
a_form = form(a_form)
apply_lifting(b,[a_form],bcs=[[bcs]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b,[bcs])

#locate disp and rotation dofs (collapsed subspace zero is used to locate 
#   dofs to apply the nodal force):
W0, disp_dofs = W.sub(0).collapse()
W1, rot_dofs = W.sub(1).collapse()

#locate dofs for point forces applied to nodes
px = 0.2*R
py = 0.05*R
pz = 0.1*R
def flap_hinge_pt_mark(x):
    x_loc = np.isclose(abs(x[0]-px), DOLFIN_EPS)
    y_loc = np.isclose(abs(x[1]-py), DOLFIN_EPS)
    z_loc = np.isclose(abs(x[2]-pz), DOLFIN_EPS)
    return  x_loc & y_loc & z_loc
# this requires the collapsing the supspace and using it to locate the dofs
#  geometrically via the map to the parent space, see:
# https://fenicsproject.discourse.group/t/dolfinx-dirichlet-bcs-for-mixed-function-spaces/7844/2
fh_force_dofs = locate_dofs_geometrical((W.sub(0),W0), flap_hinge_pt_mark)

fh_mom_dofs = locate_dofs_geometrical((W.sub(1),W1), flap_hinge_pt_mark)

# #apply point force by modified RHS vector at relevant dofs
# f_fh = [-.25,-.25,-.25]
# b.array[fh_force_dofs[0]] = f_fh

#apply point moment by modifying RHS vector at relevant dofs
# m_fh = [-.25,-.25,-.25]
# b.array[fh_mom_dofs[0]] = m_fh

# Solve with PETSc Krylov solver
uh_ptld = Function(W)
uvec = uh_ptld.vector
uvec.setUp()
ksp = PETSc.KSP().create()
ksp.setType(PETSc.KSP.Type.CG)
ksp.setTolerances(rtol=1e-15)
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b,uvec)

#save PETsC displacment solution vector to dolfinx function
# this requires separating the displacement solution from the 
# rotation solution (which is done with the previously save dof maps)
uh = Function(W)
uh.sub(0).vector.array[disp_dofs]= uvec[disp_dofs]
uh.sub(1).vector.array[rot_dofs] = uvec[rot_dofs]
#separate the displacement solution and the rotation solution
wh = uh.sub(0)
wh.name = "Displacement"
thetah = uh.sub(1)
thetah.name = "Rotation"


#################################################################
########### POST PROCESSING #####################################
#################################################################

#save compute moment field for Paraview visualization
V1 = VectorFunctionSpace(domain, ("CG", 1), dim=2)
M = Function(V1, name="Bending moments (M1,M2)")
Sig = generalized_stresses(uh)

"""
    Project function does not work the same between legacy FEniCS and FEniCSx,
    so the following project function must be defined based on this forum post:
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-function-dolfinx/4142/6
    and inspired by Ru Xiang's Shell module project function
    https://github.com/RuruX/shell_analysis_fenicsx/blob/b842670f4e7fbdd6528090fc6061e300a74bf892/shell_analysis_fenicsx/utils.py#L22
    """

def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = inner(Pv, w) * dx
    L = inner(v, w) * dx

    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)
project(as_vector([Sig[4], Sig[5]]), M)
print(M.vector.array)
M.name = "Bending Moments (M1,M2)"

with XDMFFile(MPI.COMM_WORLD, "output/rotor_flap.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(wh)
    xdmf.write_function(thetah)
    xdmf.write_function(M)

print(wh.vector.array)