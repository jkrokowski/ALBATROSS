from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar,assemble,form,assemble_vector,FunctionSpace,Function
from mpi4py import MPI
from dolfinx.mesh import meshtags
from ufl import Measure,SpatialCoordinate,Constant,as_matrix
# import matplotlib.pyplot as plt
import pyvista
from dolfinx.plot import create_vtk_mesh
import numpy as np

#read in mesh
xcName = "Box_XC_hetero"
fileName = "output/"+ xcName + ".xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#mesh XC topological dimension
tdim=2

#top
top_marker=0
top_facets = ct.find(top_marker)
top_mt = meshtags(domain, tdim, top_facets, top_marker)
#right
right_marker=1
right_facets = ct.find(right_marker)
right_mt = meshtags(domain, tdim, right_facets, right_marker)
#bottom
bottom_marker=2
bottom_facets = ct.find(bottom_marker)
bottom_mt = meshtags(domain, tdim, bottom_facets, bottom_marker)
#left
left_marker=3
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

#material data value
pu = 0 #Polyurethane
al = 1 #Aluminum
mat =[pu,al]
E_mat = [0.7e9,70.0e9] #GPa
nu_mat = [0.308,0.35] #no units req'd
rho_mat = [425,2700] #kg/m^3

#material region assignment
region_to_mat = [1,0,0,1]

#construct a DGO function space to assign material properties:
Q = FunctionSpace(domain,("DG",0))
material_tags = np.unique(ct.values)

E = Function(Q)
nu = Function(Q)
rho = Function(Q)

for tag in material_tags:
    cells = ct.find(tag)
    E.x.array[cells] = np.full_like(cells,E_mat[region_to_mat[tag]],dtype=float)
    nu.x.array[cells] = np.full_like(cells,nu_mat[region_to_mat[tag]],dtype=float)
    rho.x.array[cells] = np.full_like(cells,rho_mat[region_to_mat[tag]],dtype=float)

#compute shear modulus:
G = E /(2*(1+nu))


#plot the elastic modulus for each region
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = E.x.array
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

#confirm that we have an rectangular mesh in 2D space:
print('cell_type   :', domain.ufl_cell())

coordinates = domain.geometry.x

htop = max(coordinates[:,0])
hbot = min(coordinates[:,0])
wleft = min(coordinates[:,1])
wright = max(coordinates[:,1])

#initialize integration measures
dx = Measure("dx",domain=domain)
dx_top = Measure("dx",domain=domain,subdomain_data=top_mt,subdomain_id=top_marker)
dx_right = Measure("dx",domain=domain,subdomain_data=right_mt,subdomain_id=right_marker)
dx_bottom = Measure("dx",domain=domain,subdomain_data=bottom_mt,subdomain_id=bottom_marker)
dx_left = Measure("dx",domain=domain,subdomain_data=left_mt,subdomain_id=left_marker)

x = SpatialCoordinate(domain)

print(hbot)
print(wleft)
print(htop)
print(wright)

rho = 2700 #kg/m^3 for Aluminum

#cross section geometry
H=.25
L=1.0
#thickness of box wall starting from top and moving CW around XC
t1,t2,t3,t4 = 0.05, 0.025, 0.02, 0.035

#compute total area:
A = assemble_scalar(form(1.0*dx))
Ao =L*H
Ai =  (H-t1-t3)*(L-t2-t4)
A_exact =  Ao - Ai
print("Exact Area: %.6e" % A_exact)
print("Numerical Area: %.6e" % A)

#compute areas of subdomains:
A_top_exact = L*t1
A_top = assemble_scalar(form(1.0*dx_top))
print("Exact Top Area: %.6e" % A_top_exact)
print("Numerical Top Area: %.6e" % A_top)

A_right_exact = (H-t1-t3)*t2
A_right = assemble_scalar(form(1.0*dx_right))
print("Exact Right Area: %.6e" % A_right_exact)
print("Numerical Right Area: %.6e" % A_right)

A_bottom_exact = L*t3
A_bottom = assemble_scalar(form(1.0*dx_bottom))
print("Exact bottom Area: %.6e" % A_bottom_exact)
print("Numerical bottom Area: %.6e" % A_bottom)

A_left_exact = (H-t1-t3)*t4
A_left = assemble_scalar(form(1.0*dx_left))
print("Exact left Area: %.6e" % A_left_exact)
print("Numerical left Area: %.6e" % A_left)

x_Gi = ((L-t2-t4)/2 + t4)
y_Gi = ((H-t1-t3)/2 + t3)
x_G_exact = ( Ao*(L/2) - Ai*x_Gi ) / A
y_G_exact = ( Ao*(H/2) - Ai*y_Gi )/ A

x_G = assemble_scalar(form(x[0]*dx)) / A
y_G = assemble_scalar(form(x[1]*dx)) / A

print("Exact location of neutral axis in x: %.4e" % x_G_exact)
print("Exact location of neutral axis in y: %.4e" % y_G_exact)
print("Numerical location of neutral axis in x: %.4e" % x_G)
print("Numerical location of neutral axis in y: %.4e" % y_G)

Ix_exact = ( (L*H**3)/12 + Ao*(H/2 - y_G_exact)**2)  - (((H-t1-t3)**3*(L-t2-t4))/12 +Ai*(y_Gi-y_G_exact)**2) 
Iy_exact = ( (L**3*H )/12 + Ao*(L/2 - x_G_exact)**2) - (((H-t1-t3)*(L-t2-t4)**3)/12 +Ai*(x_Gi-x_G_exact)**2) 
Ixy_exact = ( Ao*(L/2 - x_G_exact)*(H/2 - y_G_exact)) - ( Ai*(x_Gi- x_G_exact)*(y_Gi- y_G_exact))


Ix= assemble_scalar(form(((x[1]-y_G)**2)*dx))
Iy = assemble_scalar(form(((x[0]-x_G)**2)*dx))
Ixy = assemble_scalar(form(((x[0]-x_G)*(x[1]-y_G))*dx))

print("Exact Ix: %.6e" % Ix_exact)
print("Numerical Ix: %.6e" % Ix)
print("Exact Iy: %.6e" % Iy_exact)
print("Numerical Iy: %.6e" % Iy)
print("Exact Ixy: %.6e" % Ixy_exact)
print("Numerical Ixy: %.6e" % Ixy)

#=======CONSTRUCTING TIMOSHENKO STIFFNESS MATRIX=========#
'''The Timoshenko 6x6 constitutive/stiffness matrix is 
    symmetric and takes the following form:
    K_{i,j} = { [K11,K12,K13,K14,K15,K16],
                [K21,K22,K23,K24,K25,K26],
                [K31,K32,K33,K34,K35,K36],
                [K41,K42,K43,K44,K45,K46],
                [K51,K52,K53,K54,K55,K56],
                [K61,K62,K63,K64,K65,K66]]

    where sigma is the vector of "generalized stresses":
        sigma = [Sn,S1,S2,St,Sm1,Sm2].T
    and eps is the vector of "generalized strains":
        eps = [en,e1,e2,et,em1,em2].T


    sigma = inner(K,eps)

    some a fully populated matrix indicates couplings all the permutations of
    different couplings between axial,bending,torsion and shear loadings.

    We will neglect the following couplings in our model 
    (as these involved understanding the warping shape function \omega):
        -shear <--> axial (K12=K21=K13=K31=0)
        -shear <--> shear (K23=K32=0)
        -shear <--> bending (K25=K26=K35=K36=K52=K62=K53=K63=0)
        -torsion <--> axial (K14=K41=0)
        -torsion <--> bending (K45,K46,K54,K64=0)

        *note: torsion also involves understanding this shape function, 
                but we can simplify and assume St Venant torsion to start    

    See this for more information:
    https://comet-fenics.readthedocs.io/en/latest/demo/beams_3D/beams_3D.html

    
    To compute these matrix entries, we need to first compute the location of
    three points in the cross section:
        -the center of mass
        -the elastic/tension center (like centroid, but instead of density weighted, it is elastic modulus weighted)
        -the shear center (more complicated, but would be important to know)
                '''

A = assemble_scalar(form(1.0*dx))
rhoA = assemble_scalar(form(rho*dx))
EA = assemble_scalar(form(E*dx))
Sx = assemble_scalar(form(E*x[0]*dx))
Sy = assemble_scalar(form(E*x[1]*dx))

#mass weighted centroid location:
x0 = assemble_scalar(form(rho*x[0]*dx)) / rhoA
y0 = assemble_scalar(form(rho*x[1]*dx)) / rhoA
print(x0)
print(y0)

#tension center location:
xTC = Sx / EA
yTC = Sy / EA
#elastic center location(at same location as tension center):
xEA = xTC
yEA = yTC

print(xTC)
print(yTC)
print(xEA)
print(yEA)

#plot the elastic modulus for each region and the location of some important points
points = np.array([[x0,y0,0],
                   [xTC,yTC,0]])
labels = ['o','x']
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = E.x.array
p.add_mesh(grid, show_edges=True)
p.add_point_labels(points,labels)
p.view_xy()
p.show()

#shear center is much more complicated and requires shear flow analysis/warping function computation

#axial stiffness
K11 = EA
#shear stiffness
kappa = 0.1 #estimate (need to follow this method when
        #   Real elements are implemented, as this allows for the computation of 
        #  warping function through the cross-section): 
        # https://comet-fenics.readthedocs.io/en/latest/demo/cross_section_analysis/cross_section_analysis.html#References 
K22 = kappa*assemble_scalar(form(G*dx))
K33 = kappa*assemble_scalar(form(G*dx))
#bending stiffness:
K55= assemble_scalar(form(((E*x[1]-yEA)**2)*dx))
K66 = assemble_scalar(form((E*(x[0]-xEA)**2)*dx))
#bending-bending coupling stiffness
K56 = assemble_scalar(form((E*(x[0]-xEA)*(x[1]-yEA))*dx))
K65 = K56
#axial-bending coupling stiffness
K15 = assemble_scalar(form((E*(x[1]-yEA))*dx))
K51 = K15
K16 = assemble_scalar(form((E*(x[0]-xEA))*dx))
K61 = K16
#shear-torsional coupling:
K24 = Sx
K34 = Sy
K42 = K24
K43 = K34
#torsional stiffness
K44 = K55+K66 #has additional terms if warping on a XC is accounted for
# #torsional-bending coupling
# K45 = assemble_scalar(form(()*dx))
# K54 = K45
# K46 = assemble_scalar(form(()*dx))
# K64 = K46

#assumed uncoupled values:
K12,K21,K13,K31=4*[0] #no shear/axial coupling
K23,K32=2*[0] #no shear/shear coupling
K25,K26,K35,K36,K52,K62,K53,K63=8*[0] #no shear/bending coupling
# K24,K34,K42,K43 =4*[0] #no shear/torsion coupling
K14,K41=2*[0] #no axial/torsion coupling
K45,K46,K54,K64=4*[0] #no torsional/bending coupling


K = as_matrix(((K11,K12,K13,K14,K15,K16),
              (K21,K22,K23,K24,K25,K26),
              (K31,K32,K33,K34,K35,K36),
              (K41,K42,K43,K44,K45,K46),
              (K51,K52,K53,K54,K55,K56),
              (K61,K62,K63,K64,K65,K66)))

print(K)

#mass matrix is diagonal? 
#mass matrix should just be density weighted values?