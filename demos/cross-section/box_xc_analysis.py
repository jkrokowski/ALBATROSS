from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar,assemble,form,assemble_vector,FunctionSpace,Function
from mpi4py import MPI
from dolfinx.mesh import meshtags
from ufl import Measure,SpatialCoordinate,Constant
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
#plot the elastic modulus for each region
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = E.x.array
p.add_mesh(grid, show_edges=True)
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

