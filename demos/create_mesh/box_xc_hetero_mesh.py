import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI
import meshio

xcName = "Box_XC_hetero"
#mesh parameters
gdim=2
tdim=2

#cross section geometry
H=.25
L=1.0
#thickness of box wall starting from top and moving CW around XC
t1,t2,t3,t4 = 0.05, 0.025, 0.02, 0.035
print(t4)

gmsh.initialize()
gmsh.model.add(xcName)
gmsh.model.setCurrent(xcName)
lc = 1e-1

#meshtags
r1 = gmsh.model.occ.add_rectangle(0, H-t1, 0, L, t1)
r2 = gmsh.model.occ.add_rectangle(L-t2, t3,0, t2, H-t1-t3)
r3 = gmsh.model.occ.add_rectangle(0, 0, 0, L, t3)
r4 = gmsh.model.occ.add_rectangle(0,t3, 0, t4, H-t1-t3)
gmsh.model.occ.fragment([(gdim,r1)],[(gdim,r2),(gdim,r3)])

#meshtag for hollow cross section
# hxc = 5
# box_xc,_ = gmsh.model.occ.fuse([(gdim,r1)],[(gdim,r2)])
# box_xc,_ = gmsh.model.occ.fuse([(gdim,r1)],[(gdim,r2)])#,(gdim,r2),(gdim,r3)])
# print(box_xc[0])# gmsh.model.occ.cut([(gdim,r1)],[(gdim,r2)],hxc)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# gmsh.model.add_physical_group(tdim,[r1,r2,r3,r4],5,"full_xc")
gmsh.model.add_physical_group(tdim,[r1],0,"top")
gmsh.model.add_physical_group(tdim,[r2],1,"right")
gmsh.model.add_physical_group(tdim,[r3],2,"bottom")
gmsh.model.add_physical_group(tdim,[r4],3,"left")

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.005)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write(xcName + ".msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

# close gmsh API
gmsh.finalize()

# #include physical groups in .xdmf
# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:,:2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
#     return out_mesh



# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("cell_tags", cell_type)
#     points = mesh.points[:,:2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points,
#                            cells={cell_type: cells},
#                            cell_data={"name_to_read":[cell_data]})
#     return out_mesh

# if MPI.COMM_WORLD.rank == 0:
#     # Read in mesh
#     msh = meshio.read(xcName + ".msh")
   
#     # Create and save one file for the mesh, and one file for the facets 
#     triangle_mesh = create_mesh(msh, "triangle", prune_z=True)
#     # line_mesh = create_mesh(msh, "line", prune_z=True)
#     meshio.write(xcName+".xdmf", triangle_mesh)
#     # meshio.write(xcName+.xdmf", line_mesh)
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

if MPI.COMM_WORLD.rank == 0:
    # Read in mesh
    msh = meshio.read(xcName + ".msh")

    # Create and save one file for the mesh, and one file for the facets 
    mesh = create_mesh(msh, "triangle", prune_z=True)
    print(mesh.cells)
    print(mesh.cell_data)
    meshio.write(f"output/"+xcName+".xdmf", mesh)
# #write xdmf mesh file
# with XDMFFile(msh.comm, f"output/"+xcName+".xdmf", "w") as file:
#     file.write_mesh(msh)
#     file.write_mesh(cell_markers)