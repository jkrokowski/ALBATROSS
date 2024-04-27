import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI
import meshio

xcName = "square_2iso"
#mesh parameters
gdim=2
tdim=2

#cross section geometry
H=.1
W=.1
gmsh.initialize()
gmsh.model.add(xcName)
gmsh.model.setCurrent(xcName)
lc = 1e-1

#meshtags
r1 = gmsh.model.occ.add_rectangle(-W/2, -H/2, 0, W/2,H)
r2 = gmsh.model.occ.add_rectangle(0,-H/2,0, W/2, H)
# gmsh.model.occ.fragment([(gdim,r1)],[(gdim,r2),(gdim,r3)])

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

# gmsh.model.add_physical_group(tdim,[r1,r2,r3,r4],5,"full_xc")
gmsh.model.add_physical_group(tdim,[r1],0,"left")
gmsh.model.add_physical_group(tdim,[r2],1,"right")

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.01)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.01)

# # To generate quadrangles instead of triangles, we can simply add
# gmsh.model.mesh.setRecombine(2, r1)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write(xcName + ".msh")

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

# # close gmsh API
# gmsh.finalize()

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

gmsh.finalize()
