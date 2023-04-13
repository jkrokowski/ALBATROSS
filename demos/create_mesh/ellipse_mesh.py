import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI

#mesh parameters
gdim=2
tdim=2
xcname = 'ellipse_xc'
#cross section geometry
H=0.5
L=1.0
#thickness of box wall starting from top and moving CW around XC
t1,t2,t3,t4 = 0.05, 0.025, 0.02, 0.035
print(t4)

gmsh.initialize()
gmsh.model.add(xcname)
gmsh.model.setCurrent(xcname)
lc = 1e-1

#meshtags
r1 = gmsh.model.occ.add_ellipse(0, 0, 0, L, H)
# dx = L-t4-t2
# dy = H-t3-t1
# r2 = gmsh.model.occ.add_rectangle(t4, t3, 0, dx, dy)

# #meshtag for hollow cross section
# hxc = 3
# gmsh.model.occ.cut([(gdim,r1)],[(gdim,r2)],hxc)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

gmsh.model.add_physical_group(tdim,[r1])

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.005)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
filename ="output/" +xcname+".msh"
gmsh.write(filename)

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcname
# cell_markers.name = f"{msh.name}_cells"
# facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
filename = f"output/"+xcname+f".xdmf"
with XDMFFile(msh.comm, filename, "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()