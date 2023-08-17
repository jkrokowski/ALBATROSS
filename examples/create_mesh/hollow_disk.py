import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI
import sys

#mesh parameters
gdim=2
tdim=2

#cross section geometry
R=.1 #m
t = 0.01  #m

gmsh.initialize()
gmsh.model.add("hollow_disk")
gmsh.model.setCurrent("hollow_disk")
lc = 1e-1

#meshtags
d1 = gmsh.model.occ.add_disk(0,0,0,R,R)
d2 = gmsh.model.occ.add_disk(0,0,0,R-t,R-t)

#meshtag for hollow cross section
hd = 3
gmsh.model.occ.cut([(gdim,d1)],[(gdim,d2)],hd)

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

gmsh.model.add_physical_group(tdim,[hd])

# To generate quadrangles instead of triangles, we can simply add
# gmsh.model.mesh.setRecombine(2, hd)

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.0005)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.004)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.generate(gdim)
gmsh.write("output/hollow_disk.msh")

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = 'hollow_disk'
# cell_markers.name = f"{msh.name}_cells"
# facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"output/hollow_disk.xdmf", "w") as file:
    file.write_mesh(msh)

# close gmsh API
gmsh.finalize()

if True:

    import pyvista
    from dolfinx import plot

    #read in xdmf mesh from generation process
    fileName = "output/hollow_disk.xdmf"
    with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
        domain = xdmf.read_mesh(name="hollow_disk")

    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    # plotter.view_vector((0.7,.7,.7))
    # if not pyvista.OFF_SCREEN:
    #      plotter.show()
    #  tdim = domain2.topology.dim
    #  topology, cell_types, geometry = plot.create_vtk_mesh(domain2, tdim)
    #  grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    #  # plotter = pyvista.Plotter()
    #  plotter.add_mesh(grid, show_edges=True,opacity=0.75)
    if not pyvista.OFF_SCREEN:
        plotter.show()
