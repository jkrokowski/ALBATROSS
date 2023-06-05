import gmsh
import numpy as np
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI

#mesh parameters
gdim=2
tdim=2

#cross section geometry
R=.1 #m
t = 0.01  #m

gmsh.initialize()
gmsh.model.add("hollow_disk")
gmsh.model.setCurrent("hollow_disk")
# lc = 1e-4

#meshtags
c1 = gmsh.model.occ.add_circle(0,0,0,R)
cc1 = gmsh.model.occ.addCurveLoop([c1])
d1 = gmsh.model.occ.add_plane_surface([cc1])
c2 = gmsh.model.occ.add_circle(0,0,0,R-t)
cc2 = gmsh.model.occ.addCurveLoop([c2])
d2 = gmsh.model.occ.add_plane_surface([cc2])
p1 = gmsh.model.occ.add_point(R-t,0,0)
p2 = gmsh.model.occ.add_point(R,0,0)
l1 = gmsh.model.occ.add_line(p1,p2)

#meshtag for hollow cross section
hd = 30
gmsh.model.occ.cut([(gdim,d1)],[(gdim,d2)],hd)

# gmsh.model.geo.mesh.setTransfiniteCurve(cc1, 100)
# gmsh.model.geo.mesh.setTransfiniteCurve(cc2, 50)
# gmsh.model.geo.mesh.setTransfiniteCurve(l1,4)
# gmsh.model.geo.mesh.setTransfiniteSurface(hd)

# for curve in gmsh.model.occ.getEntities(1):
#     gmsh.model.mesh.setTransfiniteCurve(curve[1], 50)
# gmsh.model.geo.mesh.setTransfiniteSurface(1, "Left", [1, 2, 3, 4])

# Synchronize OpenCascade representation with gmsh model
gmsh.model.occ.synchronize()

gmsh.model.add_physical_group(tdim,[hd])

# To generate quadrangles instead of triangles, we can simply add
gmsh.model.mesh.setRecombine(tdim, hd)

#adjust mesh size parameters
gmsh.option.setNumber('Mesh.MeshSizeMin', 0.0005)
gmsh.option.setNumber('Mesh.MeshSizeMax', 0.004)

# field = gmsh.model.mesh.field
# field.add("MathEval", 100)
# field.setString(100, "F", "x+y")
# field.setAsBackgroundMesh(1)

gmsh.option.setNumber("Mesh.Algorithm", 8)
gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.mesh.setAlgorithm(2,hd,8)
gmsh.model.mesh.generate(gdim)
gmsh.write("output/hollow_disk.msh")

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
