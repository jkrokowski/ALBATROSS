'''
C1 - Hollow Cylinder cross section of isotropic material
'''
import gmsh
from dolfinx.io import XDMFFile,gmshio
from mpi4py import MPI

#mesh parameters
gdim=2
tdim=2

#cross section properties
xcName = "hollow_disk"
R=.1 #m
t = 0.01  #m

gmsh.initialize()
gmsh.model.add(xcName)
gmsh.model.setCurrent(xcName)

#meshtags
markerId = 1
p1 = gmsh.model.geo.add_point(0,0,0)
p2 = gmsh.model.geo.add_point(0,R,0)
p3 = gmsh.model.geo.add_point(0,R-t,0)
p4 = gmsh.model.geo.add_point(0,-R,0)
p5 = gmsh.model.geo.add_point(0,-(R-t),0)

ca1 = gmsh.model.geo.add_circle_arc(p2,p1,p4)
ca2 = gmsh.model.geo.add_circle_arc(p3,p1,p5)
ca3 = gmsh.model.geo.add_circle_arc(p4,p1,p2)
ca4 = gmsh.model.geo.add_circle_arc(p5,p1,p3)
l1 = gmsh.model.geo.add_line(p2,p3)
l2 = gmsh.model.geo.add_line(p4,p5)

edges1 = gmsh.model.geo.addCurveLoop([ca1,l2,-ca2,-l1],-1)
hollow_disk1 = gmsh.model.geo.addPlaneSurface([edges1],-1)
edges2 = gmsh.model.geo.addCurveLoop([ca3,-l2,-ca4,l1],-1)
hollow_disk2 = gmsh.model.geo.addPlaneSurface([edges2],-1)

num_el_circum = 49
num_el_thick = 4
gmsh.model.geo.mesh.setTransfiniteCurve(ca1, int(num_el_circum))
gmsh.model.geo.mesh.setTransfiniteCurve(ca2, int(num_el_circum))
gmsh.model.geo.mesh.setTransfiniteCurve(ca3, int(num_el_circum))
gmsh.model.geo.mesh.setTransfiniteCurve(ca4, int(num_el_circum))
gmsh.model.geo.mesh.setTransfiniteCurve(l1,int(num_el_thick))
gmsh.model.geo.mesh.setTransfiniteCurve(l2,int(num_el_thick))
gmsh.model.geo.mesh.setTransfiniteSurface(hollow_disk1)
gmsh.model.geo.mesh.setTransfiniteSurface(hollow_disk2)

gmsh.model.add_physical_group(tdim,[hollow_disk1,hollow_disk2],0,xcName)

#generate the mesh and optionally write the gmsh mesh file
gmsh.model.geo.mesh.setRecombine(2, hollow_disk1)
gmsh.model.geo.mesh.setRecombine(2, hollow_disk2)
gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(gdim)
gmsh.write(f"output/{xcName}.msh")

# Launch the GUI to see the results:
gmsh.fltk.run()

#use meshio to convert msh file to xdmf
msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF,0,gdim=gdim)
msh.name = xcName
# cell_markers.name = f"{msh.name}_cells"
# facet_markers.name = f"{msh.name}_facets"

#write xdmf mesh file
with XDMFFile(msh.comm, f"output/{xcName}.xdmf", "w") as file:
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
