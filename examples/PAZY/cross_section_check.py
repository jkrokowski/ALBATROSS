from dolfinx.io import XDMFFile
from dolfinx import plot,mesh
from mpi4py import MPI
import pyvista
import os,sys
import numpy as np

this_file = sys.argv[0]
dirpath = os.path.dirname(this_file)

# xsName = "beam_crosssection_rib_221_quad"
# xsName = "square_2iso_quads"
xsName = "beam_crosssection_2_95_quad"
fileName =  xsName + ".xdmf"
filePath=os.path.join(dirpath,fileName)
print(filePath)

with XDMFFile(MPI.COMM_WORLD, filePath, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain,name="Grid")

domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#plot mesh:
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
print(ct.values)
p.add_mesh(grid, show_edges=True)
p.show_axes()
p.add_axes_at_origin()
p.view_xz()
p.show()