# https://fenicsproject.discourse.group/t/find-cell-tags-from-two-overlapped-meshes-with-different-resolutions/13198/2
from mpi4py import MPI
import ufl
import numpy as np
from petsc4py import PETSc
from dolfinx import fem, io, mesh, geometry, plot, cpp, default_scalar_type
from dolfinx.fem.petsc import assemble_vector,assemble_matrix,create_vector
import basix
import pyvista
import ALBATROSS

m1,n1 = 36,3
m2,n2 = 4,45
m3,n3 = 54,5
m4,n4 = 6,65
H = 1
L = 1
T1 = .1
T2 = .1

mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_0.geometry.x[:, 1] += H/2 - T1/2

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= T2
mesh_1.geometry.x[:, 1] *= H
mesh_1.geometry.x[:, 0] += L/2 - T2/2

mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3)
mesh_2.geometry.x[:, :2] -= .5
mesh_2.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_2.geometry.x[:, 1] -= H/2 - T1/2

mesh_3 = mesh.create_unit_square(MPI.COMM_WORLD, m4, n4)
mesh_3.geometry.x[:, :2] -= .5
mesh_3.geometry.x[:, 0] *= T2
mesh_3.geometry.x[:, 1] *= H
mesh_3.geometry.x[:, 0] -= L/2 - T2/2

#PLOT meshes:
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'   
plotter = pyvista.Plotter()
def add_mesh(msh):
    topology, cell_types, geom = plot.vtk_mesh(msh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
    plotter.add_mesh(grid,show_edges=True,opacity=0.25)
add_mesh(mesh_0)
add_mesh(mesh_1)
add_mesh(mesh_2)
add_mesh(mesh_3)
plotter.show_grid()
plotter.view_xy()
# plotter.show()

meshes= [mesh_0,mesh_1,mesh_2,mesh_3]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

coupled_cross_section = ALBATROSS.cross_section.CoupledXSProblem(XSs)

coupled_cross_section.get_xs_stiffness_matrix()

# ### PLOT SOLUTION
# plotter = pyvista.Plotter()
# plotter.add_text("uh", position="upper_edge", font_size=14, color="black")

# for i,region in enumerate(coupled_lin_elas.regions.values()):
#     pyvista_cells, cell_types, geom = plot.vtk_mesh(region.fxn_space)
#     name = "u"+str(i)
#     grid = pyvista.UnstructuredGrid(pyvista_cells, cell_types, geom)
#     values = np.zeros((geom.shape[0], 3), dtype=np.float64)
#     values[:, :len(region.fxn)] =region.fxn.x.array.real.reshape((geom.shape[0], len(region.fxn)))
#     grid[name] = values
#     warped = grid.warp_by_vector(name, factor=1000)

#     plotting_info={'values':values,
#                                 'grid':grid,
#                                 'warped':warped}
#     region.add_plotting_info(plotting_info)

# max_disp = np.max(np.concatenate([np.linalg.norm(region.plotting['values'],axis=1) for region in coupled_lin_elas.regions.values()]))

# for region in coupled_lin_elas.regions.values():
#     plotter.add_mesh(region.plotting['warped'], show_edges=True,opacity=1,clim=[0,max_disp])

# plotter.show_grid()
# plotter.view_xy()
# plotter.show()