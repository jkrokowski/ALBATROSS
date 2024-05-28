#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS
import numpy as np

#cross-section mesh definition
N = 20 #number of quad elements per side
W = .1 #square height  
H = .1 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()

np.set_printoptions(precision=3)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)

print("Analytical axial stiffness (EA):")
E = unobtainium.E
A = W*H
print(E*A)
print("Computed Axial Stiffness:")
print(squareXS.K[0,0])

print("Analytical Bending stiffness (EI):")
I = (W*H**3)/12
print(E*I)
print("Computed bending stiffness:")
print(squareXS.K[4,4])

import pyvista
import dolfinx.plot as plot

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
plotter = pyvista.Plotter()
#plot mesh
tdim = domain.topology.dim
topology, cell_types, geom = plot.create_vtk_mesh(domain, tdim)
grid = pyvista.UnstructuredGrid(topology, cell_types, geom)

# sensitivity_to_plot = squareXS.dK2dx[3][3].array
sensitivity_to_plot = squareXS.dK1dx[4][4].array
sensitivity = np.concatenate([sensitivity_to_plot.reshape(-1,2),np.zeros((sensitivity_to_plot.reshape(-1,2).shape[0],1))],axis=1)

grid.point_data["sensitivity"] = sensitivity
warped = grid.warp_by_vector("sensitivity")
# plotter.add_mesh(warped,show_edges=True,opacity=0.5)
plotter.add_mesh(grid,show_edges=True,opacity=1)
plotter.view_isometric()
plotter.show_bounds()
plotter.add_axes()
if not pyvista.OFF_SCREEN:
    plotter.show()



# from dolfinx import cpp
# fdim=0
# num_facets_owned_by_proc = domain.topology.index_map(fdim).size_local
# geometry_entities = cpp.mesh.entities_to_geometry(domain, fdim, np.arange(num_facets_owned_by_proc, dtype=np.int32), False)
# points = domain.geometry.x
# print('Node id, Coords')
# for e, entity in enumerate(geometry_entities):
#     print(e, points[entity])





