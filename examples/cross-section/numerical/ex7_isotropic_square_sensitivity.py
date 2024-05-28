#simple example of cross-sectional analysis of an isotropic square:
import ALBATROSS
import numpy as np

#cross-section mesh definition
N = 30 #number of quad elements per side
W = .1 #square height  
H = .2 #square depth
points = [[-W/2,-H/2],[W/2, H/2]] #bottom left and upper right point of square

domain = ALBATROSS.mesh.create_rectangle(points,[N,N])

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)

#initialize cross-section object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium])

#show me what you got
# squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.get_xs_stiffness_matrix()

np.set_printoptions(precision=3)

#output stiffness matrixprint('Stiffness matrix:')
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
plotter = pyvista.Plotter(shape=(2,3))
grids = []
warped = []
for i in range(6):
    row = int(i/3)
    col = i%3
    plotter.subplot(row,col)
    #plot mesh
    tdim = domain.topology.dim
    topology, cell_types, geom = plot.create_vtk_mesh(domain, tdim)
    grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

    # sensitivity_to_plot = squareXS.dK1inv[5,5,:]
    # sensitivity_to_plot = squareXS.dK2[5,5,:]
    # sensitivity_to_plot = squareXS.dK1invT[5,5,:]
    # sensitivity_to_plot = squareXS.dKdx[3,3,:].reshape(-1,2)
    sensitivity_to_plot = np.zeros((geom.shape[0],2))

    sensitivity_to_plot[squareXS.boundary_dofs,:] = squareXS.dSdx[i,i,:].reshape(-1,2)[squareXS.boundary_dofs,:]
    # sensitivity_to_plot[squareXS.boundary_dofs,:] = squareXS.dKdx[4,4,:]
    # sensitivity_to_plot = squareXS.dKdx[0,0,:]
    # sensitivity_to_plot[squareXS.boundary_dofs] = squareXS.dSdx[2,2,:]
    sensitivity = np.concatenate([sensitivity_to_plot,np.zeros_like(sensitivity_to_plot)],axis=1)

    sensitivity = np.concatenate([sensitivity_to_plot,np.zeros((sensitivity_to_plot.shape[0],1))],axis=1)

    grids[i].point_data["sensitivity"] = sensitivity
    warped.append(grids[i].warp_by_vector("sensitivity",factor=.00001))
    # plotter.add_mesh(warped,show_edges=True,opacity=0.5)
    # plotter.add_mesh(grids[i],show_edges=True,opacity=1,scalar_bar_args={'title': f'Sensitivity_{i}'})
    plotter.add_mesh(warped[i],show_edges=True,opacity=1,scalar_bar_args={'title': f'Sensitivity_{i}'})
    plotter.add_text(f'dS/dx({i},{i})')
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





