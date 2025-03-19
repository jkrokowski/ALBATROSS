#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot,fem
import pyvista
import ALBATROSS
import numpy as np
import ufl
from petsc4py import PETSc

# np.set_printoptions(precision=3)

# m1,n1 = 102,10
# m2,n2 = 9,104
# m1,n1 = 54,5
# m2,n2 = 4,45
# m1,n1 = 40+1,4
# m2,n2 = 4,40+1
# m1,n1 = 30,3
# m2,n2 = 3,30
# m1,n1 = 10,1
# m2,n2 = 1,9assembleLinearSystemBackground

N = 4
offset = 3

m1,n1 = N*10+offset,N+1
m2,n2 = N,N*10+offset

H = 1
W = 1
tf = .1
tw = .1


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= tf
mesh_0.geometry.x[:, 0] *= W
mesh_0.geometry.x[:, 1] += H/2 - tf/2

mesh_0.name = 'f'

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= tw
mesh_1.geometry.x[:, 1] *= W
# mesh_1.geometry.x[:, 0] += T2/2

mesh_1.name = 'w'

meshes= [mesh_0,mesh_1]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

TXS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e4)

TXS_nm.plot_meshes()

I01=ALBATROSS.nonmatching_utils.get_interpolation_matrix(TXS_nm.regions[1].fxn_space,
                                                        TXS_nm.regions[0].fxn_space,
                                                        mixed=True)
I10=ALBATROSS.nonmatching_utils.get_interpolation_matrix(TXS_nm.regions[0].fxn_space,
                                                        TXS_nm.regions[1].fxn_space,
                                                        mixed=True)
#get mass matrices:
M0 = fem.petsc.assemble_matrix(fem.form(ufl.inner(TXS_nm.XSs[0].u,TXS_nm.XSs[0].v)*TXS_nm.XSs[0].dx))
M1 = fem.petsc.assemble_matrix(fem.form(ufl.inner(TXS_nm.XSs[1].u,TXS_nm.XSs[1].v)*TXS_nm.XSs[1].dx))

#populate mesh 0 with solution:
u0 = fem.Function(TXS_nm.XSs[0].V)
u1 = fem.Function(TXS_nm.XSs[1].V)

#interpolate some function into mesh 0
for i in range(4):
    u0.sub(i).interpolate(lambda x:((i+1)*0.1*x[0]**3,
                            -(i+1)*x[1]**3,
                            x[0]**2*x[1]**2))


#====== Confirm that the stored interpolation matrix is accurate ====== #
#confirm stored interpolation matrix is accurate:
TXS_nm._assemble_system_mats()
TXS_nm._construct_coupled_system_matrix()
TXS_nm.collisions[0][1].inter_mat.mult(u0.vector,u1.vector)



#====== Validate the whole interpolation matrix ===== #

# #interpolate u0 solution into u1
# I01.mult(u0.vector,u1.vector)


#====== Validate a single sub-subspace ===== #

# I01_ubar0 = ALBATROSS.nonmatching_utils.interpolation_matrix_nonmatching_meshes(TXS_nm.regions[1].fxn_space.sub(0).sub(0).collapse()[0],
#                                                                                TXS_nm.regions[0].fxn_space.sub(0).sub(0).collapse()[0])

# I10_ubar0 = ALBATROSS.nonmatching_utils.interpolation_matrix_nonmatching_meshes(TXS_nm.regions[0].fxn_space.sub(0).sub(0).collapse()[0],
#                                                                                TXS_nm.regions[1].fxn_space.sub(0).sub(0).collapse()[0])

# I10_ubar0.assemble()
# I01_ubar0.assemble()

# u0bar0 = PETSc.Vec().create()
# u0bar0.setSizes(TXS_nm.regions[0].fxn_space.sub(0).sub(0).collapse()[0].dofmap.index_map.size_global)
# u0bar0.setFromOptions()

# u1bar0 = PETSc.Vec().create()
# u1bar0.setSizes(TXS_nm.regions[1].fxn_space.sub(0).sub(0).collapse()[0].dofmap.index_map.size_global)
# u1bar0.setFromOptions()

# u0bar0.array = u0.sub(0).sub(0).collapse().x.array

# I01_ubar0.mult(u0bar0,u1bar0)
# # I10_ubar0.mult(u0bar0,u1bar0)

# u0bar0_to_u0 =TXS_nm.regions[0].fxn_space.sub(0).sub(0).collapse()[1]
# u1bar0_to_u1 =TXS_nm.regions[1].fxn_space.sub(0).sub(0).collapse()[1]

# # u0.x.array[u0bar0_to_u0] = u0bar0.x.array
# u1.x.array[u1bar0_to_u1] = u1bar0.array


# plot the warping functions after interpolating a solution to them  
pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
plotter = pyvista.Plotter()
        
mode = ['ubar','uhat', 'utilde', 'ubreve']
plotter = pyvista.Plotter(shape=(2,2))
grids = []
warped = []
for i in range(4):
    row = int(i/2)
    col = i%2
    name = f'mode_{i}'
    plotter.subplot(row,col)
    #plot mesh
    solution_modes=[]
    indiv_grids = []
    indiv_warped = []
    for j,(xs,u) in enumerate(zip([TXS_nm.XSs[0],TXS_nm.XSs[1]],[u0,u1])):
        tdim = xs.msh.topology.dim

        V0,V0_to_V = xs.V.sub(0).collapse()
        topology, cell_types, geom = plot.vtk_mesh(V0)
        indiv_grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

        ui = u.sub(i).collapse().x.array

        ui_plot = ui.reshape((geom.shape[0], 3))[:,[1,2,0]]
        solution_modes.append(ui_plot)
    
        indiv_grids[j][name]= solution_modes[j]
    
        indiv_warped.append(indiv_grids[j].warp_by_vector(name,factor=1))

        plotter.add_mesh(indiv_warped[j],show_edges=True,opacity=.9)
        plotter.add_mesh(indiv_grids[j],show_edges=True,opacity=.5,scalar_bar_args={'title': f'warping function {i}'})
    
    grids.append(indiv_grids)
    warped.append(indiv_warped)
    
    plotter.add_text(mode[i])

    plotter.view_xy()
plotter.show_bounds()
if not pyvista.OFF_SCREEN:
    plotter.show()

#validate the other direction of interpolation:
#populate mesh 1 with solution:
u0 = fem.Function(TXS_nm.XSs[0].V)
u1 = fem.Function(TXS_nm.XSs[1].V)

#interpolate some function into mesh 0
for i in range(4):
    u1.sub(i).interpolate(lambda x:((i+1)*0.1*x[0]+0.2,
                            (i+1)*0.1*x[1],
                            x[0]*x[1]))


#====== Confirm that the stored interpolation matrix is accurate ====== #
#confirm stored interpolation matrix is accurate:
# TXS_nm._assemble_system_mats()
# TXS_nm._construct_coupled_system_matrix()
TXS_nm.collisions[1][0].inter_mat.mult(u1.vector,u0.vector)

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
plotter = pyvista.Plotter()
        
mode = ['ubar','uhat', 'utilde', 'ubreve']
plotter = pyvista.Plotter(shape=(2,2))
grids = []
warped = []
for i in range(4):
    row = int(i/2)
    col = i%2
    name = f'mode_{i}'
    plotter.subplot(row,col)
    #plot mesh
    solution_modes=[]
    indiv_grids = []
    indiv_warped = []
    for j,(xs,u) in enumerate(zip([TXS_nm.XSs[0],TXS_nm.XSs[1]],[u0,u1])):
        tdim = xs.msh.topology.dim

        V0,V0_to_V = xs.V.sub(0).collapse()
        topology, cell_types, geom = plot.vtk_mesh(V0)
        indiv_grids.append(pyvista.UnstructuredGrid(topology, cell_types, geom))

        ui = u.sub(i).collapse().x.array

        ui_plot = ui.reshape((geom.shape[0], 3))[:,[1,2,0]]
        solution_modes.append(ui_plot)
    
        indiv_grids[j][name]= solution_modes[j]
    
        indiv_warped.append(indiv_grids[j].warp_by_vector(name,factor=1))

        plotter.add_mesh(indiv_warped[j],show_edges=True,opacity=.9)
        plotter.add_mesh(indiv_grids[j],show_edges=True,opacity=.5,scalar_bar_args={'title': f'warping function {i}'})
    
    grids.append(indiv_grids)
    warped.append(indiv_warped)
    
    plotter.add_text(mode[i])

    plotter.view_xy()
plotter.show_bounds()
if not pyvista.OFF_SCREEN:
    plotter.show()



print()
# TXS_nm.get_xs_stiffness_matrix(correction=None)

