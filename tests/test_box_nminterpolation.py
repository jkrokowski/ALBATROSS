#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot,fem
import pyvista
import ALBATROSS
import numpy as np
import ufl
from petsc4py import PETSc

N = 5
offset=4

m1,n1 = N*10+offset,N
m2,n2 = N,N*10+offset
m3,n3 = N*10+offset,N
m4,n4 = N,N*10+offset
H = 1
L = 1
T1 = .1
T2 = .1
T3 = .1
T4 = .1


mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1,cell_type=mesh.CellType.quadrilateral)
# mesh_0 = mesh.create_unit_square(MPI.COMM_WORLD, m1, n1)
mesh_0.geometry.x[:, :2] -= .5
mesh_0.geometry.x[:, 1] *= T1
mesh_0.geometry.x[:, 0] *= L
mesh_0.geometry.x[:, 1] += H/2 - T1/2

mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2,cell_type=mesh.CellType.quadrilateral)
# mesh_1 = mesh.create_unit_square(MPI.COMM_WORLD, m2, n2)
mesh_1.geometry.x[:, :2] -= .5
mesh_1.geometry.x[:, 0] *= T2
mesh_1.geometry.x[:, 1] *= H
mesh_1.geometry.x[:, 0] += L/2 - T2/2

mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3,cell_type=mesh.CellType.quadrilateral)
# mesh_2 = mesh.create_unit_square(MPI.COMM_WORLD, m3, n3)
mesh_2.geometry.x[:, :2] -= .5
mesh_2.geometry.x[:, 1] *= T1
mesh_2.geometry.x[:, 0] *= L
mesh_2.geometry.x[:, 1] -= H/2 - T1/2

mesh_3 = mesh.create_unit_square(MPI.COMM_WORLD, m4, n4,cell_type=mesh.CellType.quadrilateral)
# mesh_3 = mesh.create_unit_square(MPI.COMM_WORLD, m4, n4)
mesh_3.geometry.x[:, :2] -= .5
mesh_3.geometry.x[:, 0] *= T2
mesh_3.geometry.x[:, 1] *= H
mesh_3.geometry.x[:, 0] -= L/2 - T2/2

meshes= [mesh_0,mesh_1,mesh_2,mesh_3]

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=2700)


XSs = [ALBATROSS.cross_section.CrossSection(msh,[unobtainium]) for msh in meshes]

boxXS_nm = ALBATROSS.cross_section.CoupledXSProblem(XSs,pen=1e2)

boxXS_nm.plot_meshes()

I01=ALBATROSS.nonmatching_utils.get_interpolation_matrix(boxXS_nm.regions[1].fxn_space,
                                                        boxXS_nm.regions[0].fxn_space,
                                                        mixed=True)
I10=ALBATROSS.nonmatching_utils.get_interpolation_matrix(boxXS_nm.regions[0].fxn_space,
                                                        boxXS_nm.regions[1].fxn_space,
                                                        mixed=True)
for (m,n),val in np.ndenumerate(boxXS_nm.adjacency):
    if m!=n and val==1:
        print(f'interpolation of mesh {m} onto mesh {n}')
        #populate mesh 0 with solution:
        u0 = fem.Function(boxXS_nm.XSs[m].V)
        u1 = fem.Function(boxXS_nm.XSs[n].V)

        #interpolate some function into mesh 0
        for i in range(4):
            u0.sub(i).interpolate(lambda x:((i+1)*0.1*x[0],
                                    (i+1)*0.1*x[1],
                                    x[0]*x[1]))


        #====== Confirm that the stored interpolation matrix is accurate ====== #
        #confirm stored interpolation matrix is accurate:
        boxXS_nm._assemble_system_mats()
        boxXS_nm._construct_coupled_system_matrix()
        boxXS_nm.collisions[m][n].inter_mat.mult(u0.vector,u1.vector)

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
            for j,(xs,u) in enumerate(zip([boxXS_nm.XSs[m],boxXS_nm.XSs[n]],[u0,u1])):
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
# boxXS_nm.get_xs_stiffness_matrix(correction=None)

