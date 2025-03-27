#simple example of cross-sectional analysis of an isotropic symmetric T-section:
from mpi4py import MPI
from dolfinx import mesh, plot,fem
import pyvista
import ALBATROSS
import numpy as np
import ufl
from petsc4py import PETSc

N = 4
offset = 3

# m1,n1 = (N+1)*10+offset,N+1
# m2,n2 = N,N*10+offset

m1,n1 = (N+1),N+1
m2,n2 = N,N

H = 1
W = 1
tf = 1
tw = 1


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
#get (assembled) mass matrices:
M0 = fem.petsc.assemble_matrix(fem.form(ufl.inner(TXS_nm.XSs[0].u,TXS_nm.XSs[0].v)*TXS_nm.XSs[0].dx))
M0.assemble()
M1 = fem.petsc.assemble_matrix(fem.form(ufl.inner(TXS_nm.XSs[1].u,TXS_nm.XSs[1].v)*TXS_nm.XSs[1].dx))
M1.assemble()

phi_0_0 = fem.petsc.assemble_vector(fem.form(TXS_nm.XSs[0].v[0]*TXS_nm.XSs[0].dx))
# phi_1 = fem.petsc.assemble_vector(TXS_nm.XSs[1].v*TXS_nm.XSs[1].dx)


#====== Validate interpolation matrix from mesh 0 to mesh 1 ====== #

#populate mesh 0 with solution:
u0 = fem.Function(TXS_nm.XSs[0].V)
u1 = fem.Function(TXS_nm.XSs[1].V)

#interpolate some function into mesh 0
for i in range(4):
    u0.sub(i).interpolate(lambda x:((i+1)*0.1*x[0]+0.2,
                            (i+1)*0.1*x[1],
                            x[0]*x[1]))


#construct interpolation matrix
TXS_nm._assemble_system_mats()
TXS_nm._construct_coupled_system_matrix()
TXS_nm.collisions[0][1].inter_mat.mult(u0.vector,u1.vector)

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

#====== Validate interpolation matrix from mesh 1 to mesh 0 ====== #
#populate mesh 1 with solution:
u0 = fem.Function(TXS_nm.XSs[0].V)
u1 = fem.Function(TXS_nm.XSs[1].V)

#interpolate some function into mesh 1
for i in range(4):
    u1.sub(i).interpolate(lambda x:((i+1)*0.1*x[0]+0.2,
                            (i+1)*0.1*x[1],
                            x[0]*x[1]))

#perform interpolation
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



#====== mass weighted projection instead of direct linear interpolation ====== #
B = I01.matMult(M0)
# B = I01
# B = I10.transpose().matMult(M0)



P = PETSc.Mat().createDense(I01.getSize())
P.assemble()
# P = PETSc.Vec().createSeq(M0.getSize()[0])

# M_factored = M1.duplicate(copy=True)
# M_factored.setUp()
# M_factored.factorLU(None,None)
# PETSc.MatGetFactor()

# M_factored.MatSolve(B, P)

# ksp = PETSc.KSP().create(M1.comm)
# ksp.setOperators(M1)
# ksp.setType("preonly")
# ksp.getPC().setType("lu")
# ksp.getPC().setFactorSolverType("mumps")
# for j in range(B.getSize()[1]):
#     b = B.getDenseColumnVec(j)
#     vec = P.getDenseColumnVec(j)
#     ksp.solve(b, vec)
#     P.restoreDenseColumnVec(j)

# print()

#KSP APPROACH:
ksp = PETSc.KSP().create()
ksp.setType('preonly')  # Only apply preconditioner (no iterative method)
pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('mumps')  # or 'superlu_dist' for distributed systems
ksp.setOperators(M1)
ksp.setType(PETSc.KSP.Type.PREONLY)
ksp.setUp()  # Factorizes M1

# # ksp.solve(B,P)

# Solve Mm Pnm = Inm Mn (each column of B is a RHS)
# X = M1.duplicate()  # Result matrix
for j in range(B.getSize()[1]):
    b = B.getColumnVector(j)  # Extract j-th column of B
    x = P.getDenseColumnVec(j)
    ksp.solve(b, x)  # Solve for x and store in X
    P.restoreDenseColumnVec(j, x)


#populate mesh 1 with solution:
u0 = fem.Function(TXS_nm.XSs[0].V)
u1 = fem.Function(TXS_nm.XSs[1].V)

#interpolate some function into mesh 1
for i in range(4):
    u0.sub(i).interpolate(lambda x:((i+1)*0.1*x[0]+0.2,
                            (i+1)*0.1*x[1],
                            x[0]*x[1]))

#perform projection
P.mult(u0.vector,u1.vector)

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