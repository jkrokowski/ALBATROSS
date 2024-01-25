#script to demonstrate mesh motion for a simple box xs

import os
# # print(os.environ)
os.environ['SCIPY_USE_PROPACK'] = "1"

import ALBATROSS

from dolfinx import mesh,plot,fem
import pyvista
from mpi4py import MPI
import numpy as np
from dolfinx.io import XDMFFile

#create mesh
N = 10
W = .1
H = .1
t1 = 0.01
t2 = 0.01
t3 = 0.01
t4 = 0.01

points = [(-W/2,H/2),(W/2,H/2),(W/2,-H/2),(-W/2,-H/2)]
thicknesses = [t1,t2,t3,t4]
num_el = 4*[4]
domain = ALBATROSS.utils.create_2D_box(points,thicknesses,num_el,'box_xs')

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    plotter.show_axes()
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()

mats = {'Unobtainium':{ 'TYPE':'ISOTROPIC',
                        'MECH_PROPS':{'E':100.,'nu':.2} ,
                        'DENSITY':2.7e3}
        }

#analyze cross section
boxXS = ALBATROSS.cross_section.CrossSection(domain,mats)
boxXS.getXSStiffnessMatrix()

#output stiffess matrix
print(boxXS.K)

##### MESH MOTION #####
c_el = domain.ufl_domain().ufl_coordinate_element()
V = fem.FunctionSpace(domain, c_el)

#select interior nodes
#TODO: label interior and exterior surfaces of mesh and import to save this work
def moving_boundaries(x):
    return (np.isclose(x[1], H/2-t1) |
            np.isclose(x[1], -H/2+t3) | 
            np.isclose(x[0], W/2-t2 ) |
            np.isclose(x[0], -W/2+t4) )
interior_facets= mesh.locate_entities_boundary(
    domain, domain.topology.dim-1, moving_boundaries)


def fixed_boundaries(x):
    return (np.isclose(x[1], H/2) |
            np.isclose(x[1], -H/2) | 
            np.isclose(x[0], W/2 ) |
            np.isclose(x[0], -W/2) )
exterior_facets= mesh.locate_entities_boundary(
    domain, domain.topology.dim-1, fixed_boundaries)

#prescribe motion of selected nodes
def move_interior(x):
    r = np.sqrt(x[0]**2+x[1]**2)
    return (-5*x[0]*r,
            -8*x[1]*r )


#construct bcs for interior and exterior nodes
u = fem.Function(V)
u.interpolate(move_interior)
interior_dofs= fem.locate_dofs_topological(
    V, domain.topology.dim-1, interior_facets)
interior_bc = fem.dirichletbc(u,interior_dofs)

exterior_dofs = fem.locate_dofs_topological(
    V, domain.topology.dim-1, exterior_facets)
c = fem.Constant(domain, (0.,0.))
exterior_bc = fem.dirichletbc(c, exterior_dofs, V)

bcs = [interior_bc, exterior_bc]

#solve mesh motion problem to move interior mesh
from typing import List
import ufl
def deform_mesh_poisson(V, bcs: List[fem.DirichletBCMetaClass]):
    mesh = V.mesh
    uh = fem.Function(V)
    fem.petsc.set_bc(uh.vector, bcs)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = ufl.inner(fem.Constant(mesh, (0., 0.)), v)*ufl.dx
    problem = fem.petsc.LinearProblem(a, L, bcs, uh)
    problem.solve()
    deformation_array = uh.x.array.reshape((-1, mesh.geometry.dim))
    mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array


deform_mesh_poisson(V,bcs)

if True:
    #plot mesh
    pyvista.global_theme.background = [255, 255, 255, 255]
    pyvista.global_theme.font.color = 'black'
    tdim = domain.topology.dim
    topology, cell_types, geometry = plot.create_vtk_mesh(domain, tdim)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True,opacity=0.25)
    plotter.view_isometric()
    plotter.show_axes()
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()

