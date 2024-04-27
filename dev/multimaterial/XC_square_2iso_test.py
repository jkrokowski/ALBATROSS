# Cross Sectional Analysis using the Principle of Minimum Total Complementary Energy
# ================

from mpi4py import MPI
import dolfinx.cpp.mesh
from dolfinx import mesh,plot
from dolfinx.mesh import meshtags
from dolfinx.fem import locate_dofs_topological,Constant,FunctionSpace,Function,form,assemble_scalar,VectorFunctionSpace,Expression,TensorFunctionSpace,locate_dofs_geometrical
from dolfinx.fem.petsc import create_vector,assemble_matrix,assemble_vector
from ufl import (sym,FiniteElement,split,MixedElement,dot,lhs,rhs,Identity,inner,outer,TrialFunction,TestFunction,Measure,grad,exp,sin,SpatialCoordinate,FacetNormal,indices,as_tensor,as_matrix,as_vector,VectorElement,TensorElement,Dx)
from petsc4py import PETSc
import pyvista
import numpy as np
from dolfinx.io import XDMFFile
from dolfinx.plot import create_vtk_mesh

tdim = 2
from dolfinx import geometry

import ALBATROSS
# Create 2d mesh and define function space
N = 10
W = .1
H = .1

pyvista.global_theme.background = [255, 255, 255, 255]
pyvista.global_theme.font.color = 'black'
#read in mesh
xcName = "square_2iso_quads"
fileName = "output/"+ xcName + ".xdmf"
with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
    #mesh generation with meshio seems to have difficulty renaming the mesh name
    # (but not the file, hence the "Grid" name property)
    domain = xdmf.read_mesh(name="Grid")
    ct = xdmf.read_meshtags(domain, name="Grid")   
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)

#plot mesh:
p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = plot.create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

#right
right_marker=0
right_facets = ct.find(right_marker)
right_mt = meshtags(domain, tdim, right_facets, right_marker)
#left
left_marker=1
left_facets = ct.find(left_marker)
left_mt = meshtags(domain, tdim, left_facets, left_marker)

p = pyvista.Plotter(window_size=[800, 800])
num_cells_local = domain.topology.index_map(domain.topology.dim).size_local
topology, cell_types, x = create_vtk_mesh(domain, domain.topology.dim, np.arange(num_cells_local, dtype=np.int32))
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Marker"] = ct.values
p.add_mesh(grid, show_edges=True)
p.view_xy()
p.show()

unobtainium = ALBATROSS.material.Material(name='unobtainium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':100,'nu':0.2},
                                           density=10000,
                                           celltag=0)
adamantium = ALBATROSS.material.Material(name='adamantium',
                                           mat_type='ISOTROPIC',
                                           mech_props={'E':10,'nu':0.2},
                                           density=5000,
                                           celltag=1)

#initialize cross-seciton object
squareXS = ALBATROSS.cross_section.CrossSection(domain,[unobtainium,adamantium],celltags=ct)

#show me what you got
squareXS.plot_mesh()

#compute the stiffness matrix
squareXS.getXSStiffnessMatrix()

np.set_printoptions(precision=3)

#output flexibility matrix
print('Flexibility matrix:')
print(squareXS.S)

#output stiffness matrix
print('Stiffness matrix:')
print(squareXS.K)
