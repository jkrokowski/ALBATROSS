import dolfinx.cpp.mesh
import numpy as np
import pyvista
from dolfinx import plot
import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI

def get_vtx_to_dofs(domain,V):
     '''
     solution from https://fenicsproject.discourse.group/t/application-of-point-forces-mapping-vertex-indices-to-corresponding-dofs/9646
     --------------
     input: subspace to find DOFs in
     output: map of DOFs related to their corresponding vertices
     '''
     V0, V0_to_V = V.collapse()
     dof_layout = V0.dofmap.dof_layout

     num_vertices = domain.topology.index_map(0).size_local + domain.topology.index_map(0).num_ghosts
     vertex_to_par_dof_map = np.zeros(num_vertices, dtype=np.int32)
     num_cells = domain.topology.index_map(
          domain.topology.dim).size_local + domain.topology.index_map(
          domain.topology.dim).num_ghosts
     c_to_v = domain.topology.connectivity(domain.topology.dim, 0)
     for cell in range(num_cells):
          vertices = c_to_v.links(cell)
          dofs = V0.dofmap.cell_dofs(cell)
          for i, vertex in enumerate(vertices):
               vertex_to_par_dof_map[vertex] = dofs[dof_layout.entity_dofs(0, i)]

     geometry_indices = dolfinx.cpp.mesh.entities_to_geometry(
          domain, 0, np.arange(num_vertices, dtype=np.int32), False)
     bs = V0.dofmap.bs
     vtx_to_dof = np.zeros((num_vertices,bs), dtype=np.int32)
     for vertex, geom_index in enumerate(geometry_indices):
          par_dof = vertex_to_par_dof_map[vertex]
          for b in range(bs):
               vtx_to_dof[vertex, b] = V0_to_V[par_dof*bs+b]
     # vtx_to_dof = np.reshape(vtx_to_dof, (-1,1))

     return vtx_to_dof

def plot_xdmf_mesh(msh,add_nodes=False):
     #plot mesh
     pyvista.global_theme.background = [255, 255, 255, 255]
     pyvista.global_theme.font.color = 'black'
     tdim = msh.topology.dim
     topology, cell_types, geom = plot.create_vtk_mesh(msh, tdim)
     grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
     plotter = pyvista.Plotter()
     plotter.add_mesh(grid, show_edges=True,opacity=0.25)
     if add_nodes==True:
          plotter.add_mesh(grid, style='points')
     plotter.view_isometric()
     if not pyvista.OFF_SCREEN:
          plotter.show()

def beam_interval_mesh_3D(pts,ne,meshname):
    '''
    pts = list of nx (x,y,z) locations of a beam nodes (np)
    ne = list of number of elements for each segment between nodes (np-1)
    meshname = name of mesh
    '''
    filename = 'output/'+meshname+'.xdmf'

    gdim = 3
    tdim = 1

    gmsh.initialize()

    #construct line in 3D space
    gmsh.model.add(meshname)
    gmsh.model.setCurrent(meshname)
    
    pt_tags = []
    for pt in pts:
        pt_tag = gmsh.model.geo.addPoint(pt[0],pt[1],pt[2])
        pt_tags.append(pt_tag)
    line_tags = []
    for i,n in enumerate(ne):
        line_tag = gmsh.model.geo.addLine(pt_tags[i],pt_tags[i+1])
        line_tags.append(line_tag)
        gmsh.model.geo.mesh.setTransfiniteCurve(line_tag, int(n + 1))

    # Synchronize model representation with gmsh model
    gmsh.model.geo.synchronize()

    # add physical marker
    gmsh.model.add_physical_group(tdim,line_tags)

    #generate the mesh and optionally write the gmsh mesh file
    gmsh.model.mesh.generate(gdim)
    # gmsh.write(filename)

    #use meshio to convert msh file to xdmf
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
    msh.name = meshname
    # cell_markers.name = f"{msh.name}_cells"
    # facet_markers.name = f"{msh.name}_facets"

    # close gmsh API
    gmsh.finalize()

    #write xdmf mesh file
    with XDMFFile(msh.comm, filename, "w") as file:
        file.write_mesh(msh)

    #return mesh
    with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        return xdmf.read_mesh(name=meshname) 