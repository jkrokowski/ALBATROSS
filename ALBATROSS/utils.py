import dolfinx.cpp.mesh
import numpy as np
import pyvista
from dolfinx import plot,mesh
import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI
import scipy.io
from scipy.sparse import csc_matrix,csr_matrix
import meshio

from dolfinx.geometry import BoundingBoxTree,compute_collisions,compute_colliding_cells
from ufl import TestFunction,TrialFunction,inner,dx
from dolfinx.fem.petsc import assemble_matrix,assemble_vector,apply_lifting,set_bc,create_vector
from dolfinx.fem import form
from petsc4py import PETSc

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

def plot_xdmf_mesh(msh,surface=True,add_nodes=False):
     pyvista.global_theme.background = [255, 255, 255, 255]
     pyvista.global_theme.font.color = 'black'
     plotter = pyvista.Plotter()
     #plot mesh
     if type(msh) != list:
          tdim = msh.topology.dim
          topology, cell_types, geom = plot.create_vtk_mesh(msh, tdim)
          grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
          if surface ==True:
               plotter.add_mesh(grid,show_edges=True,opacity=0.25)
          if surface == False:
               plotter.add_mesh(grid,color='k',show_edges=True)
          if add_nodes==True:
               plotter.add_mesh(grid, style='points',color='k')
          plotter.view_isometric()
          plotter.add_axes()
          if not pyvista.OFF_SCREEN:
               plotter.show()
     else:
          for m in msh:
               tdim = m.topology.dim
               topology, cell_types, geom = plot.create_vtk_mesh(m, tdim)
               grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
               # plotter.add_mesh(grid,show_edges=True,opacity=0.25)
               plotter.add_mesh(grid,color='k',show_edges=True)
               if add_nodes==True:
                    plotter.add_mesh(grid, style='points',color='k')
          plotter.view_isometric()
          plotter.add_axes()
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
     cell_markers.name = f"{msh.name}_cells"
     facet_markers.name = f"{msh.name}_facets"

     # close gmsh API
     gmsh.finalize()

     #write xdmf mesh file
     with XDMFFile(msh.comm, filename, "w") as file:
          file.write_mesh(msh)

     #return mesh
     with XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
          return xdmf.read_mesh(name=meshname)
     
def create_rectangle(pts,num_el):
     pts = np.array(pts)
     return mesh.create_rectangle( MPI.COMM_WORLD,pts,num_el, cell_type=mesh.CellType.quadrilateral)

def create_2D_box(pts,thicknesses,num_el,meshname):
     '''
     pts = list of 4 corners of hollow box in (x,y) locations 
          provide in clockwise order starting from upper left:
               [(pt1=upper left x,y), (pt2 = upper right x,y), 
               (pt3=bottom right x,y), (pt4=bottom left x,y) ]
     thicknesses = list of wall thicknesses for walls:
                    [(pt1 to pt2 thickness),(pt2 to pt3 thickness),
                    (pt3 to pt4 thickness),(pt4 to pt1 thickness)]
     num_el = list of number of elements through thickness for 
                    each specified thickness
     meshname = name of mesh
     '''

     #unpack input 
     # t: top, b:bottom, l:left, r:right, i: inside
     # e.g. tlicx is the top left inside corner's x coordinate)
     [(tlcx,tlcy),(trcx,trcy),(brcx,brcy),(blcx,blcy)]=pts
     [t1,t2,t3,t4]=thicknesses
     [n1,n2,n3,n4]=num_el
     filename = 'output/'+meshname+'.xdmf'

     #get inside corners ()
     (tlicx,tlicy)=(tlcx+t4,tlcy-t1)
     (tricx,tricy)=(trcx-t2,trcy-t1)
     (bricx,bricy)=(brcx-t2,brcy+t3)
     (blicx,blicy)=(blcx+t4,blcy+t3)
     
     #choose number of elements between corners for height and width:
     width = trcx-tlcx
     height = tlcy-blcy
     avg_cell_size = np.average(np.array(thicknesses)/np.array(num_el))

     nw = int(np.ceil((width-t2-t4)/avg_cell_size))
     nh = int(np.ceil((height-t1-t3)/avg_cell_size))
     
     #for 2 xs mesh, gdim=tdim=2
     gdim = 2
     tdim = 2

     #initialize, add model and activate
     print("Generating 2D box xs mesh...")
     gmsh.initialize()
     gmsh.option.setNumber("General.Terminal",0) #suppress gmsh output
     gmsh.model.add(meshname)
     gmsh.model.setCurrent(meshname)
     
     #list of coordinates for each sub-section of box xs
     box_nodes = [[tlcx,tlicy,tlicx,tlcy],
                  [tlicx,tlicy,tricx,trcy],
                  [tricx,tricy,trcx,trcy],
                  [bricx,bricy,trcx,tricy],
                  [bricx,brcy,brcx,bricy],
                  [blicx,blcy,bricx,bricy],
                  [blcx,blcy,blicx,blicy],
                  [blcx,blicy,tlicx,tlicy]]
     
     #number of elements in x and y directions for each sub-section
     box_el_num = [[n4,n1],
               [nw,n1],
               [n2,n1],
               [n2,nh],
               [n2,n3],
               [nw,n3],
               [n4,n3],
               [n4,nh]]
     
     #loop to build all 8 subsections of the hollow box
     for box,el_num in zip(box_nodes,box_el_num):
          x1,y1,x2,y2 = box
          num_el_x, num_el_y = el_num

          p1 = gmsh.model.geo.addPoint(x1,y1,0)
          p2 = gmsh.model.geo.addPoint(x1,y2,0)
          p3 = gmsh.model.geo.addPoint(x2,y2,0)
          p4 = gmsh.model.geo.addPoint(x2,y1,0)
          l1 = gmsh.model.geo.addLine(p1, p2)
          l2 = gmsh.model.geo.addLine(p2, p3)
          l3 = gmsh.model.geo.addLine(p3, p4)
          l4 = gmsh.model.geo.addLine(p4, p1)

          gmsh.model.geo.mesh.setTransfiniteCurve(l1, int(num_el_y + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l2, int(num_el_x + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l3, int(num_el_y + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l4, int(num_el_x + 1))

          cl1 = gmsh.model.geo.addCurveLoop([p1, p2, p3, p4])
          rect =gmsh.model.geo.addPlaneSurface([cl1])

          gmsh.model.geo.mesh.setTransfiniteSurface(rect)
          gmsh.model.geo.mesh.setRecombine(2, rect)
     
     #get list of tags of all box nodes
     tags = list(range(1,len(box_nodes)+1))

     #remove duplicate points to ensure closed section 
     # (not guaranteed to be closed otherwise!)
     gmsh.model.geo.remove_all_duplicates()
     gmsh.model.geo.synchronize()
     gmsh.model.add_physical_group(tdim,tags,0,"rect")

     #generate the mesh and optionally write the gmsh mesh file
     gmsh.model.mesh.generate(gdim)
     gmsh.write("output/" +meshname + ".msh")

     #uncomment this below if you want to run gmsh window for debug, etc
     # gmsh.fltk.run()

     # close gmsh API
     gmsh.finalize()

     #read gmsh file and write and xdmf
     if MPI.COMM_WORLD.rank == 0:
          # Read in mesh
          msh = meshio.read("output/" +meshname + ".msh")

          # Create and save one file for the mesh, and one file for the facets 
          mesh = gmsh_to_xdmf(msh, "quad", prune_z=True)
          meshio.write(f"output/"+meshname+".xdmf", mesh)

     fileName = "output/"+ meshname + ".xdmf"

     #read xdmf and return dolfinx mesh object
     with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
          #mesh generation with meshio seems to have difficulty renaming the mesh name
          # (but not the file, hence the "Grid" name property)
          domain = xdmf.read_mesh(name="Grid")
          domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
          # print("Finished meshing 2D with %i elements" % (domain.num_cells))
          return domain
def create_I_section(pts,thicknesses,num_el,meshname):
     #TODO: UPDATE THIS
     '''
     pts = list of 4 corners of hollow box in (x,y) locations 
          provide in clockwise order starting from upper left:
               [(pt1=upper left x,y), (pt2 = upper right x,y), 
               (pt3=bottom right x,y), (pt4=bottom left x,y) ]
     thicknesses = list of wall thicknesses for walls:
                    [(pt1 to pt2 thickness),(pt2 to pt3 thickness),
                    (pt3 to pt4 thickness),(pt4 to pt1 thickness)]
     num_el = list of number of elements through thickness for 
                    each specified thickness
     meshname = name of mesh
     '''

     #unpack input 
     # t: top, b:bottom, l:left, r:right, i: inside
     # e.g. tlicx is the top left inside corner's x coordinate)
     [(tlcx,tlcy),(trcx,trcy),(brcx,brcy),(blcx,blcy)]=pts
     [t1,t2,t3,t4]=thicknesses
     [n1,n2,n3,n4]=num_el
     filename = 'output/'+meshname+'.xdmf'

     #get inside corners ()
     (tlicx,tlicy)=(tlcx+t4,tlcy-t1)
     (tricx,tricy)=(trcx-t2,trcy-t1)
     (bricx,bricy)=(brcx-t2,brcy+t3)
     (blicx,blicy)=(blcx+t4,blcy+t3)
     
     #choose number of elements between corners for height and width:
     width = trcx-tlcx
     height = tlcy-blcy
     avg_cell_size = np.average(np.array(thicknesses)/np.array(num_el))

     nw = int(np.ceil((width-t2-t4)/avg_cell_size))
     nh = int(np.ceil((height-t1-t3)/avg_cell_size))
     
     #for 2 xs mesh, gdim=tdim=2
     gdim = 2
     tdim = 2

     #initialize, add model and activate
     print("Generating 2D box xs mesh...")
     gmsh.initialize()
     gmsh.option.setNumber("General.Terminal",0) #suppress gmsh output
     gmsh.model.add(meshname)
     gmsh.model.setCurrent(meshname)
     
     #list of coordinates for each sub-section of box xs
     box_nodes = [[tlcx,tlicy,tlicx,tlcy],
                  [tlicx,tlicy,tricx,trcy],
                  [tricx,tricy,trcx,trcy],
                  [bricx,bricy,trcx,tricy],
                  [bricx,brcy,brcx,bricy],
                  [blicx,blcy,bricx,bricy],
                  [blcx,blcy,blicx,blicy],
                  [blcx,blicy,tlicx,tlicy]]
     
     #number of elements in x and y directions for each sub-section
     box_el_num = [[n4,n1],
               [nw,n1],
               [n2,n1],
               [n2,nh],
               [n2,n3],
               [nw,n3],
               [n4,n3],
               [n4,nh]]
     
     #loop to build all 8 subsections of the hollow box
     for box,el_num in zip(box_nodes,box_el_num):
          x1,y1,x2,y2 = box
          num_el_x, num_el_y = el_num

          p1 = gmsh.model.geo.addPoint(x1,y1,0)
          p2 = gmsh.model.geo.addPoint(x1,y2,0)
          p3 = gmsh.model.geo.addPoint(x2,y2,0)
          p4 = gmsh.model.geo.addPoint(x2,y1,0)
          l1 = gmsh.model.geo.addLine(p1, p2)
          l2 = gmsh.model.geo.addLine(p2, p3)
          l3 = gmsh.model.geo.addLine(p3, p4)
          l4 = gmsh.model.geo.addLine(p4, p1)

          gmsh.model.geo.mesh.setTransfiniteCurve(l1, int(num_el_y + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l2, int(num_el_x + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l3, int(num_el_y + 1))
          gmsh.model.geo.mesh.setTransfiniteCurve(l4, int(num_el_x + 1))

          cl1 = gmsh.model.geo.addCurveLoop([p1, p2, p3, p4])
          rect =gmsh.model.geo.addPlaneSurface([cl1])

          gmsh.model.geo.mesh.setTransfiniteSurface(rect)
          gmsh.model.geo.mesh.setRecombine(2, rect)
     
     #get list of tags of all box nodes
     tags = list(range(1,len(box_nodes)+1))

     #remove duplicate points to ensure closed section 
     # (not guaranteed to be closed otherwise!)
     gmsh.model.geo.remove_all_duplicates()
     gmsh.model.geo.synchronize()
     gmsh.model.add_physical_group(tdim,tags,0,"rect")

     #generate the mesh and optionally write the gmsh mesh file
     gmsh.model.mesh.generate(gdim)
     gmsh.write("output/" +meshname + ".msh")

     #uncomment this below if you want to run gmsh window for debug, etc
     # gmsh.fltk.run()

     # close gmsh API
     gmsh.finalize()

     #read gmsh file and write and xdmf
     if MPI.COMM_WORLD.rank == 0:
          # Read in mesh
          msh = meshio.read("output/" +meshname + ".msh")

          # Create and save one file for the mesh, and one file for the facets 
          mesh = gmsh_to_xdmf(msh, "quad", prune_z=True)
          meshio.write(f"output/"+meshname+".xdmf", mesh)

     fileName = "output/"+ meshname + ".xdmf"

     #read xdmf and return dolfinx mesh object
     with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
          #mesh generation with meshio seems to have difficulty renaming the mesh name
          # (but not the file, hence the "Grid" name property)
          domain = xdmf.read_mesh(name="Grid")
          domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
          # print("Finished meshing 2D with %i elements" % (domain.num_cells))
          return domain
     
def create_circle(radius,num_el,meshname):
     '''
     radius = outer radius of circle
     num_el = number of elements along radial direction 
     meshname = name of mesh
     '''
     #mesh parameters
     gdim=2
     tdim=2

     #cross section properties
     xcName = meshname
     R=radius 
     num_el_thick = num_el
     #TODO: use transfinite interpolation to control mesh number...
     #set number of elements along circumference proportional to the 
     # number of elements through the thickness
     num_el_circum = 3 * num_el_thick

     gmsh.initialize()
     gmsh.model.add(xcName)
     gmsh.model.setCurrent(xcName)

     #meshtags
     d1 = gmsh.model.occ.add_disk(0,0,0,R,R)

     gmsh.model.occ.synchronize()
     gmsh.model.add_physical_group(tdim,[d1])

     gmsh.option.setNumber('Mesh.MeshSizeMin', R/num_el_thick)
     gmsh.option.setNumber('Mesh.MeshSizeMax', R/num_el_thick)

     gmsh.model.mesh.generate(gdim)

     #uncomment this below if you want to run gmsh window for debug, etc
     # gmsh.fltk.run()

     #write xdmf mesh file
     gmsh.write("output/" +meshname + ".msh")

     # close gmsh API
     gmsh.finalize()

     #read gmsh file and write and xdmf
     if MPI.COMM_WORLD.rank == 0:
          # Read in mesh
          msh = meshio.read("output/" +meshname + ".msh")

          # Create and save one file for the mesh, and one file for the facets 
          mesh = gmsh_to_xdmf(msh, "triangle", prune_z=True)
          meshio.write(f"output/"+meshname+".xdmf", mesh)

     fileName = "output/"+ meshname + ".xdmf"

     #read xdmf and return dolfinx mesh object
     with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
          #mesh generation with meshio seems to have difficulty renaming the mesh name
          # (but not the file, hence the "Grid" name property)
          domain = xdmf.read_mesh(name="Grid")
          domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
          # print("Finished meshing 2D with %i elements" % (domain.num_cells))
          return domain
def create_hollow_circle(radius,thickness,num_el,meshname):
     '''
     radius = outer radius of circle
     thicknesses = wall thickness
     num_el = number of elements through thickness
     meshname = name of mesh
     '''
     #mesh parameters
     gdim=2
     tdim=2

     #cross section properties
     xcName = meshname
     R=radius 
     t = thickness
     num_el_thick = num_el
     #set number of elements along circumference proportional to the 
     # number of elements through the thickness
     num_el_circum = int((2*R*np.pi/t)) * num_el_thick

     gmsh.initialize()
     gmsh.model.add(xcName)
     gmsh.model.setCurrent(xcName)

     #meshtags
     markerId = 1
     p1 = gmsh.model.geo.add_point(0,0,0)
     p2 = gmsh.model.geo.add_point(0,R,0)
     p3 = gmsh.model.geo.add_point(0,R-t,0)
     p4 = gmsh.model.geo.add_point(0,-R,0)
     p5 = gmsh.model.geo.add_point(0,-(R-t),0)

     ca1 = gmsh.model.geo.add_circle_arc(p2,p1,p4)
     ca2 = gmsh.model.geo.add_circle_arc(p3,p1,p5)
     ca3 = gmsh.model.geo.add_circle_arc(p4,p1,p2)
     ca4 = gmsh.model.geo.add_circle_arc(p5,p1,p3)
     l1 = gmsh.model.geo.add_line(p2,p3)
     l2 = gmsh.model.geo.add_line(p4,p5)

     edges1 = gmsh.model.geo.addCurveLoop([ca1,l2,-ca2,-l1],-1)
     hollow_disk1 = gmsh.model.geo.addPlaneSurface([edges1],-1)
     edges2 = gmsh.model.geo.addCurveLoop([ca3,-l2,-ca4,l1],-1)
     hollow_disk2 = gmsh.model.geo.addPlaneSurface([edges2],-1)

     gmsh.model.geo.mesh.setTransfiniteCurve(ca1, int(num_el_circum))
     gmsh.model.geo.mesh.setTransfiniteCurve(ca2, int(num_el_circum))
     gmsh.model.geo.mesh.setTransfiniteCurve(ca3, int(num_el_circum))
     gmsh.model.geo.mesh.setTransfiniteCurve(ca4, int(num_el_circum))
     gmsh.model.geo.mesh.setTransfiniteCurve(l1,int(num_el_thick))
     gmsh.model.geo.mesh.setTransfiniteCurve(l2,int(num_el_thick))
     gmsh.model.geo.mesh.setTransfiniteSurface(hollow_disk1)
     gmsh.model.geo.mesh.setTransfiniteSurface(hollow_disk2)

     gmsh.model.add_physical_group(tdim,[hollow_disk1,hollow_disk2],0,xcName)

     #generate the mesh and optionally write the gmsh mesh file
     # gmsh.model.geo.remove_all_duplicates()
     gmsh.model.geo.mesh.setRecombine(2, hollow_disk1)
     gmsh.model.geo.mesh.setRecombine(2, hollow_disk2)

     gmsh.model.geo.synchronize()
     gmsh.model.mesh.generate(gdim)
     # gmsh.model.mesh.removeDuplicateElements()
     # gmsh.model.mesh.removeDuplicateNodes()

     #uncomment this below if you want to run gmsh window for debug, etc
     # gmsh.fltk.run()

     #write xdmf mesh file
     gmsh.write("output/" +meshname + ".msh")

     # close gmsh API
     gmsh.finalize()

     #read gmsh file and write and xdmf
     if MPI.COMM_WORLD.rank == 0:
          # Read in mesh
          msh = meshio.read("output/" +meshname + ".msh")

          # Create and save one file for the mesh, and one file for the facets 
          mesh = gmsh_to_xdmf(msh, "quad", prune_z=True)
          meshio.write(f"output/"+meshname+".xdmf", mesh)

     fileName = "output/"+ meshname + ".xdmf"

     #read xdmf and return dolfinx mesh object
     with XDMFFile(MPI.COMM_WORLD, fileName, "r") as xdmf:
          #mesh generation with meshio seems to have difficulty renaming the mesh name
          # (but not the file, hence the "Grid" name property)
          domain = xdmf.read_mesh(name="Grid")
          domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
          # print("Finished meshing 2D with %i elements" % (domain.num_cells))
          return domain
     
def get_pts_and_cells(domain,points):
     '''
     ARGS:
          point = tuple of (x,y,z) locations to return displacements and rotations
     '''
     bb_tree = BoundingBoxTree(domain,domain.topology.dim)
     points = np.array(points)

     cells = []
     points_on_proc = []
     # Find cells whose bounding-box collide with the the points
     cell_candidates = compute_collisions(bb_tree, points)
     # Choose one of the cells that contains the point
     colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
     for i, point in enumerate(points):
          if len(colliding_cells.links(i))>0:
               points_on_proc.append(point)
               cells.append(colliding_cells.links(i)[0])

     points_on_proc = np.array(points_on_proc,dtype=np.float64)
     # points_on_proc = np.array(points_on_proc,dtype=np.float64)

     return points_on_proc,cells
     # disp = self.uh.sub(0).eval(points_on_proc,cells)
     # rot = self.uh.sub(1).eval(points_on_proc,cells)

def mat_to_mesh(filename,aux_data=None, plot_xs = False ):
     mat = scipy.io.loadmat(filename)
     data = []
     for item in aux_data:
          data.append(mat[item])

     elems = mat['vabs_2d_mesh_elements']
     nodes = mat['vabs_2d_mesh_nodes']
     print('Number of nodes:')
     print(len(nodes))
     print('Number of Elements:')
     print(len(elems))
     elems -=1

     cells = {'triangle':elems[:,0:3]}
     meshio.write_points_cells('file.xdmf',nodes,cells,file_format='xdmf')

     with XDMFFile(MPI.COMM_WORLD, 'file.xdmf', "r") as xdmf:
          msh = xdmf.read_mesh(name='Grid')

     if plot_xs:

          msh.topology.create_connectivity(msh.topology.dim-1, 0)

          plotter = pyvista.Plotter()
          num_cells_local = msh.topology.index_map(msh.topology.dim).size_local
          topology, cell_types, x = plot.create_vtk_mesh(msh, msh.topology.dim, np.arange(num_cells_local, dtype=np.int32))

          grid = pyvista.UnstructuredGrid(topology, cell_types, x)
          plotter.add_mesh(grid,show_edges=True)
          plotter.show_axes()

          if True:
               # Add labels to points on the yz plane (where x == 0)
               points = grid.points
               # mask = points[:, 0] == 0
               data=points - np.tile([[0,np.min(points[:,1]),0]],(points.shape[0],1))
               m_to_in = 39.37
               plotter.add_point_labels(points, (m_to_in*data).tolist())

               # plotter.camera_position = [(-1.5, 1.5, 3.0), (0.05, 0.6, 1.2), (0.2, 0.9, -0.25)]
          plotter.add_points(np.array((0,0,0)))

          plotter.show()
          
     if aux_data is not None:
          return msh,data
     else:
          return msh
     

"""
    Project function does not work the same between legacy FEniCS and FEniCSx,
    so the following project function must be defined based on this forum post:
    https://fenicsproject.discourse.group/t/problem-interpolating-mixed-function-dolfinx/4142/6
    and inspired by Ru Xiang's Shell module project function
    https://github.com/RuruX/shell_analysis_fenicsx/blob/b842670f4e7fbdd6528090fc6061e300a74bf892/shell_analysis_fenicsx/utils.py#L22
    """

def project(v, target_func, bcs=[]):
    # Ensure we have a mesh and attach to measure
    V = target_func.function_space

    # Define variational problem for projection
    w = TestFunction(V)
    Pv = TrialFunction(V)
    a = inner(Pv, w) * dx
    L = inner(v, w) * dx

    # Assemble linear system
    A = assemble_matrix(form(a), bcs)
    A.assemble()
    b = assemble_vector(form(L))
    apply_lifting(b, [form(a)], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setOperators(A)
    solver.solve(b, target_func.vector)

def sparseify(mat,sparse_format='csr'):
     lim = np.finfo(float).eps
     mat.real[abs(mat.real) < lim] = 0.0
     if sparse_format == 'csc':
          return csc_matrix(mat)
     elif sparse_format == 'csr':
          return csr_matrix(mat)
     

def gmsh_to_xdmf(mesh, cell_type, prune_z=False):
     cells = mesh.get_cells_type(cell_type)
     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
     points = mesh.points[:,:2] if prune_z else mesh.points
     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
     return out_mesh