import numpy as np
from dolfinx import mesh
import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI
import meshio
from ALBATROSS.utils import gmsh_to_xdmf

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

def create_hollow_box(pts,thicknesses,num_el,meshname):
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
     
def create_I_section(dims,num_el,meshname):
     '''
     dims = [height,width,flange1,flange2,web]
     num_el = [numel_flange1,numel_flange2,numel_web]
     meshname = name of mesh
     '''

     #unpack input 
     [H,W,tf,tw]=dims
     [nf,nweb]=num_el
     filename = 'output/'+meshname+'.xdmf'
  
     #choose number of elements between corners for height and width:
     avg_cell_size = np.average(np.array([tf,tw])/np.array(num_el))

     nw = int(np.ceil((0.5*(W-tw))/avg_cell_size))
     nh = int(np.ceil((H-2*tf)/avg_cell_size))
     
     #for 2D xs mesh, gdim=tdim=2
     gdim = 2
     tdim = 2

     #initialize, add model and activate
     print("Generating 2D I-section mesh...")
     gmsh.initialize()
     gmsh.option.setNumber("General.Terminal",0) #suppress gmsh output
     gmsh.model.add(meshname)
     gmsh.model.setCurrent(meshname)
     
     #repeated dims
     H2 = H/2
     W2 = W/2
     H2mtf = H/2 - tf
     tw2 = tw/2
     #list of coordinates for each sub-section of I-section
     box_nodes = [[-W2,H2mtf,-tw2,H2],
                  [-tw2,H2mtf,tw2,H2],
                  [tw2,H2mtf,W2,H2],
                  [-tw2,-H2mtf,tw2,H2mtf],
                  [-W2,-H2,-tw2,-H2mtf],
                  [-tw2,-H2,tw2,-H2mtf],
                  [tw2,-H2,W2,-H2mtf]]
     
     #number of elements in x and y directions for each sub-section
     box_el_num = [[nw,nf],
                    [nweb,nf],
                    [nw,nf],
                    [nweb,nh],
                    [nw,nf],
                    [nweb,nf],
                    [nw,nf]]
     
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
     
