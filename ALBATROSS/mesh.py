import numpy as np
from dolfinx import mesh,fem,plot
import ufl
import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI
import meshio
from ALBATROSS.utils import gmsh_to_xdmf,get_pts_and_cells
import pyvista
from petsc4py import PETSc

def smooth_mesh(msh, moved_nodes, displacement, nodes_to_move,plot_result=False,get_deriv=False):
     '''Function to apply elliptic smoothing to a mesh
     given a prescribed boundary motion
     
     msh: mesh to be smoothed
     moved_nodes: indices of nodes 
     displacement: 
     nodes_to_move:   
     '''

     c_el = msh.ufl_domain().ufl_coordinate_element()
     V = fem.functionspace(msh, c_el)

     uh = fem.Function(V)
     u_bc = fem.Function(V)
     # use u_bc.x.index_map.local_range
     # for i in u_bc.x.index_map.local_range:
     #      moved_dofs.extend(moved_nodes+i)
     moved_dofs = []
     dofs_to_move = []
     for i in range(V.num_sub_spaces):
          # _,dofmap = V.sub(i).collapse()
          # moved_dofs.extend([dofmap[j] for j in moved_nodes])
          # dofs_to_move.extend([dofmap[j] for j in nodes_to_move])
          moved_dofs.extend(fem.locate_dofs_topological(V.sub(i),0,moved_nodes))
          dofs_to_move.extend(fem.locate_dofs_topological(V.sub(i),0,nodes_to_move))
     # moved_dofs = fem.locate_dofs_topological(V.sub(0),0,moved_nodes)
     # dofs_to_move = fem.locate_dofs_topological(V,0,nodes_to_move)

     u_bc.vector.array[moved_dofs] += displacement.T.flatten()
     bc = fem.dirichletbc(u_bc,moved_nodes)
     
     # msh.geometry.x[moved_nodes,0:2] += displacement

     bcs = [bc]
          
     #TODO: need to account for case where not all exterior nodes are moved
     fem.petsc.set_bc(uh.vector, bcs)
     u = ufl.TrialFunction(V)
     v = ufl.TestFunction(V)
     a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
     L = ufl.inner(fem.Constant(msh, (0., 0.)), v)*ufl.dx
     problem = fem.petsc.LinearProblem(a, L, bcs, uh)
     problem.solve()
     deformation_array = uh.x.array.reshape((-1, msh.geometry.dim))
     new_mesh_coords = msh.geometry.x[nodes_to_move, 0:2] + deformation_array[nodes_to_move,0:2]
     
     if get_deriv is True:
          #TODO: compute only on boundary nodes (currenly computed, then restricted)
          #TODO: compute entries other than 0,0
          #compute deriv of interior disp w.r.t. boundary nodes
          X = ufl.SpatialCoordinate(msh)
          
          # #THESE ARE SHAPE DERIVATIVES, not NODAL SENSITIVITIES
          # duh_form = [[uh[idx1,idx2]*ufl.dx for idx1 in range(2)] for idx2 in range(2)]
          # args = duh_form[0][0].arguments()
          # n = max(a.number() for a in args) if args else -1
          # du=ufl.Argument(V,n+1)
          # duhdx_form = [[ufl.derivative(duh_form[idx1][idx2],X,du) for idx1 in range(2)] for idx2 in range(2)]
          # duhdx = np.array([[fem.petsc.assemble_vector(fem.form(duhdx_form[idx1][idx2]))
          #                     for idx1 in range(2)] for idx2 in range(2)])
          
          #assemble the unmodified stiffness matrix (prior to boundary condition application where rows/columns are zeroed out)
          A = fem.petsc.assemble_matrix(fem.form(a))
          A.assemble()
          
          dofs_to_move_is = PETSc.IS().createGeneral(dofs_to_move, comm=MPI.COMM_WORLD)
          moved_dofs_is = PETSc.IS().createGeneral(moved_dofs, comm=MPI.COMM_WORLD)

          # Create submatrices
          A_II = A.createSubMatrix(dofs_to_move_is, dofs_to_move_is)
          A_IB = A.createSubMatrix(dofs_to_move_is, moved_dofs_is)
                    
          # Create the inverse matrix as a dense matrix
          A_II_inv = PETSc.Mat().createDense(A_II.getSize())
          A_II_inv.setUp()
          A_II_inv.assemble()

          # Create vectors for solving
          b = PETSc.Vec().createSeq(A_II.size[0])  # RHS vector
          x = PETSc.Vec().createSeq(A_II.size[0])  # Solution vector

          # Create a KSP solver
          ksp = PETSc.KSP().create()
          ksp.setOperators(A_II)
          ksp.setType('preonly')  # Direct solve
          ksp.getPC().setType('lu')  # LU decomposition

          # Compute each column of the inverse
          for i in range(A_II.size[0]):
               b.set(0.0)  # Reset RHS
               b[i] = 1.0  # Set the i-th standard basis vector
               b.assemble()
               
               # Solve for the i-th column of the inverse
               ksp.solve(b, x)
               x.assemble()

               # Insert the solution as the i-th column of A_inv
               A_II_inv.setValues(range(A_II.size[0]), [i], x)  # Directly set the entire column
               
          A_II_inv.assemble()
          # A_II_inv.view()
          # print("these are some words....")
          
          # A_IB.view()
          # print("these are also words....")
          # A.view()

          # for i in range(A_II.size[0]):
          #      e = PETSc.Vec().createSeq(A_II.size[0])
          #      e.setValue(i, 1.0)
          #      e.assemble()
          #      identity.setColumn(i, e)

          # A_II_inv = PETSc.Mat().createDense([A_II.size[0], A_II.size[0]])
          # A_II_inv.setUp()
          # ksp = PETSc.KSP().create(MPI.COMM_WORLD)
          # for i in range(A_II.size[0]):
          #      rhs = identity.getColumnVector(i)
          #      solution = A_II.createVecRight()
          #      ksp.solve(rhs, solution)
          #      A_II_inv.setColumn(i, solution)

          # A_II_inv.assemble()

          # Compute Jacobian: -A_II^-1 * A_IB
          J = A_II_inv.matMult(A_IB)
          J.scale(-1.0)
          # print("These are nearly the same words...")
          # J.view()
          duhdx = J.getDenseArray()

          # I = ufl.Identity(2)
          # F = I+ufl.grad(uh)
          # J = ufl.det(F) #this is the jacobian determinant, not the jacobian

          # F[0,0]

          # print("Jacobian ufl shape:",J.ufl_shape)
          #NODAL SENSITIVIES:
          #these are computed by interpolating a ufl expression for the derivative
          #  of the displacements wrt to the nodal locations into the appropriate
          #  function space. This is procedurally (software-wise) different from
          #  the Gateaux derivatives used for the spatial derivatives.

          #derivative of displacement w.r.t. mesh nodes
          # grad_uh_ufl = ufl.grad(uh)

          # grad_uh_ufl = ufl.derivative(uh[0]*ufl.dx,uh)

          # grad_uh_form = fem.petsc.assemble_vector(fem.form(grad_uh_ufl))
          
          # #Construct expression to evalute
          # Vd = fem.functionspace(msh,('CG',1,(2,2)))
          # grad_uh = fem.Function(Vd)
          # grad_uh.interpolate(fem.Expression(
          #                     grad_uh_ufl,
          #                     Vd.element.interpolation_points()
          #                     ) )
          
          # points_on_proc,cells=get_pts_and_cells(msh,msh.geometry.x)
          # duhdx = grad_uh.eval(points_on_proc,cells)
          
          

          # all_dofs= 
          # dofs_to_move = all_dofs[~np.isin(alldofs,moved_dofs)]

          # duhdx = duhdx[:,:,dofs_to_move]

          return new_mesh_coords,duhdx

     if plot_result is True:
          msh.geometry.x[:,0:2] += deformation_array

          #plot mesh
          pyvista.global_theme.background = [255, 255, 255, 255]
          pyvista.global_theme.font.color = 'black'
          tdim = msh.topology.dim
          topology, cell_types, geometry = plot.vtk_mesh(msh, tdim)
          grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
          plotter = pyvista.Plotter()
          plotter.add_mesh(grid, show_edges=True,opacity=0.25)
          plotter.view_xy()
          plotter.show_axes()
          plotter.show_bounds()
          if not pyvista.OFF_SCREEN:
               plotter.show()

     return new_mesh_coords

def beam_interval_mesh_3D(pts,ne,meshname):
     '''
     pts = list of nx (x,y,z) locations of a beam nodes (np)
     ne = list of number of elements for each segment between nodes (np-1)
     meshname = name of mesh
     '''
     filename = 'output/'+meshname+'.xdmf'
     print('points shape:')
     print(np.array(pts).shape)
     print('element number shape')
     print(np.array(ne).shape)
     gdim = 3
     tdim = 1

     gmsh.initialize()
     gmsh.option.setNumber("General.Terminal",0) #hide meshing output

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
     print('num pts:')
     print(pt_tags)
     print(len(pt_tags))
     print('num lines:')
     print(line_tags)
     print(len(line_tags))
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
     gmsh.fltk.run()
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
     

def create_T_section(dims,num_el,meshname):
     '''
     dims = [height,width,flange,web]
     num_el = [numel_flange,numel_web]
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
     print("Generating 2D T-section mesh...")
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
               #    [-W2,-H2,-tw2,-H2mtf],
                  [-tw2,-H2,tw2,-H2mtf]]
               #    [tw2,-H2,W2,-H2mtf]]
     
     #number of elements in x and y directions for each sub-section
     box_el_num = [[nw,nf],
                    [nweb,nf],
                    [nw,nf],
                    [nweb,nh],
                    # [nw,nf],
                    [nweb,nf]]
                    # [nw,nf]]
     
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
     
