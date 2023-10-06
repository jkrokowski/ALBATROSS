import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI

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