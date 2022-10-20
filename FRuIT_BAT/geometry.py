import gmsh
from dolfinx.io import gmshio,XDMFFile
from mpi4py import MPI

def beamIntervalMesh3D(pts,lc,filename,meshname):

    gdim = 3
    tdim = 1

    gmsh.initialize()

    #construct line in 3D space
    gmsh.model.add("Beam")
    gmsh.model.setCurrent("Beam")

    for pt in pts:
        p1 = gmsh.model.occ.addPoint(pt[0],pt[1],pt[2],lc)
    for i in range(len(pts)):
        line = gmsh.model.occ.addLine(pts[i],pts[i])

    # Synchronize OpenCascade representation with gmsh model
    gmsh.model.occ.synchronize()

    # add physical marker
    gmsh.model.add_physical_group(tdim,[line])

    #generate the mesh and optionally write the gmsh mesh file
    gmsh.model.mesh.generate(gdim)
    # gmsh.write("output/beam_mesh.msh")

    #use meshio to convert msh file to xdmf
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0)
    msh.name = meshname
    # cell_markers.name = f"{msh.name}_cells"
    # facet_markers.name = f"{msh.name}_facets"

    #write xdmf mesh file
    with XDMFFile(msh.comm, f"output/beam_mesh.xdmf", "w") as file:
        file.write_mesh(msh)

    # close gmsh API
    gmsh.finalize()