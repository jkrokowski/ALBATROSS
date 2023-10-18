'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.

Dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from dolfinx.fem import TensorFunctionSpace,VectorFunctionSpace,Expression,Function,Constant, locate_dofs_geometrical,locate_dofs_topological,dirichletbc,form
# from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import (LinearProblem,assemble_matrix,assemble_vector, 
                                apply_lifting,set_bc,create_vector)
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross,Measure,sin,cos)
from FROOT_BAT.elements import *
from FROOT_BAT.cross_section import CrossSection
from FROOT_BAT.axial import Axial
from FROOT_BAT.utils import get_pts_and_cells
from petsc4py.PETSc import ScalarType
import numpy as np
import pyvista
from petsc4py import PETSc
from dolfinx.plot import create_vtk_mesh

class BeamModel(Axial):
    '''
    Class that combines both 1D and 2D analysis
    '''

    def __init__(self,axial_mesh,xc_info):
        '''
        axial_mesh: mesh used for 1D analysis (often this is a finer
            discretization than the 1D mesh used for locating xcs)

        info2D = (xcs,mats,axial_pos_mesh,xc_orientations)
            xcs : list of 2D xdmf meshes for each cross-section
            mats : list of materials used corresponding to each XC
            axial_pos_mesh : the beam axis discretized into the number of
                elements with nodes at the spanwise locations of the XCs
            xc_orientations: list of vectors defining orientation of 
                horizontal axis used in xc analysis
        '''
        self.axial_mesh = axial_mesh
        [self.xc_meshes, self.mats, self.axial_pos_mesh, self.orientations] = xc_info
        self.numxc = len(self.xc_meshes)

        #define rotation matrices for rotating between beam reference axis and xc reference frame
        #note: 
        #   RGB=Rgb and RBG=Rbg
        #   "mixed case" frame transformations are not equivalent as RBb != identity(3x3)

        self.RBG = np.array([[ 0,  0,  1],
                             [ 1,  0,  0],
                             [ 0,  1, 0]])
        self.RGB = np.array([[ 0,  -1,  0],
                             [ 0,  0, 1],
                             [ -1,  0,  0]])
       
        print("Orienting XCs along beam axis....")
        self.get_xc_orientations_for_1D()

        print("Getting XC Properties...")
        self.get_axial_props_from_xcs()

        print("Initializing Axial Model (1D Analysis)")
        super().__init__(self.axial_mesh,self.k,self.o)

        print("Computing Elastic Energy...")
        self.elastic_energy()

    def get_xc_orientations_for_1D(self):
        #define orientation of the x2 axis w.r.t. the beam axis x1 to allow for 
        # matching the orientation of the xc with that of the axial mesh
        # (this must be done carefully and with respect to the location of the beam axis)
        self.O2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=3)
        self.o2 = Function(self.O2)
        self.o2.vector.array = np.array(self.orientations)
        self.o2.vector.destroy() #needed for PETSc garbage collection

        #interpolate these orientations into the finer 1D analysis mesh
        self.O = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=3)
        self.o = Function(self.O)
        self.o.interpolate(self.o2)

    
    def get_axial_props_from_xcs(self):
        '''
        CORE FUNCTION FOR PROCESSING MULTIPLE 2D XCs TO PREPARE A 1D MODEL
        '''
        # (mesh1D_2D,(meshes2D,mats2D)) = info2D 
        # mesh1D_1D = info1D
        self.xcs = []
        # K_list = []
        
        def get_flat_sym_stiff(K_mat):
            K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
            return K_flat
        
        sym_cond = False #there is an issue with symmetric tensor fxn spaces in dolfinx at the moment
        K2 = TensorFunctionSpace(self.axial_pos_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        k2 = Function(K2)
        #TODO:same process for mass matrix
        A2 = FunctionSpace(self.axial_pos_mesh,('CG',1))
        a2 = Function(A2)
        C2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=2)
        c2 = Function(C2)
        
        for i,[mesh2d,mat2D] in enumerate(zip(self.xc_meshes,self.mats)):
            print('    computing properties for XC '+str(i+1)+'/'+str(self.numxc)+'...')
            #instantiate class for cross-section i
            self.xcs.append(CrossSection(mesh2d,mat2D))
            #analyze cross section
            self.xcs[i].getXCStiffnessMatrix()

            #output stiffess matrix
            if sym_cond==True:
                #need to add fxn
                print("symmetric mode not available yet")
                exit()
                k2.vector.array[21*i,21*(i+1)] = self.xcs[i].K.flatten()
            elif sym_cond==False:
                k2.vector.array[36*i:36*(i+1)] = self.xcs[i].K.flatten()
                a2.vector.array[i] = self.xcs[i].A
                c2.vector.array[2*i:2*(i+1)] = [self.xcs[i].yavg,self.xcs[i].zavg]
        print("Done computing cross-sectional properties...")
        # if sym_cond==True:
        #     K_entries = np.concatenate([get_flat_sym_stiff(K_list[i]) for i in range(self.num_xc)])
        # elif sym_cond == False:
        #     K_entries = np.concatenate([K_list[i].flatten() for i in range(self.num_xc)])
        
        # k2.vector.array = K_entries
        print("Interpolating cross-sectional properties to axial mesh...")
        #interpolate from axial_pos_mesh to axial_mesh 
        self.K = TensorFunctionSpace(self.axial_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        self.k = Function(self.K)
        self.k.interpolate(k2)

        self.A = FunctionSpace(self.axial_mesh,('CG',1))
        self.a = Function(self.A)
        self.a.interpolate(a2)

        self.V = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=2)
        self.c = Function(self.V)
        self.c.interpolate(c2)
        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        k2.vector.destroy()     #need to add to prevent PETSc memory leak 
        a2.vector.destroy()
        c2.vector.destroy()
        print("Done interpolating cross-sectional properties to axial mesh...")
    
    # def get_3D_disp(self):
    #     '''
    #     returns a fxn defined over a 3D mesh generated from the 
    #     2D xc's and the 1D analysis mesh
    #     '''
    #     return
    
    def plot_xc_orientations(self):
        warp_factor = 1

        #plot Axial mesh
        tdim = self.axial_mesh.topology.dim
        topology, cell_types, geom = create_vtk_mesh(self.axial_mesh,tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        plotter = pyvista.Plotter()
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k",line_width=5)
        actor_1 = plotter.add_mesh(grid, style='points',color='k',point_size=12)
        grid.point_data["u"]= self.o.x.array.reshape((geom.shape[0],3))
        glyphs = grid.glyph(orient="u",factor=.25)
        actor_2 = plotter.add_mesh(glyphs,color='b')

        #plot xc placement mesh
        tdim = self.axial_pos_mesh.topology.dim
        topology2, cell_types2, geom2 = create_vtk_mesh(self.axial_pos_mesh,tdim)
        grid2 = pyvista.UnstructuredGrid(topology2, cell_types2, geom2)
        actor_3 = plotter.add_mesh(grid2, style="wireframe", color="r")
        actor_4 = plotter.add_mesh(grid2, style='points',color='r')
        grid2.point_data["u"]= self.o2.x.array.reshape((geom2.shape[0],3))
        glyphs2 = grid2.glyph(orient="u",factor=0.5)
        actor_5 = plotter.add_mesh(glyphs2,color='g')

        plotter.view_isometric()
        plotter.show_axes()

        # if not pyvista.OFF_SCREEN:
        plotter.show()
        # else:
        #     pyvista.start_xvfb()
        #     figure = plot.screenshot("beam_mesh.png")

    def recover_displacement(self,plot_xcs=True):
        #get local displacements
        [u_local,theta_local] = self.get_local_disp(self.axial_pos_mesh.geometry.x)
                
        def apply_disp_to_xc(xc,u_local):
            numdofs = int(xc.xcdisp.x.array.shape[0]/3)
            xc.xcdisp.vector.array += np.tile(self.RGB@u_local,numdofs)
            
            #needed for PETSc garbage collection
            xc.xcdisp.vector.destroy()

        def apply_rot_to_xc(xc,theta_local):
            def rotation_to_disp(x):
                [[alpha],[beta],[gamma]] = self.RGB@theta_local.reshape((3,1))
                # rotation about X-axis
                Rx = np.array([[1,         0,         0],
                                [0,cos(alpha),-sin(alpha)],
                                [0,sin(alpha),cos(alpha)]])
                # rotation about Y-axis
                Ry = np.array([[cos(beta), 0,sin(beta)],
                                [0,         1,        0],
                                [-sin(beta),0,cos(beta)]])
                #rotation about Z-axis
                Rz = np.array([[cos(gamma),-sin(gamma),0],
                                [sin(gamma),cos(gamma), 0],
                                [0,         0,          1]])

                #TODO: think about how this rotation matrix could be stored?
                # 3D rotation matrix for applying twist and transverse disp
                RBb = Rz@Ry@Rx

                #centroid location in g frame (same as RgB@np.array([0,yavg,zavg]))
                centroid = np.array([[xc.yavg,xc.zavg,0]]).T

                return ((RBb@(x-centroid)-x)+centroid)

            xc.xcdisp.interpolate(rotation_to_disp)

        #compute xc displacement functions
        for i,xc in enumerate(self.xcs):
            self.xcs[i].V = VectorFunctionSpace(self.xcs[i].msh,('CG',1),dim=3)
            self.xcs[i].xcdisp = Function(self.xcs[i].V)

            
            apply_rot_to_xc(self.xcs[i],theta_local[i])
            apply_disp_to_xc(self.xcs[i],u_local[i])

            if plot_xcs:
                pyvista.global_theme.background = [255, 255, 255, 255]
                pyvista.global_theme.font.color = 'black'
                tdim = xc.msh.topology.dim
                topology, cell_types, geom = create_vtk_mesh(xc.msh, tdim)
                grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
                # grid = pyvista.UnstructuredGrid(topology, cell_types, geom).rotate_z(90).rotate_y(90)
                plotter = pyvista.Plotter()

                plotter.add_mesh(grid, show_edges=True,opacity=0.25)
                # grid.rotate_z(90).rotate_y(90)
                # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
                # have to be careful about how displacement data is populated into grid before or after rotations for visualization
                grid.point_data["u"] = xc.xcdisp.x.array.reshape((geom.shape[0],3))
                # new_grid = grid.transform(transform_matrix)

                warped = grid.warp_by_vector("u", factor=1)
                actor_1 = plotter.add_mesh(warped, show_edges=True)
                plotter.show_axes()
                # if add_nodes==True:
                #     plotter.add_mesh(grid, style='points')
                plotter.view_isometric()
                if not pyvista.OFF_SCREEN:
                    plotter.show()

            # def compute_

    def plot_xc_disp_3D(self):
        warp_factor = 1

        tdim = self.axial_mesh.topology.dim
        topology, cell_types, geom = create_vtk_mesh(self.axial_mesh,tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        plotter = pyvista.Plotter()

        grid.point_data["u"] = self.uh.sub(0).collapse().x.array.reshape((geom.shape[0],3))
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
        warped = grid.warp_by_vector("u", factor=warp_factor)
        actor_1 = plotter.add_mesh(warped)
        # plotter.add_scalar_bar('u',intesractive=True)
        self.uh.vector.destroy()
        # self.plot_axial_displacement()
        
        #plot xc meshes:
        print(self.axial_pos_mesh.geometry.x)
        grids = []
        RbA = self.get_local_basis(self.axial_pos_mesh.geometry.x)
        for i,xc in enumerate(self.xcs):
            # print(self.axial_pos_mesh.geometry.x[i])
            # #compute translation vector (transform centroid offset to relevant coordinates)
            RbA = self.get_local_basis(self.axial_pos_mesh.geometry.x)

            trans_vec = np.array([self.axial_pos_mesh.geometry.x[i]]).T-RbA[i,:,:].T@(np.array([[0,xc.yavg,xc.zavg]]).T)
            
            transform_matrix=np.concatenate((np.concatenate([RbA[i,:,:].T@self.RBG,trans_vec],axis=1),np.array([[0,0,0,1]])))
            # # transform_matrix2=np.concatenate((np.concatenate([rot_mat_xc_to_axial,trans_vec.T],axis=1),np.array([[0,0,0,1]])))
            # print("transform matrix for plotting:")
            # print(transform_matrix)
            tdim = xc.msh.topology.dim
            topology2, cell_types2, geom2 = create_vtk_mesh(xc.msh, tdim)
            grids.append(pyvista.UnstructuredGrid(topology2, cell_types2, geom2))

            grids[i].transform(transform_matrix)
            grids[i].point_data["u"] = (RbA[i,:,:].T@self.RBG@xc.xcdisp.x.array.reshape((geom2.shape[0],3)).T).T

            actor2=plotter.add_mesh(grids[i], show_edges=True,opacity=0.25)

            warped = grids[i].warp_by_vector("u", factor=warp_factor)
            actor_3 = plotter.add_mesh(warped, show_edges=True)
            # print(xc.xcdisp.x.array)
            
        plotter.view_xz()
        plotter.show_axes()

        # if not pyvista.OFF_SCREEN:
        plotter.show()
        # else:
        #     pyvista.start_xvfb()
        #     figure = plot.screenshot("beam_mesh.png")