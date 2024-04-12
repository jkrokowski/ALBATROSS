'''
The beam_model module
---------------------
This module connects an Axial model and (>=1) Cross-Section models
allowing for a complete beam model to be constructed, analyzed,
and (given the appropriate cross-section) a full 3D displacement and 
stress solution field to be obtained

'''

from dolfinx.fem import TensorFunctionSpace,VectorFunctionSpace,Expression,Function,Constant, locate_dofs_geometrical,locate_dofs_topological,dirichletbc,form
from dolfinx.fem.petsc import (LinearProblem,assemble_matrix,assemble_vector, 
                                apply_lifting,set_bc,create_vector)
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross,Measure,sin,cos)
from ALBATROSS.elements import *
from ALBATROSS.cross_section import CrossSection,CrossSectionAnalytical
from ALBATROSS.axial import Axial
from ALBATROSS.utils import get_pts_and_cells
from petsc4py.PETSc import ScalarType
import numpy as np
import pyvista
from dolfinx.plot import create_vtk_mesh

class BeamModel(Axial):
    '''
    Class that combines both 1D and 2D analysis
    '''

    def __init__(self,axial_mesh,xs_info,xs_type='EBPE',segment_type='CONSTANT'):
        '''
        axial_mesh: mesh used for 1D analysis (typically this is a finer
            discretization than the 1D mesh used for locating xss)

        xs_info = (xs_list,axial_pos_mesh,xs_orientations,xs_adj_list)
            xs_list : list of 2D xdmf meshes for each cross-section
            axial_pos_mesh : the beam axis discretized into the number of
                elements with nodes at the spanwise locations of the XSs
            xs_orientations: list of vectors defining orientation of 
                primary orthogonal axis used in xs analysis
            xs_adj_list: adjacency list mapping segment[i] of the axial position
                mesh to xs[j] of xs_list. This allows cross-section analysis to 
                be run once for each cross-section and then mapped to each section
        OPTIONAL ARGS: 
        segment_type: determines whether interpolation or 
            constant properties are used within segments
        xs_type: determines if xs analysis is to be run. 

        '''
        self.axial_mesh = axial_mesh

        #define rotation matrices for rotating between beam reference axis and xs reference frame
        #note: 
        #   RGB=Rgb and RBG=Rbg
        #   "mixed case" frame transformations are not equivalent as RBb != identity(3x3)
        self.RBG = np.array([[ 0,  0,  1],
                             [ 1,  0,  0],
                             [ 0,  1, 0]])
        self.RGB = np.array([[ 0,  1,  0],
                             [ 0,  0, 1],
                             [ 1,  0,  0]])
        
        self.segment_type = segment_type
        
        if xs_type == 'EBPE':
            #EBPE: Energy Based Polynomial Expansion
            [self.xs_list, self.axial_pos_mesh, self.orientations,self.xs_adj_list] = xs_info
            self.numxs = len(self.xs_list)
            self.numsegments = len(self.xs_adj_list)
            
            print("Orienting XSs along beam axis....")
            self.get_xs_orientations_for_1D()

            print("Getting XS Properties...")
            self.get_axial_props_from_xss()

        elif xs_type == 'precomputed':
            #For usage with fully populated beam constitutive matrices
            #   for example, from VABS
            [self.K_list,self.axial_pos_mesh,self.orientations] = xs_info
            self.numxs = len(self.K_list)

            print("Orienting XSs along beam axis....")
            self.get_xs_orientations_for_1D()

            print("Reading XS Stiffness Matrices...")
            self.get_axial_props_from_K_list()

        elif xs_type== 'analytical':
            # xs_info consists of a list of dictionaries of the relevant xs parameter
            [self.xs_params,self.axial_pos_mesh,self.orientations] = xs_info
            self.numxs = len(self.xs_params)

            print("Orienting XSs along beam axis....")
            self.get_xs_orientations_for_1D()

            print("Computing analytical cross section properties ...")
            self.get_axial_props_from_analytic_formulae()
        else:
            print("please use one of the documented methods")
        
        print("Initializing Axial Model (1D Analysis)")
        super().__init__(self.axial_mesh,self.k,self.o)

        print("Computing Elastic Energy...")
        self.elastic_energy()

    def get_xs_orientations_for_1D(self):
        #define orientation of the x2 axis w.r.t. the beam axis x1 to allow for 
        # matching the orientation of the xs with that of the axial mesh
        # (this must be done carefully and with respect to the location of the beam axis)
        #determine how the segments are constructed
        if self.segment_type == "CONSTANT":
            element_type = ('DG',0)
            # num_vals_to_enter = self.numsegments

        elif self.segment_type == "VARIABLE":
            element_type = ('CG',1)
            # num_vals_to_enter = self.numsegments + 1
            # self.orientations=self.orientations.append(self.orientations)
        
        self.O2 = VectorFunctionSpace(self.axial_pos_mesh,element_type,dim=3)
        self.o2 = Function(self.O2)
        self.o2.vector.array = np.array(self.orientations)
        self.o2.vector.destroy() #needed for PETSc garbage collection

        #interpolate these orientations into the finer 1D analysis mesh
        self.O = VectorFunctionSpace(self.axial_mesh,element_type,dim=3)
        self.o = Function(self.O)
        self.o.interpolate(self.o2)

    def get_axial_props_from_xss(self):
        '''
        CORE FUNCTION FOR PROCESSING MULTIPLE 2D XSs TO PREPARE A 1D MODEL
        '''

        # def get_flat_sym_stiff(K_mat):
        #     K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
        #     return K_flat
        print("Linking cross-sectional properties to axial mesh...")
        
        #determine how the segments are constructed
        if self.segment_type == "CONSTANT":
            element_type = ('DG',0)
            num_vals_to_enter = self.numsegments
        elif self.segment_type == "VARIABLE":
            element_type = ('CG',1)
            num_vals_to_enter = self.numsegments + 1

        #We need to construct a continuous field over the axial mesh 
        #   from the properties computed from each cross-section
        sym_cond = False #there is an issue with symmetric tensor fxn spaces in dolfinx at the moment
        # T2_66 = TensorFunctionSpace(self.axial_pos_mesh,element_type,shape=(6,6),symmetry=sym_cond)
        # k2 = Function(T2_66)
        # S2 = FunctionSpace(self.axial_pos_mesh,element_type)
        # linear_density = Function(S2)
        # # TODO: should this be a dim=3 vector? mght be easier to transform btwn frames?
        # V2_2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=2)
        # c2 = Function(V2_2)
        
        # for i,[mesh2d,mat2D] in enumerate(zip(self.xs_meshes,self.mats)):
        #     print('    computing properties for XS '+str(i+1)+'/'+str(self.numxs)+'...')
        #     #instantiate class for cross-section i
        #     self.xss.append(CrossSection(mesh2d,mat2D))
        #     #analyze cross section
        #     self.xss[i].getXSStiffnessMatrix()

        #initialize functions and functionspaces over axial positioning mesh            
        T2_66 = TensorFunctionSpace(self.axial_pos_mesh,element_type,shape=(6,6),symmetry=sym_cond)
        k2 = Function(T2_66)
        S2 = FunctionSpace(self.axial_pos_mesh,element_type)
        linear_density = Function(S2)
        print(num_vals_to_enter)
        #population cross-sectional properties over axial positioning mesh
        for i in range(num_vals_to_enter):
            #TODO: think a bit more about how to build up the xs properties over the beam
            xs_idx =  self.xs_adj_list[i][0]
            xs=self.xs_list[xs_idx]
            #output stiffess matrix
            if sym_cond==True:
                print("symmetric mode not available yet,try again soon")
                exit()
                k2.vector.array[21*i,21*(i+1)] = xs.K.flatten()
            elif sym_cond==False:
                k2.vector.array[36*i:36*(i+1)] = xs.K.flatten()
                linear_density.vector.array[i]
                # a2.vector.array[i] = xs.A
                # rho2.vector.array[i] = xs.rho
                # c2.vector.array[2*i:2*(i+1)] = [self.xss[i].yavg,self.xss[i].zavg]

        #interpolate from axial_pos_mesh to axial_mesh 

        #initialize fxn spaces
        self.T_66 = TensorFunctionSpace(self.axial_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        self.S = FunctionSpace(self.axial_mesh,('CG',1))
        
        #interpolate beam constitutive matrix
        self.k = Function(self.T_66)
        self.k.interpolate(k2)

        #interpolate linear density area
        self.linear_density = Function(self.S)
        self.linear_density.interpolate(linear_density)

        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        k2.vector.destroy()     #need to add to prevent PETSc memory leak from garbage collection issues
        linear_density.vector.destroy()

        print("Done interpolating cross-sectional properties to axial mesh...")
    
    def get_axial_props_from_K_list(self):
        '''
        FUNCTION TO POPULATE TENSOR FXN SPACE WITH BEAM CONSTITUITIVE MATRIX 
        '''
               
        def get_flat_sym_stiff(K_mat):
            K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
            return K_flat
        
        sym_cond = False #there is an issue with symmetric tensor fxn spaces in dolfinx at the moment
        T2_66 = TensorFunctionSpace(self.axial_pos_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        k2 = Function(T2_66)
        #TODO:same process for mass matrix
        # S2 = FunctionSpace(self.axial_pos_mesh,('CG',1))
        # a2 = Function(S2)
        # rho2 = Function(S2)
        # # TODO: should this be a dim=3 vector? mght be easier to transform btwn frames?
        # V2_2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=2)
        # c2 = Function(V2_2)
        
        for i,K in enumerate(self.K_list):
            print('    reading properties for XS '+str(i+1)+'/'+str(self.numxs)+'...')
            #output stiffess matrix
            if sym_cond==True:
                #need to add fxn
                print("symmetric mode not available yet,try again soon")
                exit()
                k2.vector.array[21*i,21*(i+1)] = self.xss[i].K.flatten()
            elif sym_cond==False:
                k2.vector.array[36*i:36*(i+1)] = K.flatten()
                # a2.vector.array[i] = self.xss[i].A
                # rho2.vector.array[i] = self.xss[i].rho
                # c2.vector.array[2*i:2*(i+1)] = [self.xss[i].yavg,self.xss[i].zavg]

        print("Done reading cross-sectional properties...")

        print("Interpolating cross-sectional properties to axial mesh...")
        #interpolate from axial_pos_mesh to axial_mesh 

        #initialize fxn spaces
        self.T_66 = TensorFunctionSpace(self.axial_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        self.V_2 = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=2)
        self.S = FunctionSpace(self.axial_mesh,('CG',1))
        
        #interpolate beam constitutive matrix
        self.k = Function(self.T_66)
        self.k.interpolate(k2)

        # #interpolate xs area
        # self.a = Function(self.S)
        # self.a.interpolate(a2)

        # #interpolate xs density
        # self.rho = Function(self.S)
        # self.rho.interpolate(rho2)

        # #interpolate centroidal location in g frame
        # self.c = Function(self.V_2)
        # self.c.interpolate(c2)

        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        k2.vector.destroy()     #need to add to prevent PETSc memory leak from garbage collection issues
        # a2.vector.destroy()
        # c2.vector.destroy()
        # rho2.vector.destroy()

        print("Done interpolating cross-sectional properties to axial mesh...")
    
    def get_axial_props_from_analytic_formulae(self):
        def get_flat_sym_stiff(K_mat):
            K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
            return K_flat
        
        sym_cond = False #there is an issue with symmetric tensor fxn spaces in dolfinx at the moment
        T2_66 = TensorFunctionSpace(self.axial_pos_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        k2 = Function(T2_66)
        #TODO:same process for mass matrix
        # S2 = FunctionSpace(self.axial_pos_mesh,('CG',1))
        # a2 = Function(S2)
        # rho2 = Function(S2)
        # # TODO: should this be a dim=3 vector? mght be easier to transform btwn frames?
        # V2_2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=2)
        # c2 = Function(V2_2)
        self.xss = []
        for i,xs in enumerate(self.xs_params):
            print('    computing properties for ' + str(xs['shape'])+ ' XS: '+str(i+1)+'/'+str(self.numxs)+'...')
            #get stiffess matrix
            self.xss.append(CrossSectionAnalytical(xs))
            self.xss[i].compute_stiffness()
            print(self.xss[i].K)
            if sym_cond==True:
                #need to add fxn
                print("symmetric mode not available yet,try again soon")
                exit()
                k2.vector.array[21*i,21*(i+1)] = self.xss[i].K.flatten()
            elif sym_cond==False:
                k2.vector.array[36*i:36*(i+1)] = self.xss[i].K.flatten()
                # a2.vector.array[i] = self.xss[i].A
                # rho2.vector.array[i] = self.xss[i].rho
                # c2.vector.array[2*i:2*(i+1)] = [self.xss[i].yavg,self.xss[i].zavg]

        print("Done finding cross-sectional properties...")

        print("Interpolating cross-sectional properties to axial mesh...")
        #interpolate from axial_pos_mesh to axial_mesh 

        #initialize fxn spaces
        self.T_66 = TensorFunctionSpace(self.axial_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        # self.V_2 = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=2)
        # self.S = FunctionSpace(self.axial_mesh,('CG',1))
        
        #interpolate beam constitutive matrix
        self.k = Function(self.T_66)
        self.k.interpolate(k2)

        # #interpolate xs area
        # self.a = Function(self.S)
        # self.a.interpolate(a2)

        # #interpolate xs density
        # self.rho = Function(self.S)
        # self.rho.interpolate(rho2)

        # #interpolate centroidal location in g frame
        # self.c = Function(self.V_2)
        # self.c.interpolate(c2)

        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        k2.vector.destroy()     #need to add to prevent PETSc memory leak from garbage collection issues
        # a2.vector.destroy()
        # c2.vector.destroy()
        # rho2.vector.destroy()

        print("Done interpolating cross-sectional properties to axial mesh...")

    # def get_3D_disp(self):
    #     '''
    #     returns a fxn defined over a 3D mesh generated from the 
    #     2D xs's and the 1D analysis mesh
    #     '''
    #     return
    
    def plot_xs_orientations(self):
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        warp_factor = 1

        #plot Axial mesh
        tdim = self.axial_mesh.topology.dim
        topology, cell_types, geom = create_vtk_mesh(self.axial_mesh,tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        plotter = pyvista.Plotter()
        actor_0 = plotter.add_mesh(grid, style="wireframe", color="k",line_width=5)
        actor_1 = plotter.add_mesh(grid, style='points',color='k',point_size=12)
        if self.segment_type =='CONSTANT':
            grid.point_data["u"]= np.concatenate([self.o.x.array.reshape((geom.shape[0]-1,3)),[self.o.x.array[-3:]]],axis=0)
        else:
            grid.point_data["u"]= self.o.x.array.reshape((geom.shape[0],3))
        glyphs = grid.glyph(orient="u",factor=.25)
        actor_2 = plotter.add_mesh(glyphs,color='b')

        #plot xs placement mesh
        tdim = self.axial_pos_mesh.topology.dim
        topology2, cell_types2, geom2 = create_vtk_mesh(self.axial_pos_mesh,tdim)
        grid2 = pyvista.UnstructuredGrid(topology2, cell_types2, geom2)
        actor_3 = plotter.add_mesh(grid2, style="wireframe", color="r")
        actor_4 = plotter.add_mesh(grid2, style='points',color='r')
        if self.segment_type =='CONSTANT':
            grid2.point_data["u"]= np.concatenate([self.o2.x.array.reshape((geom2.shape[0]-1,3)),[self.o2.x.array[-3:]]],axis=0)
        else:
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

    def recover_displacement(self,plot_xss=True):
        #get local displacements
        [u_local,theta_local] = self.get_local_disp(self.axial_pos_mesh.geometry.x)
                
        def apply_disp_to_xs(xs,u_local):
            numdofs = int(xs.xsdisp.x.array.shape[0]/3)
            xs.xsdisp.vector.array += np.tile(self.RGB@u_local,numdofs)
            
            #needed for PETSc garbage collection
            xs.xsdisp.vector.destroy()

        def apply_rot_to_xs(xs,theta_local):
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
                RGg = Rz@Ry@Rx

                #centroid location in g frame (same as RgB@np.array([0,yavg,zavg]))
                centroid = np.array([[xs.yavg,xs.zavg,0]]).T

                return ((RGg@(centroid-x)+x)-centroid)
                # return ((RGg@(x-centroid)-x)+centroid)

            xs.xsdisp.interpolate(rotation_to_disp)

        #compute xs displacement functions
        for i,xs in enumerate(self.xs_list):
            xs.V = VectorFunctionSpace(xs.msh,('CG',1),dim=3)
            xs.xsdisp = Function(xs.V)

            
            apply_rot_to_xs(xs,theta_local[i])
            apply_disp_to_xs(xs,u_local[i])

            if plot_xss:
                pyvista.global_theme.background = [255, 255, 255, 255]
                pyvista.global_theme.font.color = 'black'
                tdim = xs.msh.topology.dim
                topology, cell_types, geom = create_vtk_mesh(xs.msh, tdim)
                grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
                # grid = pyvista.UnstructuredGrid(topology, cell_types, geom).rotate_z(90).rotate_y(90)
                plotter = pyvista.Plotter()

                plotter.add_mesh(grid, show_edges=True,opacity=0.25)
                # grid.rotate_z(90).rotate_y(90)
                # plotter.add_mesh(grid, show_edges=True,opacity=0.25)
                # have to be careful about how displacement data is populated into grid before or after rotations for visualization
                grid.point_data["u"] = xs.xsdisp.x.array.reshape((geom.shape[0],3))
                # new_grid = grid.transform(transform_matrix)

                warped = grid.warp_by_vector("u", factor=1)
                actor_1 = plotter.add_mesh(warped, show_edges=True)
                plotter.show_axes()
                # if add_nodes==True:
                #     plotter.add_mesh(grid, style='points')
                plotter.view_isometric()
                if not pyvista.OFF_SCREEN:
                    plotter.show()

    def plot_xs_disp_3D(self):
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
        
        grids = []
        grids2 = []
        #get rotation matrices from global frame to reference beam frame
        RbA = self.get_local_basis(self.axial_pos_mesh.geometry.x)
        RTb = self.get_deformed_basis(self.axial_pos_mesh.geometry.x)
        #plot xs meshes:
        for i,xs in enumerate(self.xs_list):
            #compute translation vector (transform centroid offset to relevant coordinates)
            trans_vec = np.array([self.axial_pos_mesh.geometry.x[i]]).T-RbA[i,:,:].T@(np.array([[0,xs.yavg,xs.zavg]]).T)
            
            transform_matrix=np.concatenate((np.concatenate([RbA[i,:,:].T@self.RBG,trans_vec],axis=1),np.array([[0,0,0,1]])))

            #compute translation vector (transform centroid offset to relevant coordinates)
            # print(self.axial_pos_mesh.geometry.x[i])
            global_disp,_ = self.get_global_disp([self.axial_pos_mesh.geometry.x[i]])
            trans_vec2 = np.array([self.axial_pos_mesh.geometry.x[i]]).T-RbA[i,:,:].T@RTb[i,:,:].T@(np.array([[0,xs.yavg,xs.zavg]]).T) + np.array([global_disp]).T
            
            # print("RTb:")
            # print(RTb[i,:,:])
            # print("RbA:")
            # print(RbA[i,:,:])
            transform_matrix2=np.concatenate((np.concatenate([RbA[i,:,:].T@RTb[i,:,:].T@self.RBG,trans_vec2],axis=1),np.array([[0,0,0,1]])))

            tdim = xs.msh.topology.dim
            topology2, cell_types2, geom2 = create_vtk_mesh(xs.msh, tdim)
            grids.append(pyvista.UnstructuredGrid(topology2, cell_types2, geom2))
            grids2.append(pyvista.UnstructuredGrid(topology2, cell_types2, geom2))

            grids[i].transform(transform_matrix)
            grids2[i].transform(transform_matrix2)
            #only need to transform the displacement into the deformed frame
            #RTb[i,:,:]@
            grids[i].point_data["u"] = (RbA[i,:,:].T@self.RBG@xs.xsdisp.x.array.reshape((geom2.shape[0],3)).T).T
            actor2=plotter.add_mesh(grids[i], show_edges=True,opacity=0.25)
            actor4 = plotter.add_mesh(grids2[i], show_edges=True,opacity=0.25)
            # #add mesh for Tangential frame:
            # copied_mesh = actor2.copy(deep=True)
            # copied_mesh.transform(transform_matrix)

            warped = grids[i].warp_by_vector("u", factor=warp_factor)
            actor_3 = plotter.add_mesh(warped, show_edges=True)
            
        plotter.view_yz()
        plotter.show_axes()

        # if not pyvista.OFF_SCREEN:
        plotter.show()
        # else:
        #     pyvista.start_xvfb()
        #     figure = plot.screenshot("beam_mesh.png")

    def recover_stress(self):
        # self.generalized_stresses(self.uh)
        
        #need to interpolate evaluate at axial_pos_mesh nodes, but 
        #   keep stress/reactions defined in the axial mesh or stress recovery will be wildly off.
        points_on_proc,cells=get_pts_and_cells(self.axial_mesh,self.axial_pos_mesh.geometry.x)
        
        S = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=6)
        s = Function(S)
        
        s.interpolate(Expression(self.generalized_stresses(self.uh),S.element.interpolation_points()))
        Sig = s.eval(points_on_proc,cells)

        #SIG are the reaction forces in the xs. These are can then be fed back to the xs to recover the stresses
        # print("reaction forces along beam:")
        # print(Sig)
        for i,xs in enumerate(self.xs_list):
            # print("xs area = " + str(xs.A))
            # print("xs averaged stresses: ")
            # print(Sig[i])

            xs.recover_stress_xs(Sig[i])

            # print('stiffness matrix:')
            # print(xs.K)
            # print('displacements and rotations:')
            # print(self.uh.sub(0).x.array)

#TODO: need to add joints to allow for the assembly of models with connections (e.g. branching, loops, frames, etc)
#TODO:
# ExampleBeam.get_max_stress()
# ExampleBeam.plot_stress_over_xs()