'''
The axial module for executing a 1D analysis
---------------------
Contains the most important classes for beam formulations 
using shear-deformable Timoshenko Beam Theory

'''
#TODO: implement a pure displacement based Euler-Bernoulli beam based on examples
#TODO: upgrade to geometrically nonlinear capable models
#TODO: include dynamics by adding mass properties

# from dolfinx.fem import (functionspace,Expression,Function,Constant, 
#                         locate_dofs_geometrical,locate_dofs_topological,
#                         dirichletbc,form)
from dolfinx import fem,plot,default_scalar_type
from dolfinx.fem.petsc import (LinearProblem,assemble_matrix,assemble_vector, 
                                apply_lifting,set_bc,create_vector)
import ufl
from ufl import (Jacobian, TestFunction,TrialFunction,as_vector, sqrt, 
                inner,dot,grad,split,cross,Measure,derivative)
from ALBATROSS.elements import LinearTimoshenkoElement
from petsc4py.PETSc import ScalarType
import numpy as np
import pyvista
from ALBATROSS.petsc_utils import convert_petsc_to_numpy
from ALBATROSS.utils import get_pts_and_cells
from ALBATROSS.mesh import beam_interval_mesh_3D

from petsc4py import PETSc

class BeamAxis:
    def __init__(self,points,ele,name):
        '''
        points: list of points defining the start and stop points of each unique beam segment
        ele: number of element for each beam segment (array of size (len(points)-1,))
        name: name of the beam (string)
        '''
        print("Constructing beam axis...")
        axial_pos_meshname = name+'_axial_pos_mesh'
        self.axial_pos_mesh = beam_interval_mesh_3D(points,np.ones((len(points)-1)),axial_pos_meshname)
        axial_meshname = name+'_axial_mesh'
        self.axial_mesh = beam_interval_mesh_3D(points,ele,axial_meshname)
        print("DONE constructing beam axis")

class Axial:
    
    '''
    Timoshenko shear deformable beam formulation
    
    Inputs:

    domain: 1D analysis mesh
    beam_props: 2-tensor (6x6) function defining beam properties along span
    '''

    def __init__(self,domain,k,orientation):
        #import domain, function, and beam properties
        self.domain = domain
        self.beam_element = LinearTimoshenkoElement(domain)
        self.eleDOFs = 6
        self.k = k

        self.dx = Measure('dx',self.domain)
        self.dx_shear = Measure('dx',self.domain,metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
        
        self.w = fem.Function(self.beam_element.W)
        self.v = TestFunction(self.beam_element.W)
        self.dw = TrialFunction(self.beam_element.W)
        (self.u, self.theta) = split(self.w)
        (self.v_u, self.v_theta) = split(self.v)

        self.a_form = None
        self.L_form = None
        self.bcs = []

        self.f_pt = []
        self.m_pt = []
    
        self.t = self.tangent(domain)
        
        self.a1 = orientation
        self.a1 /= sqrt(dot(self.a1, self.a1))

        self.compute_local_axes()
    
    def elastic_energy(self):
        self.Sig = self.generalized_stresses(self.v)
        self.Eps = self.generalized_strains(self.w)

        #assemble variational form separately for shear terms (using reduced integration)
        self.F_form = (sum([self.Sig[i]*self.Eps[i]*self.dx for i in [0, 3, 4, 5]]) 
                        + sum([self.Sig[i]*self.Eps[i]*self.dx_shear for i in [1,2]])) 
        # self.a_form = (inner(self.Sig,self.Eps))

        self.a_form = derivative(self.F_form,self.w,self.dw)

    def tangent(self,domain):
        t = Jacobian(domain)
        return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))     

    def compute_local_axes(self):
        #compute section local axes
        # self.ez = as_vector([0, 0, 1])
        # self.a1 = cross(self.t, self.ez)
        # self.a1 /= sqrt(dot(self.a1, self.a1))

        #Can add a TRY/EXCEPT and then warn the user if they haven't defined proper orientations
        self.a2 = cross(self.t, self.a1)
        self.a2 /= sqrt(dot(self.a2, self.a2))
        
    def tgrad(self,w):
        return dot(grad(w), self.t)

    def generalized_strains(self,w):
        (u, theta) = split(w)
        return as_vector([dot(self.tgrad(u), self.t),
                        dot(self.tgrad(u), self.a1)-dot(theta, self.a2),
                        dot(self.tgrad(u), self.a2)+dot(theta, self.a1),
                        dot(self.tgrad(theta), self.t),
                        dot(self.tgrad(theta), self.a1),
                        dot(self.tgrad(theta), self.a2)])

    def generalized_stresses(self,w):
        # return dot(self.k, self.generalized_strains(w))
        return dot(self.generalized_strains(w),self.k)

    #constructing RHS:
    def add_dist_load(self,f):
        '''K
        f = tuple for (x,y,z) components of distributed load
        '''
        
        print("Adding distributed load....")
        # f_vec = as_vector([self.linear_density[i]*Constant(self.domain,default_scalar_type(f[i])) for i in range(3)])
        f_vec = self.linear_density*fem.Constant(self.domain,default_scalar_type(f))
        # print(f_vec.ufl_shape)
        # print(self.v_.ufl_shape)
        print(dot(f_vec,self.v_u).ufl_shape)
        if self.L_form is None:
            self.L_form = dot(f_vec,self.v_u)*self.dx
        else:
            self.L_form += dot(f_vec,self.v_u)*self.dx

    def add_point_load(self,f_list,pts):
        '''
        f = list of tuples for (x,y,z) components of point force force
        pts : list of (x,y,z) location along beam axis to apply point force
        '''
        
        print("Adding point loads....")
        #TODO: add data format checks here
        for f,pt in zip(f_list,pts):
            self.f_pt.append((f,pt))

    def add_point_moment(self,m_list,pts):
        '''
        f = list of tuples for (x,y,z) components of point force force
        pts : list of (x,y,z) location along beam axis to apply point force
        '''
        
        print("Adding point loads....")
        #TODO: add data format checks here
        for m,pt in zip(m_list,pts):
            self.m_pt.append((m,pt))
        
    def xyz_to_span(self,pt):
        #TODO: method to convert an xyz coordinate to the length along the beam
        return
    
    def span_to_xyz(self,l):
        #TODO: method to convert spanwise coordinate l to an xyz location
        return
        
    def solve(self):
        print("Solving for beam axis displacement...")

        #assemble the system from 
        self._construct_system()

        #initialize function to store solution of assembled system:
        uvec = self.w.x.petsc_vec#petsc vector
        uvec.setUp()
        self.solver = PETSc.KSP().create()
        self.solver.setType(PETSc.KSP.Type.CG)
        self.solver.setTolerances(rtol=1e-15)
        self.solver.setOperators(self.A_mat)
        # ksp.setFromOptions()
        self.solver.solve(self.b,uvec)

    def _solve_simple(self):    
        # --------
        # between these lines is the vastly simplified solution in the case of:
        #   -no point loads
        #   -no point moments
        # initialize function for displacement and rotation solution
        # self.w = fem.Function(self.beam_element.W)
        if self.L_form is None:
            f = fem.Constant(self.domain,ScalarType((0,0,0)))
            self.L_form = -dot(f,self.v_u)*self.dx
        
        self.problem = LinearProblem(self.a_form, self.L_form, u=self.w, bcs=self.bcs)
        self.w = self.problem.solve()
        # --------

    def _construct_system(self):
        self.A_mat = assemble_matrix(fem.form(self.a_form),bcs=self.bcs)
        self.A_mat.assemble()

        if self.L_form is None:
            f0 = fem.Constant(self.domain,ScalarType((0,0,0)))
            self.L_form = -dot(f0,self.v_u)*self.dx
        
        self.b=create_vector(fem.form(self.L_form))
        with self.b.localForm() as b_loc:
                    b_loc.set(0)
        assemble_vector(self.b,fem.form(self.L_form))

        # APPLY dirichlet bc: these steps are directly pulled from the 
        # petsc.py LinearProblem().solve() method
        # self.a_form = form(self.a_form)
        apply_lifting(self.b,[fem.form(self.a_form)],bcs=[self.bcs])
        self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b,self.bcs)
        
        #if there are any point forces, apply them to the assembled rhs vector
        if bool(self.f_pt):
            W0, disp_dofs = self.beam_element.W.sub(0).collapse()
            for f_pt in self.f_pt:
                f = f_pt[0]
                pt = f_pt[1]
                def locate_dofs(x):
                    return np.logical_and.reduce((np.isclose(x[0],pt[0]),
                                                  np.isclose(x[1],pt[1]),
                                                  np.isclose(x[2],pt[2])))
                f_dofs = fem.locate_dofs_geometrical((self.beam_element.W.sub(0),W0),locate_dofs)
                self.b.array[f_dofs[0]] = f
        if bool(self.m_pt):
            W1, rot_dofs = self.beam_element.W.sub(1).collapse()
            for m_pt in self.m_pt:
                m = m_pt[0]
                pt = m_pt[1]
                def locate_dofs(x):
                    return np.logical_and.reduce((np.isclose(x[0],pt[0]),
                                                  np.isclose(x[1],pt[1]),
                                                  np.isclose(x[2],pt[2])))
                m_dofs = fem.locate_dofs_geometrical((self.beam_element.W.sub(1),W1),locate_dofs)
                self.b.array[m_dofs[0]] = m

    def add_clamped_point(self,pt):
        '''
        pt = x,y,z location of clamped point
        '''
        print("Adding clamped point...")
        #function for bc application
        ubc = fem.Function(self.beam_element.W)
        # with ubc.vector.localForm() as uloc:
        #     uloc.set(0.)
        
        clamped_disp_dofs = self._get_dofs(pt,'disp')
        clamped_rot_dofs = self._get_dofs(pt,'rot')

        clamped_dofs= np.concatenate([clamped_disp_dofs,clamped_rot_dofs])
        clamped_bc = fem.dirichletbc(ubc,clamped_dofs)
        self.bcs.append(clamped_bc)

        # # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        # ubc.vector.destroy()    #need to add to prevent PETSc memory leak 
    
    def _get_dofs(self,pt,dof_type="disp"):
            def locate_pt(x):
                x_check = np.isclose(x[0],pt[0])
                y_check = np.isclose(x[1],pt[1])
                z_check = np.isclose(x[2],pt[2])
                return np.logical_and.reduce([x_check,y_check,z_check])
            if dof_type=='disp':
                #find displacement DOFs
                W0, disp_map = self.beam_element.W.sub(0).collapse()
                dofs,_ = fem.locate_dofs_geometrical((self.beam_element.W.sub(0),W0),locate_pt)
            if dof_type=='rot':
                W1, rot_map = self.beam_element.W.sub(1).collapse()
                dofs,_ = fem.locate_dofs_geometrical((self.beam_element.W.sub(1),W1),locate_pt)
    
            return dofs
    def add_clamped_point_topo(self,dof):
        ubc = fem.Function(self.beam_element.W)
        with ubc.vector.localForm() as uloc:
            uloc.set(0.)
        locate_BC = fem.locate_dofs_topological(self.beam_element.W,0,dof)
        self.bcs.append(fem.dirichletbc(ubc,locate_BC))
        ubc.vector.destroy()

    def get_global_disp(self,points):
        '''
        returns the displacement and rotation at a specific 
        point on the beam axis with respect to the global coordinate system

        ARGS:
            points = list of (x,y,z) locations to return displacements and rotations
        RETURNS:
            list of [disp,rotations], where disp and rot are both shape (numpts,3) 
        '''
        points_on_proc,cells=get_pts_and_cells(self.domain,points)

        disp = self.w.sub(0).eval(points_on_proc,cells)
        rot = self.w.sub(1).eval(points_on_proc,cells)
        
        return disp,rot
    
    def get_local_basis(self,points):
        '''
        returns the basis vectors for a set of points given as a (numpts,3,3) ndarray
        '''
        points_on_proc,cells=get_pts_and_cells(self.domain,points)
        #get coordinate system at each mesh node
        T = fem.functionspace(self.domain,('CG',1,(self.domain.geometry.dim,)))

        t = fem.Function(T)
        t.interpolate(fem.Expression(self.t,T.element.interpolation_points()))
        tangent = t.eval(points_on_proc,cells)
        a1 = fem.Function(T)
        a1.interpolate(fem.Expression(self.a1,T.element.interpolation_points()))
        y = a1.eval(points_on_proc,cells)
        # y = self.a1.eval(points_on_proc,cells)
        a2 = fem.Function(T)
        a2.interpolate(fem.Expression(self.a2,T.element.interpolation_points()))
        z = a2.eval(points_on_proc,cells)

        return np.moveaxis(np.array([tangent,y,z]),0,1)
    
    def get_local_disp(self,points):
        '''
        returns the displacement and rotation at a specific 
        point on the beam axis with respect to the axial direction and xs principle axes

        ARGS:
            point = tuple of (x,y,z) locations to return displacements and rotations
        RETURNS:
            ndarray of 3x2 of [disp,rotations], where disp and rot are both shape 3x1 
        '''
        [disp,rot] = self.get_global_disp(points)
        
        # get local basis (rotation matrix)
        RbA = self.get_local_basis(points)

        #TODO: vectorize this for loop, is tensordot the right approach?

        if len(RbA.shape) == 2:
            disp = RbA@disp
            rot = RbA@rot
        else:
            for i in range(len(disp)):
                disp[i,:] = RbA[i,:,:]@disp[i,:]
                rot[i,:] = RbA[i,:,:]@rot[i,:]
        
        return [disp,rot]
    
    def get_deformed_basis(self,points):
        '''
        get the transformation matrix from the reference beam frame (b) to the
          deformed beam frame (B)

        This only works under the assumption of small displacments (e.g. linear beam theory)
        '''
        # self.RTb = 
        T = fem.functionspace(self.domain,('CG',1,(self.domain.geometry.dim,)))
        T2 =fem.functionspace(self.domain,('CG',1,(3,3)))
        grad_w_interp = fem.Function(T2)
        grad_w = grad(self.w.sub(0))
        grad_w_0 = grad(self.w.sub(0)[0])
        grad_w_0_interp= fem.Function(T)
        grad_w_0_interp.interpolate(fem.Expression(grad_w_0,T.element.interpolation_points()))
        grad_w_interp.interpolate(fem.Expression(grad_w,T2.element.interpolation_points()))
        
        points_on_proc,cells=get_pts_and_cells(self.domain,points)
        # print("strains:")
        # print(grad_w_interp.eval(points_on_proc,cells))
        strains = grad_w_interp.eval(points_on_proc,cells).reshape((len(points),3,3))
        first_comp = grad_w_0_interp.eval(points_on_proc,cells)
        #TODO: ensure this works for multiple points at once
        RbA = self.get_local_basis(points)

        local_strains= np.zeros_like(RbA)
        RTb = np.zeros_like(RbA)

        #rotation angles
        alpha = 0
        beta = local_strains[2,0]
        gamma = local_strains[1,0]
        Rx = np.array([[1,         0,         0],
                        [0,np.cos(alpha),-np.sin(alpha)],
                        [0,np.sin(alpha),np.cos(alpha)]])
        # rotation about Y-axis
        Ry = np.array([[np.cos(beta), 0,np.sin(beta)],
                        [0,         1,        0],
                        [-np.sin(beta),0,np.cos(beta)]])
        #rotation about Z-axis
        Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
                        [np.sin(gamma),np.cos(gamma), 0],
                        [0,         0,          1]])

        RTb[:,:] = Rz@Ry@Rx
        # for i in range(RbA.shape[0]):
        #     local_strains[i,:,:] = RbA[i,:,:]@strains[i,:,:]@RbA[i,:,:].T
        #     # print("local strain:")
        #     # print(local_strains[i,:,:])
        #     # print(strains)
        #     # print(local_strains)
        #     alpha = 0
        #     beta = local_strains[i,2,0]
        #     gamma = local_strains[i,1,0]
        #     Rx = np.array([[1,         0,         0],
        #                     [0,np.cos(alpha),-np.sin(alpha)],
        #                     [0,np.sin(alpha),np.cos(alpha)]])
        #     # rotation about Y-axis
        #     Ry = np.array([[np.cos(beta), 0,np.sin(beta)],
        #                     [0,         1,        0],
        #                     [-np.sin(beta),0,np.cos(beta)]])
        #     #rotation about Z-axis
        #     Rz = np.array([[np.cos(gamma),-np.sin(gamma),0],
        #                     [np.sin(gamma),np.cos(gamma), 0],
        #                     [0,         0,          1]])

        #     RTb[i,:,:] = Rz@Ry@Rx

        # print("Deformed basis:")
        # print(RTb)

        return RTb

    def plot_axial_displacement(self,warp_factor=1):
        '''
        returns a fxn defined over a 3D mesh generated from the 
        2D xs's and the 1D analysis mesh
        '''
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'

        tdim = self.domain.topology.dim
        topology, cell_types, geom = plot.vtk_mesh(self.domain,tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
        plotter = pyvista.Plotter()
        sargs = dict(
            title_font_size=20,
            label_font_size=16,
            shadow=True,
            n_labels=3,
            italic=True,
            fmt="%.3f",
            font_family="arial",
        )
        grid.point_data["Beam axis displacement"] = self.w.sub(0).collapse().x.array.reshape((geom.shape[0],3))
        actor_0 = plotter.add_mesh(grid, style="wireframe", line_width=5,color="k",scalar_bar_args=sargs)
        warped = grid.warp_by_vector("Beam axis displacement", factor=warp_factor)
        actor_1 = plotter.add_mesh(warped, line_width=5,show_edges=True)
        plotter.view_isometric()
        plotter.show_axes()
        plotter.show_bounds()
        # plotter.show_grid()
        if not pyvista.OFF_SCREEN:
            plotter.show()
        else:
            figure = plot.screenshot("beam_mesh.png")

    def get_reactions(self,points):
        '''
        returns a list of forces and moments at the specified 
            beam axis location
        '''
        
        #Construct expression to evalute
        R = fem.functionspace(self.axial_mesh,('DG',0,(6,)))
        r = fem.Function(R)
        r.interpolate(fem.Expression(
                        self.generalized_stresses(self.w),
                        R.element.interpolation_points()
                        ) )
        
        points_on_proc,cells=get_pts_and_cells(self.domain,points)
        Reactions = r.eval(points_on_proc,cells)

        return Reactions
    


    #========== derivative computations ============#
    def compute_vjp(self,d_residuals):
        #set up input vector sizes
        d_residuals_vec_size = self.b.getSize()
        d_residuals_vec = PETSc.Vec().createSeq(d_residuals_vec_size, comm=PETSc.COMM_SELF)
        
        #set up output vector sizes
        # d_inputs_vec_size = self.T_66.dofmap.index_map_bs*self.T_66.dofmap.index_map.size_global
        # d_inputs_vec = PETSc.Vec().createSeq(d_inputs_vec_size, comm=PETSc.COMM_SELF)
        # dRdk_vec = PETSc.Vec().createSeq(d_residuals_vec_size, comm=PETSc.COMM_SELF)

        # d_inputs_vec =  PETSc.Vec().createSeq(36, comm=PETSc.COMM_SELF)
        
        d_residuals_vec.array[:] = d_residuals

        d_inputs = np.zeros((6,6))
        for i in range(6):
            for j in range(6):
                #compute the derivative of the beam linear functional w.r.t. the Kij entry
                dRdk_vec = fem.petsc.assemble_vector(fem.form(ufl.diff(self.F_form,self.k[i,j])))
                # populate value of d_inputs
                d_inputs[i,j] = d_residuals_vec.dot(dRdk_vec)
                
                
                
                # dFdKij=fem.petsc.assemble_matrix(fem.form(ufl.diff(self.a_form,self.k[i,j])))        
                # dFdKij.assemble()

                # dFdKij.mult(self.w.x.petsc_vec,dRdk_vec)
                #numpy version for debugging:
                # dFdKijnp = convert_petsc_to_numpy(dFdKij)
                # d_inputs[i,j] = d_residuals_vec.array.dot(dFdKijnp@self.w.x.array )


        # dRdK = fem.petsc.assemble_matrixf(fem.form(derivative(self.F_form,self.k,ufl.TestFunction(self.T_66))))
        # dRdK.assemble()
        
        # d_residuals_vec.array[dof] = d_residual
        # dRdK.multTranspose(d_residuals_vec,d_inputs_vec) #perform vec-mat product
        
        return d_inputs   

    def apply_inverse_jacobian(self,d_outputs):
        d_residuals = self.A_mat.createVecLeft()
        d_residuals.setUp()
        d_outputs_vec = self.A_mat.createVecRight()
        d_outputs_vec.setUp()

        d_outputs_vec.array[:] = d_outputs

        self.solver.solveTranspose(d_outputs_vec,d_residuals)
    
        return d_residuals.array