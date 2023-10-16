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
                inner,dot,grad,split,cross,Measure)
from FROOT_BAT.elements import *
from petsc4py.PETSc import ScalarType
import numpy as np

from petsc4py import PETSc

class Axial:
    
    '''
    Timoshenko shear deformable beam formulation
    
    Inputs:

    domain: 1D analysis mesh
    beam_props: 2-tensor (6x6) function defining beam properties along span
    '''

    def __init__(self,domain,xcinfo,orientation):
        #import domain, function, and beam properties
        self.domain = domain
        self.beam_element = BeamElementRefined(domain)
        self.eleDOFs = 6
        self.xcinfo = xcinfo

        self.dx = Measure('dx',self.domain)
        
        self.w = TestFunction(self.beam_element.W)
        self.dw = TrialFunction(self.beam_element.W)
        (self.u_, self.theta_) = split(self.w)
        (self.du_, self.dtheta) = split(self.dw)

        self.a_form = None
        self.L_form = None
        self.bcs = []
    
        self.t = self.tangent(domain)
        
        self.a1 = orientation

        self.compute_local_axes()
       
    def elastic_energy(self):
        self.Sig = self.generalized_stresses(self.dw)
        self.Eps = self.generalized_strains(self.w)

        self.a_form = (inner(self.Sig,self.Eps))*self.dx

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
        # Q = diag(as_vector([1,2,3,4,5,6]))
        # return dot(Q, self.generalized_strains(w))
        return dot(self.xcinfo, self.generalized_strains(w))

    #constructing RHS:
    def add_body_force(self,f):
        '''
        f = tuple for (x,y,z) components of body force
        '''
        print("Adding body force....")
        f_vec = Constant(self.domain,ScalarType(f))
        if self.L_form ==None:
            self.L_form = dot(f_vec,self.u_)*self.dx
        else:
            self.L_form += dot(f_vec,self.u_)*self.dx

    def solve(self):
        print("Solving for beam axis displacement...")
        self.uh = Function(self.beam_element.W)
        if self.L_form == None:
            f = Constant(self.domain,ScalarType((0,0,0)))
            self.L_form = -dot(f,self.u_)*self.dx
        
        self.problem = LinearProblem(self.a_form, self.L_form, u=self.uh, bcs=self.bcs)
        self.uh = self.problem.solve()
    def add_clamped_point(self,pt):
        '''
        pt = x,y,z location of clamped point
        '''
        print("Adding clamped point...")
        #marker fxn
        def clamped_point(x):
            x_check = np.isclose(x[0],pt[0])
            y_check = np.isclose(x[1],pt[1])
            z_check = np.isclose(x[2],pt[2])
            return np.logical_and.reduce([x_check,y_check,z_check])
        #function for bc application
        ubc = Function(self.beam_element.W)
        with ubc.vector.localForm() as uloc:
            uloc.set(0.)
        #find displacement DOFs
        W0, disp_dofs = self.beam_element.W.sub(0).collapse()
        clamped_disp_dofs,_ = locate_dofs_geometrical((self.beam_element.W.sub(0),W0),clamped_point)

        #find rotation DOFs
        W1, rot_dofs = self.beam_element.W.sub(1).collapse()
        clamped_rot_dofs,_ = locate_dofs_geometrical((self.beam_element.W.sub(1),W1),clamped_point)
        
        clamped_dofs= np.concatenate([clamped_disp_dofs,clamped_rot_dofs])
        clamped_bc = dirichletbc(ubc,clamped_dofs)
        self.bcs.append(clamped_bc)

        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        ubc.vector.destroy()    #need to add to prevent PETSc memory leak 

    def add_clamped_point_topo(self,dof):
        ubc = Function(self.beam_element.W)
        with ubc.vector.localForm() as uloc:
            uloc.set(0.)
        locate_BC = locate_dofs_topological(self.beam_element.W,0,dof)
        print(locate_BC)
        self.bcs.append(dirichletbc(ubc,locate_BC))
        ubc.vector.destroy()

    def get_global_disp(self,point):
        '''
        returns the displacement and rotation at a specific 
        point on the beam axis with respect to the global coordinate system

        ARGS:
            point = tuple of (x,y,z) locations to return displacements and rotations
        RETURNS:
            ndarray of 3x2 of [disp,rotations], where disp and rot are both shape 3x1 
        '''
        from dolfinx import geometry as df_geom
        bb_tree = df_geom.BoundingBoxTree(self.domain,self.domain.topology.dim)
        points = np.array([point]).T
        
        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = df_geom.compute_collisions(bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = df_geom.compute_colliding_cells(self.domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc,dtype=np.float64)
        disp = self.uh.sub(0).eval(points_on_proc,cells)
        rot = self.uh.sub(1).eval(points_on_proc,cells)

        return np.array([disp,rot])
    
    def get_local_disp(self,point,return_rotation_mat = False):
        '''
        returns the displacement and rotation at a specific 
        point on the beam axis with respect to the axial direction and xc principle axes

        ARGS:
            point = tuple of (x,y,z) locations to return displacements and rotations
        RETURNS:
            ndarray of 3x2 of [disp,rotations], where disp and rot are both shape 3x1 
        '''
        from dolfinx import geometry as df_geom
        bb_tree = df_geom.BoundingBoxTree(self.domain,self.domain.topology.dim)
        points = np.array([point]).T

        cells = []
        points_on_proc = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = df_geom.compute_collisions(bb_tree, points.T)
        # Choose one of the cells that contains the point
        colliding_cells = df_geom.compute_colliding_cells(self.domain, cell_candidates, points.T)
        for i, point in enumerate(points.T):
            if len(colliding_cells.links(i))>0:
                points_on_proc.append(point)
                cells.append(colliding_cells.links(i)[0])

        points_on_proc = np.array(points_on_proc,dtype=np.float64)
        disp = self.uh.sub(0).eval(points_on_proc,cells)
        rot = self.uh.sub(1).eval(points_on_proc,cells)
        
        #interpolate ufl expressions for tangent basis vectors and xc basis
        # vectors into appropriate fxn space, then eval to get the local 
        # orientation w.r.t a global ref frame
        T = VectorFunctionSpace(self.domain,('CG',1),dim=3)

        t = Function(T)
        t.interpolate(Expression(self.t,T.element.interpolation_points()))
        tangent = t.eval(points_on_proc,cells)
        # a1 = Function(T)
        # a1.interpolate(Expression(self.a1,T.element.interpolation_points()))
        # y = a1.eval(points_on_proc,cells)
        y = self.a1.eval(points_on_proc,cells)
        a2 = Function(T)
        a2.interpolate(Expression(self.a2,T.element.interpolation_points()))
        z = a2.eval(points_on_proc,cells)
        print("local xc bases:")
        print(tangent)
        print(y)
        print(z)
        # T2 =TensorFunctionSpace(self.domain,('CG',1),shape=(3,3))
        # grad_uh_interp = Function(T2)
        # grad_uh = grad(self.uh.sub(0))
        # grad_uh_0 = grad(self.uh.sub(0)[0])
        # grad_uh_0_interp= Function(T)
        # grad_uh_0_interp.interpolate(Expression(grad_uh_0,T.element.interpolation_points()))
        # grad_uh_interp.interpolate(Expression(grad_uh,T2.element.interpolation_points()))
        
        # grad_uh_pt = grad_uh_interp.eval(points_on_proc,cells).reshape(3,3)
        R = np.array([tangent,y,z])
        
        disp = R @disp
        rot = R@rot
        # if return_rotation_mat == True:
        #     return [np.array([disp,rot]),R,rot_angle]
        # else:
        #     return np.array([disp,rot])
        return [np.array([disp,rot]),R]
    
    def get_3D_disp(self):
        '''
        returns a fxn defined over a 3D mesh generated from the 
        2D xc's and the 1D analysis mesh
        '''
        return

    # def solve2(self):
    #     self.A_mat = assemble_matrix(form(self.a_form),bcs=self.bcs)
    #     self.A_mat.assemble()

    #     if self.L_form == None:
    #         f = Constant(self.domain,ScalarType((0,0,0)))
    #         # self.L_form = -dot(f,self.u_)*self.dx
    #         self.L_form = Constant(self.domain,ScalarType(10.))*self.u_[2]*self.dx
        
    #     form(self.L_form)
    #     self.b=create_vector(form(self.L_form))
    #     with self.b.localForm() as b_loc:
    #                 b_loc.set(0)
    #     assemble_vector(self.b,form(self.L_form))

    #     # APPLY dirchelet bc: these steps are directly pulled from the 
    #     # petsc.py LinearProblem().solve() method
    #     self.a_form = form(self.a_form)
    #     apply_lifting(self.b,[self.a_form],bcs=self.bcs)
    #     self.b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    #     set_bc(self.b,self.bcs)

    #     uh_ptld = Function(self.beam_element.W)
    #     uvec = uh_ptld.vector
    #     uvec.setUp()
    #     ksp = PETSc.KSP().create()
    #     ksp.setType(PETSc.KSP.Type.CG)
    #     ksp.setTolerances(rtol=1e-15)
    #     ksp.setOperators(self.A_mat)
    #     ksp.setFromOptions()
    #     ksp.solve(self.b,uvec)





