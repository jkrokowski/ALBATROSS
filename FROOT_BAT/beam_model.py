'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.

Dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from dolfinx.fem import Function,Constant, locate_dofs_geometrical,dirichletbc
from dolfinx.fem.petsc import LinearProblem
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross,dx,Measure)
from FROOT_BAT.elements import *
from petsc4py.PETSc import ScalarType
import numpy as np

class LinearTimoshenko(object):
    
    '''
    Timoshenko shear deformable beam formulation
    
    Inputs:

    domain: 1D analysis mesh
    beam_props: 2-tensor (6x6) function defining beam properties along span
    
    OUTPUT:
        Residual: assembled weak form
    '''

    def __init__(self,domain,xcinfo):
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

        self.compute_local_axes()
       
    def elasticEnergy(self):
        self.Sig = self.generalized_stresses(self.dw)
        self.Eps = self.generalized_strains(self.w)

        self.a_form = (inner(self.Sig,self.Eps))*self.dx

    def tangent(self,domain):
        t = Jacobian(domain)
        return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))     

    def compute_local_axes(self):
        #compute section local axes
        self.ez = as_vector([0, 0, 1])
        self.a1 = cross(self.t, self.ez)
        self.a1 /= sqrt(dot(self.a1, self.a1))
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
        return dot(self.xcinfo, self.generalized_strains(w))

    #constructing RHS:
    def addBodyForce(self,f):
        '''
        f = tuple for (x,y,z) components of body force
        '''
        f_vec = Constant(self.domain,ScalarType(f))
        if self.L_form ==None:
            self.L_form = -dot(f_vec,self.u_)*self.dx
        else:
            self.L_form += -dot(f_vec,self.u_)*self.dx

    def solve(self):
        self.uh = Function(self.beam_element.W)
        if self.L_form == None:
            f = Constant(self.domain,ScalarType((0,0,0)))
            self.L_form = -dot(f,self.u_)*self.dx
        
        self.problem = LinearProblem(self.a_form, self.L_form, u=self.uh, bcs=self.bcs)
        self.uh = self.problem.solve()

    def addClampedPoint(self,pt):
        '''
        pt = x,y,z location of clamped point
        '''
        #marker fxn
        def clamped_point(x):
            x_check = np.isclose(x[0],pt[0])
            y_check = np.isclose(x[1],pt[1])
            z_check = np.isclose(x[2],pt[2])
            return np.logical_and.reduce([x_check,y_check,z_check])
        
        #displacement DOFs fixed
        W0, disp_dofs = self.beam_element.W.sub(0).collapse()
        udispbc = Function(W0)
        with udispbc.vector.localForm() as uloc:
            uloc.set(0.)
        clamped_disp_dofs = locate_dofs_geometrical((self.beam_element.W.sub(0),W0),clamped_point)
        disp_bc = dirichletbc(udispbc,clamped_disp_dofs)
        self.bcs.append(disp_bc)

        #rotation DOFs fixed
        W1, rot_dofs = self.beam_element.W.sub(1).collapse()
        urotbc = Function(W1)
        with urotbc.vector.localForm() as uloc:
            uloc.set(0.)
        clamped_rot_dofs = locate_dofs_geometrical((self.beam_element.W.sub(1),W1),clamped_point)
        rot_bc = dirichletbc(urotbc,clamped_rot_dofs)
        self.bcs.append(rot_bc)






