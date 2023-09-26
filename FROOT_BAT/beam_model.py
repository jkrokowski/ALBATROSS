'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.

Dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from dolfinx.fem import Function,Constant
from dolfinx.fem.petsc import LinearProblem
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross,dx)
from FROOT_BAT.elements import *
from petsc4py.PETSc import ScalarType

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
        
        self.w = TestFunction(self.beam_element.W)
        self.dw = TrialFunction(self.beam_element.W)
        (self.u_, self.theta_) = split(self.w)
        (self.du_, self.dtheta) = split(self.dw)

        self.a_form = None
        self.L_form = None
    
        self.t = self.tangent(domain)

        self.compute_local_axes()
       
    def elasticEnergy(self):
        self.Sig = self.generalized_stresses(self.dw)
        self.Eps = self.generalized_strains(self.w)

        self.a_form = sum([self.Sig[i]*self.Eps[i]*self.beam_element.dx_beam[i] for i in range(self.eleDOFs)])

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
        if self.L_form ==None:
            self.L_form = -dot(f,self.u_)*dx
        else:
            self.L_form += -dot(f,self.u_)*dx

    def solve(self):
        self.uh = Function(self.beam_element.W)
        if self.L_form == None:
            f = Constant(self.domain,ScalarType((0,0,0)))
            self.L_form = -dot(f,self.u_)*dx
        self.problem = LinearProblem(self.a_form, self.L_form, u=self.uh, bcs=[bcs])
        self.uh = self.problem.solve()

        
    

