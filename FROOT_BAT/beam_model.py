'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.
f dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from dolfinx.fem import Function
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross)
from FROOT_BAT.elements import *

class LinearTimoshenko(object):
    
    '''
    Timoshenko shear deformable beam formulation
    
    Inputs:

    domain: 1D analysis mesh
    beam_props: 2-tensor (6x6) function defining beam properties along span
    
    OUTPUT:
        Residual: assembled weak form
    '''

    def __init__(self,domain,beam_props):
        #import domain, function, and beam properties
        self.domain = domain
        self.beam_element = BeamElementRefined(domain)
        self.eleDOFs = 6 #link to fxn space
        self.beam_props = beam_props
        #TODO: incorporate beam fxn space
        [self.S, 
         self.ES, self.GS1, self.GS2,
         self.GJ, self.EI1, self.EI2] = self.beam_props
        
        self.w = TestFunction(self.beam_element.W)
        self.dw = TrialFunction(self.beam_element.W)
        (self.u_, self.theta_) = split(self.w)
        (self.du_, self.dtheta) = split(self.dw)
    
        self.t = self.tangent(domain)

        self.compute_local_axes()

    def getResidual():

        
    def elasticEnergy(self):
        self.Sig = self.generalized_stresses(self.dw)
        self.Eps = self.generalized_strains(self.w)

        return sum([self.Sig[i]*self.Eps[i]*self.beam_element.dx_beam[i] for i in range(self.eleDOFs)])

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
        #TODO: reformulate to a "stiffness matrix"
        return dot(diag(as_vector(self.beam_props[1:])), self.generalized_strains(w))

    #constructing RHS:
    def addBodyForce(self,f,frame='ref',ax=0):
        if frame == 'ref':
            vec_dir = self.u_[ax]
        return -f*self.S*vec_dir*dx
