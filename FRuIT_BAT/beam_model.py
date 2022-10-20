'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.
f dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from ufl import (Jacobian, as_vector, sqrt, inner)

class BeamModelRefined(object):
    
    '''
    Timoshenko shear deformable beam formulation
    '''

    def __init(self,domain,w):
        self.domain = domain
        self.w = w

    def tangent(domain):
        t = Jacobian(domain)
        return as_vector([t[0,0], t[1, 0], t[2, 0]])/sqrt(inner(t,t))     
        


