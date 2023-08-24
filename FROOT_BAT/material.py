from dolfinx.fem import Constant
from ufl import Identity,as_tensor,indices,diag,as_vector,as_matrix

def getMatConstitutiveIsotropic(E,nu):
    #lame parameters from Young's Modulus and Poisson Ratio
    _lam = (E*nu)/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))

    #elasticity tensor construction
    delta = Identity(3)
    i,j,k,l=indices(4)
    D = as_tensor(_lam*(delta[i,j]*delta[k,l]) \
                    + mu*(delta[i,k]*delta[j,l]+ delta[i,l]*delta[j,k])  ,(i,j,k,l))

    return D

def getMatConstitutiveOrthotropic(E11,E22,E33,G12,G13,G23,nu12,nu13,nu23):
    
    nu21 = (E22/E11)*nu12
    nu31 = (E33/E11)*nu13
    nu32 = (E33/E22)*nu23
    
    gamma = 1 / (1-nu12*nu21-nu23*nu23-nu13*nu31-2*nu21*nu32*nu13)

    D1111 = E11*(1-nu23*nu32)*gamma
    D2222 = E22*(1-nu13*nu31)*gamma
    D3333 = E33*(1-nu12*nu21)*gamma
    D1122 = E11*(nu21+nu31*nu23)*gamma
    D1133 = E11*(nu31+nu21*nu32)*gamma
    D2233 = E22*(nu32+nu12*nu31)*gamma
    D1212 = G12
    D1313 = G13
    D2323 = G23

    D_voigt = as_matrix([[D1111,D1122,D1133,0,    0,    0    ],
                         [D1122,D2222,D2233,0,    0,    0    ],
                         [D1133,D2233,D3333,0,    0,    0    ],
                         [0,    0,    0,    D1212,0,    0    ],
                         [0,    0,    0,    0,    D1313,0    ],
                         [0,    0,    0,    0,    0,    D2323]])
    
    D = voigt2tensor(D_voigt)

    return D

def voigt2tensor(D_voigt):
    '''converts a 6x6 matrix into the appropriately constructed
      3 dimensional, Fourth order tensor
      '''
    
    D = as_tensor([[as_matrix([[D_voigt[0,0],D_voigt[0,3],D_voigt[0,4]],\
                               [D_voigt[0,3],D_voigt[0,1],D_voigt[0,5]],\
                               [D_voigt[0,4],D_voigt[0,5],D_voigt[0,2]]]),\
                   as_matrix([[D_voigt[0,3],D_voigt[3,3],D_voigt[3,4]],\
                               [D_voigt[3,3],D_voigt[1,3],D_voigt[3,5]],\
                               [D_voigt[3,4],D_voigt[3,5],D_voigt[2,3]]]),\
                   as_matrix([[D_voigt[0,4],D_voigt[3,4],D_voigt[4,4]],\
                               [D_voigt[3,4],D_voigt[1,4],D_voigt[4,5]],\
                               [D_voigt[4,4],D_voigt[4,5],D_voigt[2,4]]])],\
                   [as_matrix([[D_voigt[0,3],D_voigt[3,3],D_voigt[3,4]],\
                               [D_voigt[3,3],D_voigt[1,3],D_voigt[3,5]],\
                               [D_voigt[3,4],D_voigt[3,5],D_voigt[2,3]]]),\
                    as_matrix([[D_voigt[0,1],D_voigt[1,3],D_voigt[1,4]],\
                               [D_voigt[1,3],D_voigt[1,1],D_voigt[1,5]],\
                               [D_voigt[1,4],D_voigt[1,5],D_voigt[1,2]]]),\
                    as_matrix([[D_voigt[0,5],D_voigt[3,5],D_voigt[4,5]],\
                               [D_voigt[3,5],D_voigt[1,5],D_voigt[5,5]],\
                               [D_voigt[4,5],D_voigt[5,5],D_voigt[2,5]]])],\
                   [as_matrix([[D_voigt[0,4],D_voigt[3,4],D_voigt[4,4]],\
                               [D_voigt[3,4],D_voigt[1,4],D_voigt[4,5]],\
                               [D_voigt[4,4],D_voigt[4,5],D_voigt[2,4]]]),\
                    as_matrix([[D_voigt[0,5],D_voigt[3,5],D_voigt[4,5]],\
                               [D_voigt[3,5],D_voigt[1,5],D_voigt[5,5]],\
                               [D_voigt[4,5],D_voigt[5,5],D_voigt[2,5]]]),\
                    as_matrix([[D_voigt[0,2],D_voigt[2,3],D_voigt[2,4]],\
                               [D_voigt[2,3],D_voigt[1,2],D_voigt[2,5]],\
                               [D_voigt[2,4],D_voigt[2,5],D_voigt[2,2]]])]   ])
    return D

def tensor2voigt(D):
    
    return

def getConstitutiveField():

    return
def assignMaterialProps(mesh,mats):
    '''
    Input: 
        mesh: mesh including subdomain labels
        mats: material constitutive tensor for each  

    '''
    return


def getBeamProps(domain,geo,mat,xc='rectangular',):
    if xc == 'rectangular':
        [thick,width] = geo
        [E,nu]=mat

        G = E/2/(1+nu)
        S = thick*width
        ES = E*S
        EI1 = E*width*thick**3/12
        EI2 = E*width**3*thick/12
        GJ = G*0.26*thick*width**3
        kappa = Constant(domain,5./6.)
        GS1 = kappa*G*S
        GS2 = kappa*G*S

        return [S,ES,GS1,GS2,GJ,EI1,EI2]