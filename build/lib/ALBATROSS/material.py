from dolfinx.fem import Constant
from ufl import Identity,as_tensor,indices,diag,as_vector,as_matrix

class Material:
    def __init__(self,name=None,mat_type='ISOTROPIC',mech_props=None,density=None,celltag=None):
        self.name = name
        self.type = mat_type
        if mech_props == None:
            raise Exception("No mechanical properties provided.")
        if mat_type == 'ISOTROPIC':
            self.E = mech_props['E']
            self.nu = mech_props['nu']
        if density!=None:
            self.density=density
        if celltag !=None:
            self.id = celltag
        elif celltag == None:
            #default to 0
            self.id = 0

def getMatConstitutive(mesh,material):
    if material.type == 'ISOTROPIC':
        return getMatConstitutiveIsotropic(mesh,material.E,material.nu)
    elif material.type == 'ORTHOTROPIC':
        mat_consts = [material.E1,
                      material.E2,
                      material.E3,
                      material.G12,
                      material.G13,
                      material.G23,
                      material.nu12,
                      material.nu13,
                      material.nu23]
        return getMatConstitutiveOrthotropic(mesh,mat_consts)
    else:
        return 'ERROR: please use existing material model'
    
def getMatConstitutiveIsotropic(mesh,E,nu):
    #lame parameters from Young's Modulus and Poisson Ratio
    _lam = (E*nu)/((1+nu)*(1-2*nu))
    mu = (E/(2*(1+nu)))

    #elasticity tensor construction
    delta = Identity(3)
    i,j,k,l=indices(4)
    C = as_tensor(_lam*(delta[i,j]*delta[k,l]) \
                    + mu*(delta[i,k]*delta[j,l]+ delta[i,l]*delta[j,k])  ,(i,j,k,l))

    return C

def getMatConstitutiveOrthotropic(E11,E22,E33,G12,G13,G23,nu12,nu13,nu23):
    
    nu21 = (E22/E11)*nu12
    nu31 = (E33/E11)*nu13
    nu32 = (E33/E22)*nu23
    
    gamma = 1 / (1-nu12*nu21-nu23*nu23-nu13*nu31-2*nu21*nu32*nu13)

    C1111 = E11*(1-nu23*nu32)*gamma
    C2222 = E22*(1-nu13*nu31)*gamma
    C3333 = E33*(1-nu12*nu21)*gamma
    C1122 = E11*(nu21+nu31*nu23)*gamma
    C1133 = E11*(nu31+nu21*nu32)*gamma
    C2233 = E22*(nu32+nu12*nu31)*gamma
    C1212 = G12
    C1313 = G13
    C2323 = G23

    C_voigt = as_matrix([[C1111,C1122,C1133,0,    0,    0    ],
                         [C1122,C2222,C2233,0,    0,    0    ],
                         [C1133,C2233,C3333,0,    0,    0    ],
                         [0,    0,    0,    C1212,0,    0    ],
                         [0,    0,    0,    0,    C1313,0    ],
                         [0,    0,    0,    0,    0,    C2323]])
    
    C = voigt2tensor(C_voigt)

    return C

def voigt2tensor(C_voigt):
    '''converts a 6x6 matrix into the appropriately constructed
      3 dimensional, Fourth order tensor
      '''
    
    C = as_tensor([[as_matrix([[C_voigt[0,0],C_voigt[0,3],C_voigt[0,4]],\
                               [C_voigt[0,3],C_voigt[0,1],C_voigt[0,5]],\
                               [C_voigt[0,4],C_voigt[0,5],C_voigt[0,2]]]),\
                   as_matrix([[C_voigt[0,3],C_voigt[3,3],C_voigt[3,4]],\
                               [C_voigt[3,3],C_voigt[1,3],C_voigt[3,5]],\
                               [C_voigt[3,4],C_voigt[3,5],C_voigt[2,3]]]),\
                   as_matrix([[C_voigt[0,4],C_voigt[3,4],C_voigt[4,4]],\
                               [C_voigt[3,4],C_voigt[1,4],C_voigt[4,5]],\
                               [C_voigt[4,4],C_voigt[4,5],C_voigt[2,4]]])],\
                   [as_matrix([[C_voigt[0,3],C_voigt[3,3],C_voigt[3,4]],\
                               [C_voigt[3,3],C_voigt[1,3],C_voigt[3,5]],\
                               [C_voigt[3,4],C_voigt[3,5],C_voigt[2,3]]]),\
                    as_matrix([[C_voigt[0,1],C_voigt[1,3],C_voigt[1,4]],\
                               [C_voigt[1,3],C_voigt[1,1],C_voigt[1,5]],\
                               [C_voigt[1,4],C_voigt[1,5],C_voigt[1,2]]]),\
                    as_matrix([[C_voigt[0,5],C_voigt[3,5],C_voigt[4,5]],\
                               [C_voigt[3,5],C_voigt[1,5],C_voigt[5,5]],\
                               [C_voigt[4,5],C_voigt[5,5],C_voigt[2,5]]])],\
                   [as_matrix([[C_voigt[0,4],C_voigt[3,4],C_voigt[4,4]],\
                               [C_voigt[3,4],C_voigt[1,4],C_voigt[4,5]],\
                               [C_voigt[4,4],C_voigt[4,5],C_voigt[2,4]]]),\
                    as_matrix([[C_voigt[0,5],C_voigt[3,5],C_voigt[4,5]],\
                               [C_voigt[3,5],C_voigt[1,5],C_voigt[5,5]],\
                               [C_voigt[4,5],C_voigt[5,5],C_voigt[2,5]]]),\
                    as_matrix([[C_voigt[0,2],C_voigt[2,3],C_voigt[2,4]],\
                               [C_voigt[2,3],C_voigt[1,2],C_voigt[2,5]],\
                               [C_voigt[2,4],C_voigt[2,5],C_voigt[2,2]]])]   ])
    return C

def tensor2voigt(C):
    
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