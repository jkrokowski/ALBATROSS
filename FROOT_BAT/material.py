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
    gamma = 1 / (1-2*nu12-2*nu23-2*nu13-2*nu12*nu13*nu23)

    D1111 = E11*(1-2*nu23)*gamma
    D2222 = E22*(1-2*nu13)*gamma
    D3333 = E33*(1-2*nu12)*gamma
    D1122 = E11*(nu12+nu13*nu23)*gamma
    D1133 = E11*(nu13+nu12*nu23)*gamma
    D2233 = E22*(nu23+nu12*nu13)*gamma
    D1212 = G12
    D1313 = G13
    D2323 = G23

    D = as_tensor([[as_matrix(diag(as_vector([D1111,D1122,D1133]))),\
                   as_matrix(),\
                   as_matrix()],\
                   [as_matrix(diag(as_vector([D1122,D2222,D2233]))),\
                   as_matrix(),\
                   as_matrix()],\
                   [as_matrix(diag(as_vector([D1133,D2233,D3333]))),\
                   as_matrix(),\
                   as_matrix()]])
    return

def voigt2tensor(D):
    return
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