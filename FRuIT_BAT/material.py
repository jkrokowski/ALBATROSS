
from dolfinx.fem import Constant

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