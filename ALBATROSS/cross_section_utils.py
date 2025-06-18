from ufl import split,cross,as_vector,as_tensor,sym,grad

# constraints on averages generalized stresses (forces + moments)
def rigid_constraints(u,x):
    ubar,uhat,_,_ = split(u)
    x1,x2 = x

    ubar_r = cross(as_vector([0,x1,x2]),ubar)
    # uhat_r = cross(as_vector([0,x1,x2]),uhat)

    U = [ ubar[0],                      # translation x
        ubar[1],                        # translation y
        ubar[2],                        # translation z
        ubar_r[0],      # rotation about x
        ubar_r[1],                 # rotation about y
        ubar_r[2],                 # rotation about z
        ]

    return U

# constraints on average generalized stresses (forces + moments)
def stress_constraints1(u):
    '''Constraints for average forces/moments'''

    sigma = warping2stress(u,0)
    P = stress2loads(sigma)

    return P

def stress_constraints_x1(u):
    '''Constraints for average forces/moments'''
    sigma = warping2stress(u,1)
    P = stress2loads(sigma)

    return P

def stress_constraints_x1_2(u):
    '''Constraints for average forces/moments'''
    sigma = warping2stress(u,2)
    P = stress2loads(sigma)

    return P

def stress_constraints_x1_3(u):
    '''Constraints for average forces/moments'''
    sigma = warping2stress(u,3)
    P = stress2loads(sigma)

    return P

def constraints(u):
    #order stress constraints above rigid constraints for more intuitive indexing
    P = stress_constraints1(u)
    Px1= stress_constraints_x1(u)
    Px1_2= stress_constraints_x1_2(u)
    Px1_3= stress_constraints_x1_3(u)
    U = rigid_constraints(u)

    return as_vector(P+Px1+Px1_2+Px1_3+U)


def stress_constraints2(u):
    ubar,uhat,utilde,ubreve = split(u)

    sigma=warping2stress(u,0)+warping2stress(u,1)
    P = stress2loads(sigma)

    return P


def warping2strain(u,order):
    #get list of ubar,uhat,utilde,ubreve
    try:
        u_list = split(u)
    except:
        u_list = []
        for idx in range(4):
            u_list.append(as_tensor([u[3*idx],u[3*idx+1],u[3*idx+2]]))
    
    gradu = grad(u_list[order])

    if order < 3:
        eps = sym(as_tensor([
                [(order+1)*u_list[order+1][0], gradu[0,0], gradu[0,1]],
                [(order+1)*u_list[order+1][1], gradu[1,0], gradu[1,1]],
                [(order+1)*u_list[order+1][2], gradu[2,0], gradu[2,1]],
            ]))
        
    else:
        eps = sym(as_tensor([
                [0, gradu[0,0], gradu[0,1]],
                [0, gradu[1,0], gradu[1,1]],
                [0, gradu[2,0], gradu[2,1]],
            ]))

    return eps


def warping2stress(u,order,C,indices):
    i,j,k,l=in.i,self.j,self.k,self.l

    eps = warping2strain(u,order)

    sigma = as_tensor(self.C[i,j,k,l]*eps[k,l],(i,j))

    return sigma


def stress2loads(sigma,x):
    x1,x2 = x
    sigma11 = sigma[0,0]
    sigma12 = 0.5*(sigma[1,0]+sigma[0,1])
    sigma13 = 0.5*(sigma[2,0]+sigma[0,2])

    m = cross(as_vector([0,x1,x2]),
                as_vector([sigma11,sigma12,sigma13]))
    
    P = [   sigma11,    # extension
            sigma12,    # shear 1
            sigma13,    # shear 2
            m[0],       # torsion
            m[1],       # bending 1
            m[2]        # bending 2
                ]
    
    return P
