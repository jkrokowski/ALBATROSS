from ufl import (grad,as_matrix,SpatialCoordinate,FacetNormal,Measure,as_tensor,indices,VectorElement,MixedElement,TrialFunction,TestFunction,split)
from dolfinx.fem import (sin,cos,Function,FunctionSpace,VectorFunctionSpace)

from FROOT_BAT.material import *


class CrossSection:
    def __init__(self, msh, celltags, material):
        #analysis domain
        self.msh = msh
        self.ct = celltags
        #dictionary of material properties 
        self.material = material
        '''
        dictionary defined in the following manner:
        {MATERIAL:{
            TYPE: 'ISOTROPIC', 'ORTHOTROPIC', etc
            MECH_PROPS: {}
            DENS: float } }
        ....
        '''
        
        #geometric dimension
        self.d = 3
        #number of materials
        self.num_mat = len(self.material)
        self.mat_ids = list(range(self.num_mat))
        #indices
        self.i,self.j,self.k,self.l=indices(4)
        self.p,self.q,self.r,self.s=indices(4)
        self.a,self.B = indices(2)

        #integration measures (subdomain data accounts for different materials)
        self.dx = Measure("dx",domain=self.msh,subdomain_data=self.ct)
        self.ds = Measure("ds",domain=self.msh)
        #spatial coordinate and facet normals
        self.x = SpatialCoordinate(self.msh)
        self.n = FacetNormal(self.msh)

    def getXCStiffnessMatrix(self):
        
        # Construct Displacement Coefficient mixed function space
        self.Ve = VectorElement("CG",self.msh.ufl_cell(),1,dim=3)
        self.V = FunctionSpace(self.msh, MixedElement(4*[self.Ve]))
        
        #displacement and test functions
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        #displacement coefficient trial functions
        self.ubar,self.uhat,self.utilde,self.ubreve=split(self.u)

        #displacement coefficient test functions
        self.vbar,self.vhat,self.vtilde,self.vbreve=split(self.v)
        
        #construct residual by looping through materials
        self.Residual = 0
        for mat_id in self.mat_ids:
            self.Residual += self.constructMatResidual(mat_id)

        
   
    def applyRotation(self,C,alpha,beta,gamma):
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        p,q,r,s=self.p,self.q,self.r,self.s
        #rotation about X-axis
        Rx = as_matrix([[1,         0,         0],
                        [0,cos(alpha),-sin(alpha)],
                        [0,sin(alpha),cos(alpha)]])
        #rotation about Y-axis
        Ry = as_matrix([[cos(beta), 0,sin(beta)],
                        [0,         1,        0],
                        [-sin(beta),0,cos(beta)]])
        #rotation about Z-axis
        Rz = as_matrix([[cos(gamma),-sin(gamma),0],
                        [sin(gamma),cos(gamma), 0],
                        [0,         0,          1]])
        
        #3D rotation matrix
        R = Rz*Ry*Rx

        Cprime = as_tensor(R[p,i]*R[q,j]*C[i,j,k,l]*R.T[k,r]*R.T[l,s],(p,q,r,s))

        return Cprime
    
    def constructMatOrientation(self,orientation):
        #orientation is a 
        self.Q = VectorFunctionSpace(self.msh,("DG",0),dim=3)
        self.theta = Function(self.Q)

        self.theta.interpolate(orientation)

    def constructMatResidual(self,mat_id):
        #geometric dimension
        d = self.d
        #indices
        i,j,k,l=self.i,self.j,self.k,self.l
        a,B = self.a,self.b
        #trial and test functions
        ubar,uhat,utilde,ubreve=self.ubar,self.uhat,self.utilde,self.ubreve
        vbar,vhat,vtilde,vbreve=self.vbar,self.vhat,self.vtilde,self.vbreve
        #restricted integration domain
        dx = self.dx(mat_id)
                
        C_mat = getMatConstitutive(self.material[mat_id])

        #if an orthotropic material is used, the constructMatOrientation method
        #must be called prior to applying rotations
        if self.material['type'] == 'orthotropic':
            C = self.applyRotation(C_mat,self.theta[0],self.theta[1],self.theta[2])
        elif self.material['type'] == 'isotropic':
            C = C_mat

        #sub-tensors of stiffness tensor
        Ci1k1 = as_tensor(C[i,0,k,0],(i,k))
        Ci1kB = as_tensor([[[C[i, 0, k, l] for l in [1,2]]
                    for k in range(d)] for i in range(d)])
        Ciak1 = as_tensor([[[C[i, j, k, 0] for k in range(d)]
                    for j in [1,2]] for i in range(d)])
        CiakB = as_tensor([[[[C[i, j, k, l] for l in [1,2]]
                    for k in range(d)] for j in [1,2]] 
                    for i in range(d)])
        
        #partial derivatives of displacement:
        ubar_B = grad(ubar)
        uhat_B = grad(uhat)
        utilde_B = grad(utilde)
        ubreve_B = grad(ubreve)

        #partial derivatives of shape fxn:
        vbar_a = grad(vbar)
        vhat_a = grad(vhat)
        vtilde_a = grad(vtilde)
        vbreve_a = grad(vbreve)

        # equation 1,2,3
        L1= 2*Ci1k1[i,k]*utilde[k]*vbar[i]*dx\
            + Ci1kB[i,k,B]*uhat_B[k,B]*vbar[i]*dx \
            - Ciak1[i,a,k]*uhat[k]*vbar_a[i,a]*dx \
            - CiakB[i,a,k,B]*ubar_B[k,B]*vbar_a[i,a]*dx \

        # # equation 4,5,6
        L2 = 6*Ci1k1[i,k]*ubreve[k]*vhat[i]*dx\
            + 2*Ci1kB[i,k,B]*utilde_B[k,B]*vhat[i]*dx \
            - 2*Ciak1[i,a,k]*utilde[k]*vhat_a[i,a]*dx \
            - CiakB[i,a,k,B]*uhat_B[k,B]*vhat_a[i,a]*dx \

        # equation 7,8,9
        L3 = 3*Ci1kB[i,k,B]*ubreve_B[k,B]*vtilde[i]*dx \
            - 3*Ciak1[i,a,k]*ubreve[k]*vtilde_a[i,a]*dx \
            - CiakB[i,a,k,B]*utilde_B[k,B]*vtilde_a[i,a]*dx\

        #equation 10,11,12
        L4= -CiakB[i,a,k,B]*ubreve_B[k,B]*vbreve_a[i,a]*dx\
        
        return L1+L2+L3+L4

    def getModes(self):
        return
    def decoupleModes(self):
        return
    def computeXCStiffnessMat(self):
        return
    def getXCMassMatrix(self):
        return