from ufl import (VectorElement,MixedElement,TrialFunction,TestFunction,split)
from dolfinx.fem import (FunctionSpace)

from FROOT_BAT import material


class CrossSection:
    def __init__(self, msh, material):
        self.msh = msh
        self.material = material

    def getXCStiffnessMatrix(self):
        
        # Construct Displacment Coefficient mixed function space
        Ve = VectorElement("CG",self.msh.ufl_cell(),1,dim=3)
        #TODO:check on whether TensorFxnSpace is more suitable for this
        V = FunctionSpace(self.msh, MixedElement([Ve,Ve,Ve,Ve]))
        
        #displacement and test functions
        u = TrialFunction(V)
        v = TestFunction(V)

        #displacement coefficient trial functions
        ubar,uhat,utilde,ubreve=split(u)

        #displacement coefficient test functions
        vbar,vhat,vtilde,vbreve=split(v)

        C = self.constructMatTensorField(self)



    def getXCMassMatrix(self):


    def constructMatTensorField(self):
        D = 'matConstTensor'+'rotation_from_material_data'
        return D
    
    def getResidual(self):
        Res = 'uflexpression'

    def getModes(self):

    def decoupleModes(self):

    def computeXCStiffnessMat(self):