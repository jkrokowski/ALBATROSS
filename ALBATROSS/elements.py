''''
Elements module
---------------
Defines the finite element spaces with user-defined element type and
 quadrature degrees
'''
from dolfinx.fem import functionspace
from ufl import dx
from basix.ufl import element,mixed_element

class LinearTimoshenkoElement():

    def __init__(self, domain, element_type=None, 
                                quad_data=None):
        self.domain = domain
        self.cell = domain.ufl_cell()
        if element_type is None:
            self.element_type = "CG1" # default
        else:
            self.element_type = element_type
            
        self.W = self.setUpFunctionSpace()

        self.dx_beam = self.getQuadratureRule(quad_data)
        
    
    def setUpFunctionSpace(self):
    
        """
        Set up function space and the order of integration, with the first 
        vector element being nodal displacement, and the second vector 
        element being linearized rotation.
        """
        
        domain = self.domain
        cell = self.cell
        element_type = self.element_type
        W = None
            
        if(element_type == "CG1"):
            # ------ CG2-CG1 ----------
            Ue =  element("CG", domain.topology.cell_name(), 1, shape=(domain.geometry.dim,))
            W = functionspace(domain,mixed_element([Ue,Ue]))
            
        else:
            print("Invalid element type.")
            
        return W
    
    def getQuadratureRule(self, quad_data):

        """
        Returns a list of the cell integrals for 3 displacement
        and 3  energy with given quadrature degrees.
        
        Shear locking is possible, so we use reduced integration by 
        setting the integration measure associated with the "shear DOFs".
        The axial displacement and rotational DOFs integration measures 
        remain at the default "dx"

        One can optionally set the quadrature degree by providing a ordered list 
        ordered by: [ux,uy,uz,thetax,thetay,thetaz] (similar to the default case below)
        """
        
        if quad_data is None:
            dx_shear = dx(scheme="default",metadata={"quadrature_scheme":"default", "quadrature_degree": 1})
            dx_beam = [dx, dx_shear, dx_shear,
                  dx, dx,       dx]
                
        else: 
            dx_beam = []
            for i in range(6):
                dx_beam.append(dx(metadata={"quadrature_degree":quad_data}))
        
        return dx_beam
        
  