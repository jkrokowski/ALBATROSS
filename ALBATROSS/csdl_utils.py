import csdl_alpha as csdl
import ALBATROSS
# from scipy.spatial import cKDTree
import numpy as np
from dolfinx.io import XDMFFile
from mpi4py import MPI

# custom cross-sectional model
class CrossSection(csdl.CustomExplicitOperation):

    def __init__(self, 
                 domain,
                 xs_analysis_type,
                 material_type,
                 material_name,
                 mech_props,
                 boundary_nodes=None,
                 interior_nodes=None):
        
        super().__init__()

        self.domain = domain
        self.xs_analysis_type = xs_analysis_type
        self.material_type = material_type
        self.material_name =material_name
        self.mech_props = mech_props

        self.material = ALBATROSS.material.Material(name=self.material_name,
                            mat_type=self.material_type,
                            mech_props=self.mech_props,
                            density=2700)
        
        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes

        self.xs = ALBATROSS.cross_section.CrossSection(self.domain,[self.material])

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        # declare output variables
        if self.xs_analysis_type == 'TS':
            shape = (6,6)
        elif self.xs_analysis_type == 'EB':
            shape = (4,4)
        K = self.create_output('K', shape)
        A = self.create_output('A',(1,))

        # construct output of the model
        output = csdl.VariableGroup()
        output.K = K

        output.A = A

        return output
    
    def compute(self, inputs, outputs):     
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=inputs['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
        outputs['K'] = self.xs.K
        outputs['A'] = self.xs.A

    def compute_derivatives(self, inputs, outputs_vals, derivatives):
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.domain.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.domain.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        
        else: 
            self.domain.geometry.x[:,0:2]=inputs['xy']
        
        if self.xs_analysis_type == 'TS':
            self.xs.get_xs_stiffness_matrix()
            self.xs.compute_xs_stiffness_matrix_sensitivities()
            #TODO: need to restrict to just derivatives on boundary
            if self.boundary_nodes is not None: 
                # derivatives['K', 'xy'] = xs.dKdx_boundary
                derivatives['K', 'xy'] = self.xs.dKdx_boundary.reshape((36,inputs['xy'].flatten().shape[0]))
            else: 
                derivatives['K', 'xy'] = self.xs.dKdx
            # derivatives['K', 'xy'] = xs.dKdx.reshape((36,xy.flatten().shape[0]))
            # derivatives['K', 'xy'] = xs.dKdx.reshape((xy.flatten().shape[0],36))
        elif self.xs_analysis_type == 'EB':
            self.xs.get_xs_stiffness_matrix_EB()
            self.xs.compute_xs_stiffness_matrix_sensitivities_EB()

class WarpingFunctionState(csdl.experimental.CustomImplicitOperation):
    '''
    inputs: nodal positions of cross-sectional mesh'''
    def __init__(self,xs,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs

        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes

    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.w = self.create_output('w', (self.xs.V.dofmap.index_map.size_global,6))
        outputs.lmbda = self.create_output('lmbda', (self.xs.LM.value_size,6))

        return outputs
    
    def solve_residual_equations(self, inputs, outputs):
        print("solve residual equations:")
        #update boundary nodes:
        if self.boundary_nodes is not None: 
            self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
        #update interior nodes
        if self.interior_nodes is not None: 
            self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
        else: 
            self.xs.msh.geometry.x[:,0:2]=inputs['xy']

        #compute warping functions
        self.xs._get_warping_functions()
        
        outputs['w'] = np.vstack([self.xs.warping_functions[i].x.array for i in range(6)]).T
        outputs['lmbda'] = np.vstack([self.xs.lmbdas[i].x.array for i in range(6)]).T
    
    def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
        # print("apply_inverse_jacobian:")
        #TODO: do we need to update the inputs, etc (eg. does the mesh update need to happen here?)
        xy = inputs['xy']
        xy_interior = inputs['xy_interior']
        w = outputs['w']
        lmbda = outputs['lmbda']

        # for mode = rev:
        # d_outputs --> d_residuals
    
        if mode == 'rev':    
            # compute d_residuals = (dr_du^-1)*d_outputs

            # dr_du is simply the finite element stiffness matrix in this case
            # these are just the A00, A01, and A10 blocks of the assembled stiffness matrix
            # we can leverage the already existing ksp solver and compute the multMatTranspose() using petsc,
            # then we output these two terms to numpy matrices
            d_residuals['w'],d_residuals['lmbda'] = self.xs.apply_inverse_jacobian(d_outputs['w'],d_outputs['lmbda'])
            # #which does this under the hood: 
            #     d_outputs_petsc = stack(d_outputs['w'],d_outputs['lmbda'])
            #     self.xs.solver.solveTranspose(d_outputs_petsc,d_residuals_petsc)
            #     d_residuals_numpy = convert_petsc_to_numpy(d_residuals_petsc)
            #====
            # d_residuals['w'] =  d_residuals_numpy[w_slice]
            # d_residuals['lmbda']  = d_residuals_numpy[lmbda_slice]

            # d_residuals['w'] = drw_dw_inv @ d_outputs['w'] + drw_dl_inv @ d_outputs['lmbda']
            # d_residuals['lmbda'] = drl_dl_inv @ d_outputs['lmbda'] # + drl_dw_inv @ d_outputs['w'] <-- this term is = 


    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
        # print("compute vector-jacobian product:")
        xy = inputs['xy']
        xy_interior = inputs['xy_interior']
        w = outputs['w']
        lmbda = outputs['lmbda']

        # for mode = rev
        # d_residuals --> d_inputs
        if mode == 'rev':
            # compute d_input = (dr_dinput)*d_residuals
            #TODO: can also just return d_inputs as a numpy matrix here and prevent the memory overhead of converting to numpy,etc
            # dRwdx,dRldx = self.xs.compute_dRdx() #return numpy matrices 

            # d_inputs = dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']

            # d_inputs['xy'] = d_inputs['boundary']
            # d_inputs['xy_interior'] = d_inputs['interior']

            dRdx_dr = self.xs.compute_VJP(d_residuals['w'],d_residuals['lmbda'])

            #TODO: map to boundary or interior nodes
            d_inputs['xy'] = np.vstack([dRdx_dr[self.xs.dofs_x_boundary],
                                        dRdx_dr[self.xs.dofs_y_boundary]]).T
            d_inputs['xy_interior'] = np.vstack([dRdx_dr[self.xs.dofs_x_interior],
                                                 dRdx_dr[self.xs.dofs_y_interior]]).T
            
            # d_inputs['xy'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda'] )['boundary']
            # d_inputs['xy_interior'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']) ['interior']



class BeamMatrixFromWarping(csdl.CustomExplicitOperation):
    def __init__(self,xs,boundary_nodes=None,interior_nodes=None,check_partials='False'):
        super().__init__()
        self.xs = xs
        self.check_partials =check_partials

        if boundary_nodes is not None:
            self.boundary_nodes = boundary_nodes

        if interior_nodes is not None:
            self.interior_nodes = interior_nodes


    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        if self.check_partials != 'w':
            self.declare_input('xy',inputs.xy)
            self.declare_input('xy_interior',inputs.xy_interior)
        if self.check_partials != 'x':
            self.declare_input('w',inputs.w)
            self.declare_input('lmbda',inputs.lmbda)
        
        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K', (6,6))
        outputs.K.name = 'beam stiffness matrix'
        outputs.A = self.create_output('A',(1,))
        outputs.A.name = 'beam xs area'

        return outputs
    
    def compute(self, inputs, outputs):
        print('compute beam matrix from warping function state')
        if self.check_partials != 'w':
            #update boundary nodes:
            if self.boundary_nodes is not None: 
                self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

            #update interior nodes
            if self.interior_nodes is not None: 
                self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
            else: 
                self.xs.msh.geometry.x[:,0:2]=inputs['xy']

        if self.check_partials != 'x':
            for i in range(6):
                self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
                self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        
        # self.xs.plot_mesh()
        self.xs._compute_xs_stiffness_matrix()

        outputs['K'] = self.xs.K
        outputs['A'] = self.xs.A
    
    # def compute_derivatives(self, inputs, outputs, derivatives):
    #     print('compute beam matrix derivatives...')
    #     #update boundary nodes:
    #     # if self.boundary_nodes is not None: 
    #     #     self.xs.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']
        
    #     # #update interior nodes
    #     # if self.interior_nodes is not None: 
    #     #     self.xs.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']
    #     # else: 
    #     #     self.xs.msh.geometry.x[:,0:2]=inputs['xy']
        
    #     # for i in range(6):
    #     #     self.xs.warping_functions[i].x.array[:] = inputs['w'][:,i]
    #     #     self.xs.lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        
    #     # self.xs._compute_xs_stiffness_matrix()
    #     if self.check_partials != 'w':                
    #         pKpx = self.xs.compute_pKpx()
    #         derivatives['K', 'xy'] = pKpx[:,self.xs.dofs_boundary]
    #         derivatives['K', 'xy_interior'] = pKpx[:,self.xs.dofs_interior]
        
    #     if self.check_partials != 'x':
    #         pKpw = self.xs.compute_pKpw()
    #         pKpl = self.xs.compute_pKpl()
    #         derivatives['K', 'w'] = pKpw #return (36 x num_warping_function_dofs*6) but need to be ordered  
    #         derivatives['K', 'lmbda'] = pKpl #return (36 x 30*6)
    
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        # dxA = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dxB = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dwA = np.zeros_like(self.d_outputs['w_A'])
        # dwB = np.zeros_like(self.d_outputs['w_B'])
        # dlmbda = np.zeros_like(self.d_outputs['lmbda'])
        
        if self.check_partials != 'w':
            dx = self.xs._compute_pK_action(d_outputs['K'],
                                            derivative_type='x')

            d_inputs['xy'] = np.vstack([dx[self.xs.dofs_x_boundary],
                                            dx[self.xs.dofs_y_boundary]]).T
            d_inputs['xy_interior'] = np.vstack([dx[self.xs.dofs_x_interior],
                                            dx[self.xs.dofs_y_interior]]).T
            
        if self.check_partials != 'x':
            d_inputs['w'] = self.xs._compute_pK_action(d_outputs['K'],
                                                        derivative_type='w')
            d_inputs['lmbda'] = self.xs._compute_pK_action(d_outputs['K'],
                                                        derivative_type='l')

        # print('input vals:')
        # print(inputs['xy'])
        # print('mesh coords:')
        # print(self.xs.msh.geometry.x[:,0:2])
        
        #declare derivatives
        
        # derivatives['K', 'xy'] = pKpx[:,np.concatenate([self.xs.dofs_x_boundary,self.xs.dofs_y_boundary])]
        # derivatives['K', 'xy'] = np.hstack([pKpx[:,self.xs.dofs_x_boundary],
        #                                 pKpx[:,self.xs.dofs_y_boundary]]) #return (36 x num_boundary_nodes*2)
        # derivatives['K', 'xy_interior'] = np.hstack([pKpx[:,self.xs.dofs_x_interior],
        #                                 pKpx[:,self.xs.dofs_y_interior]]) #return (36 x num_interior_nodes*2)

class EllipticSmoothing(csdl.CustomExplicitOperation):

    def __init__(self,domain,boundary_nodes,interior_nodes,filename='xs'):
        super().__init__()
        self.domain = domain
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes
        self.step = 0.0
        self.filename =filename


    def evaluate(self, inputs: csdl.VariableGroup):
        #boundary node position inputs:
        self.declare_input('xy',inputs.xy)
        # self.declare_input('xy_interior',inputs.xy)

        # construct output of the model
        output = csdl.VariableGroup()

        #create output for new interior node positions
        output.xy_interior = self.create_output('xy_interior',(self.interior_nodes.shape[0],2))
        # output.xy_interior.name = 'xy_interior, output'

        return output

    def compute(self, inputs, outputs):
        print(f'perform elliptic smoothing (step {self.step})')
        # displacement = inputs['xy']-inputs['xy_prev']
        displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]
        
        #
        xy_interior = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                    self.boundary_nodes,
                                                    displacement,
                                                    self.interior_nodes,
                                                    mode='lin_elas',
                                                    plot_result=False,
                                                    step=self.step,
                                                    filename=self.filename)
        self.step += 1.0

        outputs['xy_interior']=xy_interior

    def compute_derivatives(self, inputs, outputs, derivatives):
        #returns displacement w.r.t to boundary nodes

        displacement = inputs['xy']-self.domain.geometry.x[self.boundary_nodes,0:2]

        xy_interior,duhdx = ALBATROSS.mesh.smooth_mesh(self.domain,
                                                        self.boundary_nodes,
                                                        displacement,
                                                        self.interior_nodes,
                                                        plot_result=False,
                                                        get_deriv=True,
                                                        mode='lin_elas')

        # derivatives['xy_interior','xy'] = np.ones_like(xy_interior)
        # print('duhdx shape:')
        # print(duhdx.shape)
        derivatives['xy_interior','xy'] = duhdx
        # derivatives['xy_interior','xy'] = duhdx.reshape((xy_interior.flatten().shape[0],
        #                                                  inputs['xy'].flatten().shape[0]))

class NonmatchingInterpolationMatrix(csdl.CustomExplicitOperation):
    '''
    return the interpolation matrix that maps the dofs from mesh = mesh_id to 
    the mortar mesh corresponding to collision = collsion


    '''
    def __init__(self,xs,
                    mesh_id=0,
                    collision=(0,1),
                    foreground_boundary = None,
                    foreground_interior = None,
                    mortar_boundary = None,
                    mortar_interior = None):
        super().__init__()
        self.xs = xs
        self.mesh_id = mesh_id
        self.collision = collision

        self.foreground_boundary = foreground_boundary
        self.foreground_interior = foreground_interior
        self.mortar_boundary = mortar_boundary
        self.mortar_interior = mortar_interior

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy_foreground',inputs.xy_foreground)
        self.declare_input('xy_interior_foreground',inputs.xy_interior_foreground)
        self.declare_input('xy_mortar',inputs.xy_mortar)
        self.declare_input('xy_interior_mortar',inputs.xy_interior_mortar)

        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.P = self.create_output('P', (self.xs.collisions[self.collision].mortar_mesh.size,
                                             self.xs.system_sizes[self.mesh_id][self.mesh_id][0]))
        outputs.P.name = 'interpolation_matrix'+str(self.mesh_id)

        return outputs
    
    def compute(self, inputs, outputs):
        print('compute beam matrix from warping function state')
        #update foreground mesh:
        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_boundary,0:2]=inputs['xy_foreground']
        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.foreground_interior,0:2]=inputs['xy_interior_foreground']
        
        #update foreground mesh:
        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_boundary,0:2]=inputs['xy_mortar']
        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.mortar_interior,0:2]=inputs['xy_interior_mortar']

        #get the interpolation matrix:
        #TODO: this loops over each collision and reconstruct the interpolation matrix each time, 
        #       this can be made more efficient
        self.xs._construct_interpolation_operators()

        if self.collision.index(self.mesh_id) == 0:
            P_petsc = self.xs.collisions[self.collision].PA
        elif self.collision.index(self.mesh_id) == 1:
            P_petsc = self.xs.collisions[self.collision].PB

        outputs['P'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(P_petsc)
    
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        print('computing interpolation matrix derivatives...')
        
        #TODO: this is a sketch, these matrices are both unassembled and unexpanded
        conn_A = self.xs.XSs[self.collision[0]].V.sub(0).sub(0).collapse()[0].dofmap.list
        conn_C = self.xs.collisions[self.collision].fxn_space.sub(0).sub(0).collapse()[0].dofmap.list
        
        #need to loop over subspaces and accumulate the effect of the d_outputs entries :)
        #get subspace portion of d_outputs:
        #subspace and subsubspace numbers should match between A and C
        
        dxA = np.zeros((self.xs.XSs[self.mesh_id].msh.geometry.x.shape[0],2))
        dxC = np.zeros((self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x.shape[0],2))
        for i in range(4):
            for j in range(3):
                sub_space_A,sub_space_A_dofmap = self.xs.XSs[self.mesh_id].V.sub(i).sub(j).collapse()
                sub_space_C,sub_space_C_dofmap = self.xs.collisions[self.collision].fxn_space.sub(i).sub(j).collapse()
                dP = d_outputs['P'][np.ix_(sub_space_C_dofmap,sub_space_A_dofmap)]
                dxAij,dxCij = ALBATROSS.nonmatching_utils.action_of_geom_on_nm_interpolation_matrix(sub_space_C,
                                                                                sub_space_A,
                                                                                dP=dP)
                dxA += dxAij
                dxC += dxCij
                
        d_inputs['xy_foreground'] = dxA[self.xs.XSs[self.mesh_id].boundary_nodes,:]
        d_inputs['xy_interior_foreground'] = dxA[self.xs.XSs[self.mesh_id].interior_nodes,:]
        
        d_inputs['xy_mortar'] = dxC[self.xs.collisions[self.collision].mortar_mesh.boundary_nodes]
        d_inputs['xy_interior_mortar'] = dxC[self.xs.collisions[self.collision].mortar_mesh.interior_nodes]
        
        
        # dPdx_A  = ALBATROSS.nonmatching_utils.derivative_of_interpolation_matrix_nonmatching_meshes(target_space,
        #                                                                                             source_space,
        #                                                                                             wrt='FROM')
    
        # dPdx_C  = ALBATROSS.nonmatching_utils.derivative_of_interpolation_matrix_nonmatching_meshes(target_space,
        #                                                                                             source_space,
        #                                                                                             wrt='TO')


class CrossSectionSystemComponents(csdl.CustomExplicitOperation):
    '''
    '''
    def __init__(self,xs,mesh_id=None,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs
        self.mesh_id = mesh_id
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K',self.xs.system_sizes[self.mesh_id][self.mesh_id])
        outputs.K.name = 'foreground_stiffness_matrix_'+str(self.mesh_id)
        outputs.C = self.create_output('C',self.xs.system_sizes[-1][self.mesh_id])
        outputs.C.name = 'foreground_constraint_matrix_'+str(self.mesh_id)

        return outputs

    def compute(self,inputs,outputs):
        #update boundary nodes:
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.XSs[self.mesh_id].msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        #construct forms for uncoupled problem
        self.xs.XSs[self.mesh_id]._construct_xs_form()
        self.xs.XSs[self.mesh_id]._construct_KKT_forms()

        K_petsc = self.xs.XSs[self.mesh_id]._assemble_block([0,0])
        C_petsc = self.xs.XSs[self.mesh_id]._assemble_block([1,0])
        
        outputs['K'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(K_petsc)
        outputs['C'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(C_petsc)

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        
        '''
        Derivatives of K and C w.r.t. the spatial coordinates
        in reverse mode, this is derivative of outputs w.r.t. d_inputs
        '''

        # we can do this in an entirely matrix free manner using the ufl.derivative(), 
        # then constructing a field with the d_inputs array
        # then, use ufl.action(ufl.adjoint(FORM),d_inputs_function)
        # and fem.petsc.assemble_vector()
        print("getting system component derivatives!")

        #return the two vectors for the 
        pKpxT_dK = self.xs.XSs[self.mesh_id]._compute_vjp_component_spatial(self.xs.XSs[self.mesh_id].a_form[0][0],
                                                                            d_outputs['K'],
                                                                            self.xs.XSs[self.mesh_id].V)
        #TODO: this needs to have the proper functions (not square, so test != trial function)
        pCpxT_dC = self.xs.XSs[self.mesh_id]._compute_vjp_component_spatial(self.xs.XSs[self.mesh_id].a_form[1][0],
                                                                            d_outputs['C'],
                                                                            test_space = self.xs.XSs[self.mesh_id].LM,
                                                                            trial_space = self.xs.XSs[self.mesh_id].V)

        d_inputs_full = pKpxT_dK + pCpxT_dC

        d_inputs['xy'] = np.vstack([d_inputs_full[self.xs.XSs[self.mesh_id].dofs_x_boundary],
                                        d_inputs_full[self.xs.XSs[self.mesh_id].dofs_y_boundary]]).T
        d_inputs['xy_interior'] = np.vstack([d_inputs_full[self.xs.XSs[self.mesh_id].dofs_x_interior],
                                                 d_inputs_full[self.xs.XSs[self.mesh_id].dofs_y_interior]]).T
        
         

class CrossSectionCouplingComponents(csdl.CustomExplicitOperation):
    '''
    '''
    def __init__(self,xs,collision=None,boundary_nodes=None,interior_nodes=None):
        super().__init__()
        self.xs = xs
        self.collision = collision
        self.boundary_nodes = boundary_nodes
        self.interior_nodes = interior_nodes

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('xy',inputs.xy)
        self.declare_input('xy_interior',inputs.xy_interior)
        
        outputs = csdl.VariableGroup()
        size_c = self.xs.collisions[self.collision].mortar_mesh.size
        outputs.MC = self.create_output('MC',(size_c,size_c))
        outputs.MC.name = 'mass_coupling_matrix'+str(self.collision[0])+str(self.collision[1])
        outputs.SC = self.create_output('SC',(size_c,size_c))
        outputs.SC.name = 'boundary_coupling_matrix'+str(self.collision[0])+str(self.collision[1])

        return outputs

    def compute(self,inputs,outputs):
        #update boundary nodes:
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.boundary_nodes,0:2]=inputs['xy']

        #update interior nodes
        self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.interior_nodes,0:2]=inputs['xy_interior']

        #construct forms for uncoupled problem
        self.xs._construct_mortar_forms()
        self.xs._assemble_mortar_matrices()

        MC_petsc = self.xs.collisions[self.collision].MC
        SC_petsc = self.xs.collisions[self.collision].S_C
                
        outputs['MC'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(MC_petsc)
        outputs['SC'] = ALBATROSS.petsc_utils.convert_petsc_to_numpy(SC_petsc)

    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        
        '''
        Derivatives of M_C and S_C w.r.t. the spatial coordinates
        '''
        print('getting coupling component derivatives:')

        #return the two vectors for the 
        pMCpxT_dMC = self.xs._compute_vjp_component_spatial(self.xs.collisions[self.collision].MC_form,
                                                                            self.collision,
                                                                            d_outputs['MC'],
                                                                            test_space = self.xs.collisions[self.collision].fxn_space)
        
        pSCpxT_dSC = self.xs._compute_vjp_component_spatial(self.xs.collisions[self.collision].SC_form,
                                                                            self.collision,
                                                                            d_outputs['SC'],
                                                                            test_space = self.xs.collisions[self.collision].fxn_space)

        d_inputs_full = pMCpxT_dMC + pSCpxT_dSC

        d_inputs['xy'] = np.vstack([d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_x_boundary],
                                        d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_y_boundary]]).T
        d_inputs['xy_interior'] = np.vstack([d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_x_interior],
                                                 d_inputs_full[self.xs.collisions[self.collision].mortar_mesh.dofs_y_interior]]).T


class CoupledBeamMatrixFromWarping(csdl.CustomExplicitOperation):
    def __init__(self,xs,collision=(0,1),check_partials='False'):
        super().__init__()
        self.xs = xs
        self.check_partials = check_partials
        self.collision = collision
        


    def evaluate(self,inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        if self.check_partials != 'w':
            self.declare_input('xy_A',inputs.xy_A)
            self.declare_input('xy_A_interior',inputs.xy_A_interior)
            self.declare_input('xy_B',inputs.xy_B)
            self.declare_input('xy_B_interior',inputs.xy_B_interior)
        # self.declare_input('xy_C',inputs.xy_C)
        # self.declare_input('xy_C_interior',inputs.xy_C_interior)
        if self.check_partials != 'x':
            self.declare_input('w_A',inputs.w_A)
            self.declare_input('w_B',inputs.w_B)
            self.declare_input('lmbda',inputs.lmbda)
        
        # construct output of the model
        outputs = csdl.VariableGroup()
        outputs.K = self.create_output('K', (6,6))
        outputs.K.name = 'beam stiffness matrix'
        # outputs.A = self.create_output('A',(1,))
        # outputs.A.name = 'beam xs area'

        return outputs
    
    def compute(self, inputs, outputs):
        print('compute beam matrix from warping function state')
        if self.check_partials != 'w':

            #UPDATE FOREGROUND MESHES GEOMETRY:
            self.xs.XSs[self.collision[0]].msh.geometry.x[self.xs.XSs[self.collision[0]].boundary_nodes,0:2]=inputs['xy_A']
            self.xs.XSs[self.collision[0]].msh.geometry.x[self.xs.XSs[self.collision[0]].interior_nodes,0:2]=inputs['xy_A_interior']
            
            self.xs.XSs[self.collision[1]].msh.geometry.x[self.xs.XSs[self.collision[1]].boundary_nodes,0:2]=inputs['xy_B']
            self.xs.XSs[self.collision[1]].msh.geometry.x[self.xs.XSs[self.collision[1]].interior_nodes,0:2]=inputs['xy_B_interior']

        # #TODO: is this necessary? or is the mortar mesh just used for the warping function discovery?
        # #UPDATE MORTAR MESH GEOMETRY:
        # self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.xs.collisions[self.collision].mortar_mesh.boundary_nodes,0:2]=inputs['xy_C']
        # self.xs.collisions[self.collision].mortar_mesh.msh.geometry.x[self.xs.collisions[self.collision].mortar_mesh.interior_nodes,0:2]=inputs['xy_C_interior']
        if self.check_partials != 'x':

            #UPDATE WARPING FUNCTIONS:
            for i in range(6):
                self.xs.XSs[self.collision[0]].warping_functions[i].x.array[:] = inputs['w_A'][:,i]
                self.xs.XSs[self.collision[0]].lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
                
                self.xs.XSs[self.collision[1]].warping_functions[i].x.array[:] = inputs['w_B'][:,i]
                self.xs.XSs[self.collision[1]].lmbdas[i].x.array[:] = inputs['lmbda'][:,i]
        
        # self.xs.plot_mesh()
        self.xs._compute_xs_stiffness_matrix()

        outputs['K'] = self.xs.K
        # outputs['A'] = self.xs.A
    
    def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, mode):
        # dxA = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dxB = np.zeros_like(self.xs.XSs[self.collision[0]].msh.geometry.x[:,:2].shape)
        # dwA = np.zeros_like(self.d_outputs['w_A'])
        # dwB = np.zeros_like(self.d_outputs['w_B'])
        # dlmbda = np.zeros_like(self.d_outputs['lmbda'])
        
        if self.check_partials != 'w':
            dxA = self.xs._compute_pK_action(d_outputs['K'],
                                            mesh_id=self.collision[0],
                                            derivative_type='x')
            dxB = self.xs._compute_pK_action(d_outputs['K'],
                                            mesh_id=self.collision[1],
                                            derivative_type='x')

            d_inputs['xy_A'] = np.vstack([dxA[self.xs.XSs[self.collision[0]].dofs_x_boundary],
                                            dxA[self.xs.XSs[self.collision[0]].dofs_y_boundary]]).T
            d_inputs['xy_A_interior'] = np.vstack([dxA[self.xs.XSs[self.collision[0]].dofs_x_interior],
                                            dxA[self.xs.XSs[self.collision[0]].dofs_y_interior]]).T
            
            d_inputs['xy_B'] = np.vstack([dxB[self.xs.XSs[self.collision[1]].dofs_x_boundary],
                                            dxB[self.xs.XSs[self.collision[1]].dofs_y_boundary]]).T
            d_inputs['xy_B_interior'] = np.vstack([dxB[self.xs.XSs[self.collision[1]].dofs_x_interior],
                                            dxB[self.xs.XSs[self.collision[1]].dofs_y_interior]]).T

        if self.check_partials != 'x':
            d_inputs['w_A'] = self.xs._compute_pK_action(d_outputs['K'],
                                                        mesh_id=self.collision[0],
                                                        derivative_type='w')
            d_inputs['w_B'] = self.xs._compute_pK_action(d_outputs['K'],
                                                        mesh_id=self.collision[1],
                                                        derivative_type='w')
            d_inputs['lmbda'] = self.xs._compute_pK_action(d_outputs['K'],
                                                        mesh_id=self.collision[0],
                                                        derivative_type='l')

    # def compute_derivatives(self, inputs, outputs, derivatives):
    #     print('compute beam matrix derivatives...')
                
    #     if self.check_partials != 'w':                
    #         pKpx = self.xs.compute_pKpx()
    #         derivatives['K', 'xy'] = pKpx[:,self.xs.dofs_boundary]
    #         derivatives['K', 'xy_interior'] = pKpx[:,self.xs.dofs_interior]
        
    #     if self.check_partials != 'x':
    #         pKpw = self.xs.compute_pKpw()
    #         pKpl = self.xs.compute_pKpl()
    #         derivatives['K', 'w'] = pKpw #return (36 x num_warping_function_dofs*6) but need to be ordered  
    #         derivatives['K', 'lmbda'] = pKpl #return (36 x 30*6)

        
    #     derivatives['K', 'xy_A']
    #     derivatives['K', 'xy_A_interior']
    #     derivatives['K', 'xy_B']
    #     derivatives['K', 'xy_B_interior']
    #     # derivatives['K', 'xy_C']
    #     # derivatives['K', 'xy_C_interior']
    #     derivatives['K', 'w_A']
    #     derivatives['K', 'w_B']
    #     derivatives['K', 'lmbda']
        
        
        #declare derivatives
        
        # derivatives['K', 'xy'] = pKpx[:,np.concatenate([self.xs.dofs_x_boundary,self.xs.dofs_y_boundary])]
        # derivatives['K', 'xy'] = np.hstack([pKpx[:,self.xs.dofs_x_boundary],
        #                                 pKpx[:,self.xs.dofs_y_boundary]]) #return (36 x num_boundary_nodes*2)
        # derivatives['K', 'xy_interior'] = np.hstack([pKpx[:,self.xs.dofs_x_interior],
        #                                 pKpx[:,self.xs.dofs_y_interior]]) #return (36 x num_interior_nodes*2)

#TODO: this is a massive operation that requires a lot of validation of individual components, 
# but would perfom much better than the current coupled approach where the system matrices are provided along with derivatives 
# class WarpingFunctionStateCoupled(csdl.experimental.CustomImplicitOperation):
#     '''
#     inputs: nodal positions of cross-sectional meshes (both overlapping and mortarmesh)
#     '''
#     def __init__(self,xs):
#         super().__init__()
#         #TODO: 
#         #this is a coupled cross-section problem:
#         self.xs = xs

#     def evaluate(self,inputs: csdl.VariableGroup):
#         # assign method inputs to input dictionary
#         #TODO: this is rigidly fixed to two overlapping meshes & one intersection; make more general
#         self.declare_input('xy_A',inputs.xy_A)
#         self.declare_input('xy_A_interior',inputs.xy_A_interior)
#         self.declare_input('xy_B',inputs.xy_B)
#         self.declare_input('xy_B_interior',inputs.xy_B_interior)
#         self.declare_input('xy_C',inputs.xy_C)
#         self.declare_input('xy_C_interior',inputs.xy_C_interior)

#         # construct output of the model
#         outputs = csdl.VariableGroup()
#         outputs.w_A = self.create_output('w_A', (self.xs.XSs[0].V.dofmap.index_map.size_global,6))
#         outputs.w_B = self.create_output('w_B', (self.xs.XSs[1].V.dofmap.index_map.size_global,6))
#         outputs.lmbda = self.create_output('lmbda', (self.xs.XSs[0].LM.value_size,6))

#         return outputs
    
#     def solve_residual_equations(self, inputs, outputs):
#         print("solve residual equations:")
#         #update boundary nodes:
#         self.xs.XSs[0].msh.geometry.x[self.xs.XSs[0].boundary_nodes,0:2]=inputs['xy_A']
#         self.xs.XSs[1].msh.geometry.x[self.xs.XSs[1].boundary_nodes,0:2]=inputs['xy_B']
#         self.xs.collisions[(0,1)].mortar_mesh.msh.geometry.x[self.xs.collisions[(0,1)].mortar_mesh.boundary_nodes,0:2]=inputs['xy_C']
        
#         #update interior nodes
#         self.xs.XSs[0].msh.geometry.x[self.xs.XSs[0].interior_nodes,0:2]=inputs['xy_A_interior']
#         self.xs.XSs[1].msh.geometry.x[self.xs.XSs[1].interior_nodes,0:2]=inputs['xy_B_interior']
#         self.xs.collisions[(0,1)].mortar_mesh.msh.geometry.x[self.xs.collisions[(0,1)].mortar_mesh.interior_nodes,0:2]=inputs['xy_C_interior']
       
#         #compute warping functions
#         self.xs._get_warping_functions()
        
#         outputs['w_A'] = np.vstack([self.xs.XSs[0].warping_functions[i].x.array for i in range(6)]).T
#         outputs['w_B'] = np.vstack([self.xs.XSs[1].warping_functions[i].x.array for i in range(6)]).T
#         outputs['lmbda'] = np.vstack([self.xs.XSs[0].lmbdas[i].x.array for i in range(6)]).T
    
#     def apply_inverse_jacobian(self, inputs, outputs, d_outputs, d_residuals, mode):
#         # print("apply_inverse_jacobian:")
#         xy = inputs['xy']
#         xy_interior = inputs['xy_interior']
#         w = outputs['w']
#         lmbda = outputs['lmbda']

    
#         if mode == 'rev':    
#             # d_outputs --> d_residuals
#             # compute d_residuals = (dr_du^-1)*d_outputs

#             # dr_du is simply the finite element stiffness matrix in this case
#             # these are just the A00, A01, and A10 blocks of the assembled stiffness matrix
#             # we can leverage the already existing ksp solver and compute the multMatTranspose() using petsc,
#             # then we output these two terms to numpy matrices
#             d_residuals['w'],d_residuals['lmbda'] = self.xs.apply_inverse_jacobian(d_outputs['w'],d_outputs['lmbda'])
#             # #which does this under the hood: 
#             #     d_outputs_petsc = stack(d_outputs['w'],d_outputs['lmbda'])
#             #     self.xs.solver.solveTranspose(d_outputs_petsc,d_residuals_petsc)
#             #     d_residuals_numpy = convert_petsc_to_numpy(d_residuals_petsc)
#             #====
#             # d_residuals['w'] =  d_residuals_numpy[w_slice]
#             # d_residuals['lmbda']  = d_residuals_numpy[lmbda_slice]

#             # d_residuals['w'] = drw_dw_inv @ d_outputs['w'] + drw_dl_inv @ d_outputs['lmbda']
#             # d_residuals['lmbda'] = drl_dl_inv @ d_outputs['lmbda'] # + drl_dw_inv @ d_outputs['w'] <-- this term is = 


#     def compute_jacvec_product(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
#         # print("compute vector-jacobian product:")
#         xy = inputs['xy']
#         xy_interior = inputs['xy_interior']
#         w = outputs['w']
#         lmbda = outputs['lmbda']

#         # for mode = rev
#         # d_residuals --> d_inputs
#         if mode == 'rev':
#             # compute d_input = (dr_dinput)*d_residuals
#             #TODO: can also just return d_inputs as a numpy matrix here and prevent the memory overhead of converting to numpy,etc
#             # dRwdx,dRldx = self.xs.compute_dRdx() #return numpy matrices 

#             # d_inputs = dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']

#             # d_inputs['xy'] = d_inputs['boundary']
#             # d_inputs['xy_interior'] = d_inputs['interior']

#             dRdx_dr = self.xs.compute_VJP(d_residuals['w'],d_residuals['lmbda'])

#             #TODO: map to boundary or interior nodes
#             d_inputs['xy'] = np.vstack([dRdx_dr[self.xs.dofs_x_boundary],
#                                         dRdx_dr[self.xs.dofs_y_boundary]]).T
#             d_inputs['xy_interior'] = np.vstack([dRdx_dr[self.xs.dofs_x_interior],
#                                                  dRdx_dr[self.xs.dofs_y_interior]]).T
            
#             # d_inputs['xy'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda'] )['boundary']
#             # d_inputs['xy_interior'] = (dRwdx @ d_residuals['w'] + dRldx @ d_residuals['lmbda']) ['interior']


class BeamModel(csdl.CustomExplicitOperation):
    '''
    initialization inputs: beam axis and xs info

    evaluation inputs: loads (at some subset of points?)
    
    outputs: deflection
    '''
    def __init__(self,beam_axis,xs_info):
        super().__init__()
        self.beam_axis = beam_axis
        self.xs_info = xs_info
        
        ALBATROSS.beam.Beam(beam_axis,xs_info)

    def evaluate(self,inputs: csdl.VariableGroup):
        self.declare_input('K',inputs.xy)
    
    def compute(self, inputs, outputs):
        return super().compute(inputs, outputs)
    
    def compute_derivatives(self, inputs, outputs, derivatives):
        return super().compute_derivatives(inputs, outputs, derivatives)
    

