import numpy as np

num_states = 100

if __name__ == '__main__':

    import csdl_alpha as csdl

    # custom paraboloid model
    class CFD(csdl.experimental.CustomImplicitOperation):
        def __init__(self, mesh):
            super().__init__()
            
            # assign parameters to the class
            self.mesh = mesh

        def evaluate(self, Ma, aoa):
            # set inputs using self.declare_input
            self.declare_input('Ma', Ma)
            self.declare_input('aoa', aoa)
            # add dynamic mesh here is mesh is dynamic

            # declare state variables
            u = self.create_output('u', (num_states,))
            return u
        
        def solve_residual_equations(self, input_vals, output_vals):
            Ma_np = input_vals['Ma'] # numpy array
            aoa_np = input_vals['aoa'] # numpy array

            # Forward evaluation CFD

            # Example:
            # u = compute_su2(Ma_np, aoa_np, self.mesh)

            # Dummy example quadratic polynomial
            # solve residual  0 = Ma*u^2.0 + aoa using quadratic formula
            u_solved = np.sqrt(-4*aoa_np*Ma_np) / (2*Ma_np)
            output_vals['u'] = u_solved

        def apply_inverse_jacobian(self, input_vals, outputs, d_outputs, d_residuals, mode):
            Ma_np = input_vals['Ma'] # numpy array
            aoa_np = input_vals['aoa'] # numpy array
            u = outputs['u'] # numpy array

            # for mode = rev:
            # d_outputs --> d_residuals
            if mode == 'rev':
                # compute d_residuals = (dr_du^-1)*d_outputs
                dr_du_inv_diags = 1.0/(2*Ma_np*u*np.ones((num_states,)))
                dr_du_inv = np.diag(dr_du_inv_diags)
                d_residuals['u'] = dr_du_inv@d_outputs['u']
            # ignore mode = fwd

        def compute_jacvec_product(self, input_vals, outputs, d_inputs, d_outputs, d_residuals, mode):
            Ma_np = input_vals['Ma'] # numpy array
            aoa_np = input_vals['aoa'] # numpy array
            u = outputs['u'] # numpy array

            # for mode = rev
            # d_residuals --> d_inputs
            if mode == 'rev':
                # compute d_input = (dr_dinput)*d_residuals
                d_inputs['Ma'] = np.sum(u**2.0*d_residuals['u'])
                d_inputs['aoa'] = np.sum(d_residuals['u'])
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    Ma = csdl.Variable(value = 1.0, name='Ma')
    aoa = csdl.Variable(value = -2.0, name='aoa')

    # Tell CSDL inputs and outputs
    # Mesh can be any python object
    cfd = CFD(mesh=0.0)
    u = cfd.evaluate(Ma, aoa)

    # Set as optijmization vatiables
    Ma.set_as_design_variable()
    aoa.set_as_design_variable()
    (csdl.sum(u)).set_as_objective()

    recorder.stop()

    # Uncomment to run and check derivatives via finite difference
    # sim = csdl.experimental.PySimulator(recorder)
    # sim.run()
    # sim.check_optimization_derivatives(raise_on_error=True)