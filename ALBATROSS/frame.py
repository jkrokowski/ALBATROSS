'''
The Frame model
--------------------------------------
Used for connecting multiple beams together, joining their dofs,
and solving a model comprised of multiple different meshes
'''

import pyvista
from dolfinx import plot
import numpy as np
from dolfinx.fem import Function
from petsc4py import PETSc
from scipy.sparse import csr_matrix
from scipy.linalg import block_diag


class Frame():
    def __init__(self,Beams):
        #intialize the list of members of the frame
        self.Members = []
        for Beam in Beams:
            self.add_beam(Beam)
        #intialize the list of connections information
        self.Connections = []

    def add_beam(self,Beam):
        #add a beam to the list of members of the frame
        self.Members.append(Beam)

    def add_connection(self,cxn_members,cxn_pt,cxn_type='rigid'):
        '''
        adds a dictionary to the list of connections with the following information:
            -key:value pairs detailing the index number of each member in the self.Members list
              and the dofs associated with the connection at the point for each member
        connection types:
            -rigid 6-dof
            -rigid translation (hinged)
            -???

        by default, we can search for overlapping geometric points, 
        then fix those related dofs to match

        another type of dof would be a "rigid link", where the axes'
        are non-intersecting, but perscribed.

        A third, more challenging style is the application of a flexible joint
        This could be implemented with a penalty method, among other approaches
        '''
        #note, this process is a "global assembly" of all the beam subsystems
        #1. construct systems for beams
        # for beam in self.Members: 
        #     beam._construct_system()
        #since we have constructed axial models in a global coordinate frame,
        #    we can simply locate match the dofs that are geometrical coincident
        #1. identify beams to be connected
        # for cxn_member in cxn_members:
        #     for member in self.Members:
        #         if cxn_member==member:
        #             print("TRUE")
        #         else:
        #             print("FALSE")
        #2. locate dofs at location on each beam
        cxn = {}
        for cxn_member in cxn_members:
            cxn_member_disp_dofs = cxn_member._get_dofs(cxn_pt,'disp')
            cxn_member_rot_dofs = cxn_member._get_dofs(cxn_pt,'rot')
            beam_number = self.Members.index(cxn_member)
            cxn[beam_number]= np.concatenate([cxn_member_disp_dofs,cxn_member_rot_dofs])
        #3. connect dofs based on rotation of coordinate frame
        #store matching dofs to reduce system
        # cxn = {cxn_member1:cxn_member1_dofs,cxn_member2:cxn_member2_dofs}
        self.Connections.append(cxn)

    def solve(self):
        #TODO: PETSc implementation of all the below stuff
        #assemble all subsystems
        Alist = []
        blist = []
        size = 0
        for beam in self.Members: 
            beam._construct_system() 

            #initialize function to store solution of assembled system:
            beam.uh = Function(beam.beam_element.W)
            # beam.uvec = beam.uh.vector #petsc vector
            # beam.uvec.setUp()
            beam.Adense = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size).toarray()
            # beam.Acsr = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size)
            beam.b_vec = beam.b.array
            print(beam.Adense.shape)
            print(type(beam.b_vec))
            Alist.append(beam.Adense)
            blist.append(beam.b_vec)
            # size+=beam.Adense.shape[0]

        #TODO: account for multiple beams connected at the same point
            #currently, this only accounts for a connection btwn two beams
        # size -= len(self.Connections)*        
            
        #use scipy block_diag to construct overall numpy array and stack the vectors, then solve system

        #reduce the arrays and vectors for one of the matrices that is connected
        #this should just be done by progressivly adding the list of dofs to reduce
        for cxn in self.Connections:
            member_to_reduce = max(cxn.keys())
            self.Members[member_to_reduce].Adense
        # np.block()
        #build overall system from each beam:
        # for cxn in self.Connections:
        #     print()
        
        # self.A = self.assemble_lhs()
        # self.b = self.assemble_rhs()

        # ksp = PETSc.KSP().create()
        # ksp.setType(PETSc.KSP.Type.CG)
        # ksp.setTolerances(rtol=1e-15)
        # ksp.setOperators(self.A)
        # # ksp.setFromOptions()
        # ksp.solve(self.b,uvec)


    def plot_frame(self):
        pyvista.global_theme.background = [255, 255, 255, 255]
        pyvista.global_theme.font.color = 'black'
        plotter = pyvista.Plotter()
        #plot mesh
        
        for member in self.Members:
            msh = member.axial_mesh
            tdim = msh.topology.dim
            topology, cell_types, geom = plot.create_vtk_mesh(msh, tdim)
            grid = pyvista.UnstructuredGrid(topology, cell_types, geom)
            # plotter.add_mesh(grid,show_edges=True,opacity=0.25)
            plotter.add_mesh(grid,color='k',show_edges=True)
            # if add_nodes==True:
            #     plotter.add_mesh(grid, style='points',color='k')
        plotter.view_isometric()
        plotter.add_axes()
        if not pyvista.OFF_SCREEN:
            plotter.show() 