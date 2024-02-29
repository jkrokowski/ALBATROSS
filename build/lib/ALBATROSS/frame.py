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

    def create_frame_connectivity(self):
        #compute the number of dofs that the total system is reduced due to the connection
        num_cxn_reduction_dofs = 0
        #loop through the connections to find the number of beams at each connection 
        #   and the number of connected dofs at each
        for cxn in self.Connections:
            num_cxn_reduction_dofs += (len(cxn)-1)*len(cxn[list(cxn.keys())[0]])
        
        #store member dof numbers
        for member in self.Members:
            member.num_local_dofs=member.member.beam_element.W.dofmap.index_map.size_global

        num_dofs_unconnected = sum([member.num_local_dofs for member in self.Members])
        self.num_dofs_global = num_dofs_unconnected - num_cxn_reduction_dofs
        print(self.num_dofs_global)

        #use connection dictionary to 

        # for member in self.Members:
        #     member.global_frame_dofs = 

    def solve(self):
        #TODO: PETSc implementation of all the below stuff
        #assemble all subsystems
        Alist = []
        blist = []
        for beam in self.Members: 
            beam._construct_system() 

            #initialize function to store solution of assembled system:
            beam.uh = Function(beam.beam_element.W)
            # beam.uvec = beam.uh.vector #petsc vector
            # beam.uvec.setUp()
            beam.Adense = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size).toarray()
            # beam.Acsr = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size)
            beam.b_vec = beam.b.array

            Alist.append(beam.Adense)
            blist.append(beam.b_vec)

        #TODO: account for multiple beams connected at the same point
            #currently, this only accounts for a connection btwn two beams
            

        #reduce the arrays and vectors for one of the matrices that is connected
        #this should just be done by progressivly adding the list of dofs to reduce
        for cxn in self.Connections:
            #identify members that are in this connection
            cxn_members = list(cxn.keys())
            parent = cxn_members[0]
            children = cxn_members[1:]
            #store parent dofs
            parent_dofs = cxn[parent]
            for child in children:
                #store child dofs
                child_dofs = cxn[child]
                # self.Members[child].parent_dofs_map = cxn[parent]
                #add contribution to parent matrix
                Alist[parent][parent_dofs,parent_dofs] += Alist[child][child_dofs,child_dofs]
                #delete rows and columns of child matrix and vector
                Alist[child] = np.delete(np.delete(Alist[child],child_dofs,axis=0),child_dofs,axis=1)
                blist[child]=np.delete(blist[child],child_dofs)
                # self.Members[child].parent_dofs = parent_dofs
                # self.Members[child].child_dofs = child_dofs
        
        #construct assembled system
        Atotal = block_diag(*Alist)
        btotal = np.concatenate(blist)
        # Atotal = block_diag(*[member.Adense for member in self.Members])
        # btotal = np.concatenate([member.b_vec for member in self.Members])

        #get offsets
        # for i,member in enumerate(self.Members):
        #     if i ==0:
        #         member.dof_offset = 0
        #     else:
        #         member.dof_offset = self.Members[i-1].dof_offset + Alist[i-1].shape[0]

        #get displacement solution of system
        u = np.linalg.solve(Atotal,btotal)

        #remap solution to functions for each beam
        #for parents, simply populate the dof vector with the solution, accounting for the offset
        for i,member in enumerate(self.Members):
            if i ==0:
                member.dof_offset = 0
            else:
                member.dof_offset = self.Members[i-1].dof_offset + Alist[i-1].shape[0]

            dof_start = member.dof_offset
            dof_stop = member.dof_offset+blist[i].shape[0] 
            u_vals = u[dof_start:dof_stop]
            # if member.parent_dofs:
            #     u_vals = np.insert(u_vals,member.parent_dofs,u[member.parent_dofs])
            #this logical test looks nasty, but it determines if any member is a child at least once 
            #   so we can get the dofs for that member 
            # if i in np.array([list(cxn.keys())[1:] for cxn in self.Connections]).flatten():
            #     #get 
                
            # if member.par
            # u_member = np.insert()
            member.uh.vector.array = u_vals
        #for children

        #if the frame consists of a distinct members that are not connected (e.g. multiple frames?)
        # we need to construct


        # print(self.Members[0].uh.vector.array)
        # for i,member in enumerate(self.Members):
        #     member.uh.x.array = u[blist[i].shape[i]]

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