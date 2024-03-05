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
        #store dof numbers for each member and initialize 
        #TODO: move this to the point that the beam is added to the frame?
        self.num_dofs_global=0
        for i,member in enumerate(self.Members):
            member.num_local_dofs=member.beam_element.W.dofmap.index_map.size_global
            #initialize number of global dofs corresponding to each member
            #   these will be reduced for each connection
            member.num_global_dofs = member.num_local_dofs
            member.child_dofs = {}
            member.parent_dofs = {}
            #this is an array of length member.num_local_dofs, which maps the local dofs to the global dofs
            member.local_to_global = np.arange(self.num_dofs_global,self.num_dofs_global+member.num_local_dofs)
            print('member %i:' % i)
            print(member.local_to_global)
            #increment the total number of global dofs
            self.num_dofs_global += member.num_local_dofs
            #build local to global dof map
        
        #next, we modify the above maps by building a map from global space to a reduced global space
        #   and using the individual maps from the local space to the global space, we can build the local to the reduced global space 
        self.num_dofs_global_reduced = self.num_dofs_global
        self.reduced_dofs = []
        self.reduced_dofs_vals = []
        for cxn in self.Connections:
            #get number of dofs in this 
            # num_cxn_dofs = (len(cxn)-1)*len(cxn[list(cxn.keys())[0]])
            #identify members that are in this connection
            cxn_members = list(cxn.keys())
            parent = cxn_members[0]
            children = cxn_members[1:]
            parent_dofs = cxn[parent]
            for child in children:
                #add child dofs tuple to parent
                child_dofs = cxn[child]
                self.Members[parent].child_dofs[child] = child_dofs
                #add parent dofs tuple to child
                self.Members[child].parent_dofs[parent]= parent_dofs
                #reduce number of unique global dofs for the assembled system by the number of child dofs
                self.num_dofs_global_reduced -= child_dofs.shape[0]
                # self.Members[child].num_global_dofs -= num_cxn_dofs
                self.Members[child].local_to_global[child_dofs] = self.Members[parent].local_to_global[parent_dofs]
                #keep track of which global dofs don't exist in reduced global system
                self.reduced_dofs.extend(child_dofs)
                self.reduced_dofs_vals.extend(parent_dofs)

        #construct map from global to global reduced dofs
        #TODO: more thought may be needed here to manage the sparsity pattern
        # self.global_to_global_reduced = np.arange(self.num_dofs_global)
        # #loop through each dof and if the dof is greater than the reduced dof value, decrease all dofs by 1
        # for i,dof in enumerate(self.global_to_global_reduced):
        #     for j,reduced_dof in enumerate(self.reduced_dofs):
        #         if j>=i:
        #             dof -= 1

        #ANOTHER IDEA: construct rectangular incidence matrix to transform between global and reduced global 
        


        print(self.num_dofs_global)
        print(self.num_dofs_global_reduced)
        print('reduced dofs:')
        print(self.reduced_dofs)
        # print(self.global_to_global_reduced)

        print('after modification:')
        for i,member in enumerate(self.Members):
            print('member %i:' % i)
            print(member.local_to_global)



        #TODO: NOTHING BELOW HERE CAN BE TRUSTED
        # print()
        # print("check yourself!")
        # print()

        # #build maps from local dofs to global dofs

        # # num_dofs_unconnected = sum([member.num_local_dofs for member in self.Members])
        # # self.num_dofs_global = num_dofs_unconnected - num_cxn_dofs_total
        # # self.num_dofs_global = sum([member.num_global_dofs for member in self.Members])
        
        # #assign global degrees of freedom for each member
        # self.num_dofs_global = 0
        # for i,member in enumerate(self.Members):
        #     #populate list of dofs in global_frame_dofs property
        #     member.global_frame_dofs = np.arange(self.num_dofs_global,self.num_dofs_global+member.num_global_dofs)

        #     #increment the number of global dofs by the number of unique global dofs in the member
        #     self.num_dofs_global += member.num_global_dofs
        #     print("Member %i:" % i)
        #     print('unique global frame dofs:')
        #     print(member.global_frame_dofs)
        #     print()

        # print(self.num_dofs_global)

        # # for each beam, map local dofs to global dofs with a list of length = local_dofs with global dof values
        # #for each member
        # member.global_to_local = np.zeros((member.num_local_dofs))


        # #add global dofs of parent to global dofs of child
        # print()
        # for cxn in self.Connections:
        #     cxn_members = list(cxn.keys())
        #     parent = cxn_members[0]
        #     children = cxn_members[1:]
        #     # parent_dofs = cxn[parent]
        #     for child in children:
        #         print('indices:')
        #         print(self.Members[parent].child_dofs[child])
        #         print('vals:')
        #         print(self.Members[parent].global_frame_dofs[self.Members[child].parent_dofs[1]])
        #         self.Members[child].global_frame_dofs = np.insert(self.Members[child].global_frame_dofs,
        #                                                           self.Members[parent].child_dofs[1],
        #                                                           self.Members[parent].global_frame_dofs[self.Members[child].parent_dofs[1]])

    def solve(self):
        #TODO: PETSc implementation of all the below stuff
        #assemble all subsystems
        # Alist = []
        # blist = []
        Atotal = np.zeros((self.num_dofs_global_reduced,self.num_dofs_global_reduced))
        btotal = np.zeros((self.num_dofs_global_reduced))
        for i,member in enumerate(self.Members): 
            member._construct_system() 

            #initialize function to store solution of assembled system:
            member.uh = Function(member.beam_element.W)
            # beam.uvec = beam.uh.vector #petsc vector
            # beam.uvec.setUp()
            member.Adense = csr_matrix(member.A_mat.getValuesCSR()[::-1], shape=member.A_mat.size).toarray()
            # beam.Acsr = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size)
            member.b_vec = member.b.array

            Atotal[member.local_to_global,member.local_to_global]
            # Alist.append(beam.Adense)
            # blist.append(beam.b_vec)

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
        # Atotal = block_diag(*Alist)
        # btotal = np.concatenate(blist)
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