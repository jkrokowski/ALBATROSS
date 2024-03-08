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
        current connection types:
            -rigid 6-dof

        TODO: connection types in the future:
            -rigid translation (hinged)
            -....

        another type of dof would be a "rigid link", where the axes'
        are non-intersecting, but perscribed.

        A third, more challenging style is the application of a flexible joint
        This could be implemented with a penalty method, among other approaches
        '''
        cxn = {}
        for cxn_member in cxn_members:
            cxn_member_disp_dofs = cxn_member._get_dofs(cxn_pt,'disp')
            cxn_member_rot_dofs = cxn_member._get_dofs(cxn_pt,'rot')
            beam_number = self.Members.index(cxn_member)
            cxn[beam_number]= np.concatenate([cxn_member_disp_dofs,cxn_member_rot_dofs])
        self.Connections.append(cxn)

    def create_frame_connectivity(self):
        #store dof numbers for each member and initialize 
        #TODO: move this to the point that the beam is added to the frame?
        self.num_dofs_global=0
        self.global_offsets=[0]
        for i,member in enumerate(self.Members):
            member.num_local_dofs=member.beam_element.W.dofmap.index_map.size_global
            #initialize number of global dofs corresponding to each member
            #   these will be reduced for each connection
            member.num_global_dofs = member.num_local_dofs

            # increment the total number of global dofs
            self.num_dofs_global += member.num_local_dofs
            if i!=0:
                self.global_offsets.append(self.Members[i-1].num_local_dofs+self.global_offsets[i-1])
        
        #next, we modify the above maps by building a map from global space to a reduced global space
        #   and using the individual maps from the local space to the global space, we can build the local to the reduced global space 
        self.num_dofs_global_reduced = self.num_dofs_global
        self.reduced_dofs = []
        self.GtR = np.eye((self.num_dofs_global)) #trim this array later
        
        for cxn in self.Connections:
            #identify members that are in this connection
            cxn_members = list(cxn.keys())
            parent = cxn_members[0]
            children = cxn_members[1:]
            parent_dofs = cxn[parent]
            for child in children:
                child_dofs = cxn[child]
                #reduce number of unique global dofs for the assembled system by the number of child dofs
                self.num_dofs_global_reduced -= child_dofs.shape[0]
                #keep track of which global dofs don't exist in reduced global system
                self.reduced_dofs.extend(child_dofs+self.global_offsets[child])
                #add connection between child and parent in global incidence matrix
                self.GtR[self.global_offsets[parent]+parent_dofs,self.global_offsets[child]+child_dofs] = 1
                
        #trim global to global_reduced incidence matrix
        self.GtR = np.delete(self.GtR,self.reduced_dofs,axis=0)
        
    def solve(self):
        #TODO: PETSc implementation of all the below stuff
        #assemble all subsystems
        Alist = []
        blist = []
        Atotal = np.zeros((self.num_dofs_global_reduced,self.num_dofs_global_reduced))
        btotal = np.zeros((self.num_dofs_global_reduced))
        for i,member in enumerate(self.Members): 
            member._construct_system() 

            #initialize function to store solution of assembled system:
            member.uh = Function(member.beam_element.W)
            
            member.Adense = csr_matrix(member.A_mat.getValuesCSR()[::-1], shape=member.A_mat.size).toarray()
            # beam.Acsr = csr_matrix(beam.A_mat.getValuesCSR()[::-1], shape=beam.A_mat.size)
            member.b_vec = member.b.array

            Alist.append(member.Adense)
            blist.append(member.b_vec) 

        #construct assembled system
        Atotal = block_diag(*Alist)
        btotal = np.concatenate(blist)
        
        #reduce the system:
        Ar = self.GtR@Atotal@self.GtR.T
        br = self.GtR@btotal

        #get displacement solution of system
        ur = np.linalg.solve(Ar,br)

        #get the full solution vector
        self.u = self.GtR.T@ur

        #use offsets to populate solution values to member solution functions
        for i,member in enumerate(self.Members):
            if i == len(self.Members)-1:
                member.uh.vector.array = self.u[self.global_offsets[i]:]
            else:
                member.uh.vector.array = self.u[self.global_offsets[i]:self.global_offsets[i+1]]
        
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

    def plot_axial_displacement(self,warp_factor=1):
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
            grid.point_data["u"] = member.uh.sub(0).collapse().x.array.reshape((geom.shape[0],3))

            warped = grid.warp_by_vector("u", factor=warp_factor)
            plotter.add_mesh(warped, show_edges=True)

        plotter.view_isometric()
        plotter.show_axes()

        if not pyvista.OFF_SCREEN:
            plotter.show()
        else:
            figure = plot.screenshot("beam_mesh.png")

    def recover_displacement(self,plot_xss=False):
        for member in self.Members:
            member.recover_displacement(plot_xss=False)
    
    def recover_stress(self,):
        return