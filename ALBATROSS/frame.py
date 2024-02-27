'''
The Frame model
--------------------------------------
Used for connecting multiple beams together, joining their dofs,
and solving a model comprised of multiple different meshes
'''

import pyvista
from dolfinx import plot

class Frame():
    def __init__(self,Beams):
        self.Members = []
        for Beam in Beams:
            self.add_beam(Beam)

    def add_beam(self,Beam):
        self.Members.append(Beam)

    def add_connection(self):
        '''
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
        #TODO
        #note, this process is a "global assembly" of all the beam subsystems
        #1. construct systems for beams
        for beam in self.Members: 
            beam._construct_system()
        #since we have constructed axial models in a global coordinate frame,
        #    we can simply locate match the dofs that are geometrical coincident
        #1. identify beams to be connected
        #2. locate dofs at location on each beam
        #3. connect dofs based on rotation of coordinate frame


        return
    
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