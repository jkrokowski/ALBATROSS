'''
The beam_model module
---------------------
Contains the most important classes for beam formulations using either
Euler-Bernoulli Beam Theory or shear-deformable Timoshenko Beam Theory

Euler-Bernoulli Beam models consist of a 4x4 stiffness matrix (K) for static cases.

Dynamic analysis is to be completed, a 4x4damping matrix(C) 
and 4x4mass (M) matrix
'''

from dolfinx.fem import TensorFunctionSpace,VectorFunctionSpace,Expression,Function,Constant, locate_dofs_geometrical,locate_dofs_topological,dirichletbc,form
# from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import (LinearProblem,assemble_matrix,assemble_vector, 
                                apply_lifting,set_bc,create_vector)
from ufl import (Jacobian, TestFunction,TrialFunction,diag,as_vector, sqrt, 
                inner,dot,grad,split,cross,Measure)
from FROOT_BAT.elements import *
from FROOT_BAT.cross_section import CrossSection
from FROOT_BAT.axial import Axial
from petsc4py.PETSc import ScalarType
import numpy as np

from petsc4py import PETSc

class BeamModel(Axial):
    '''
    Class that combines both 1D and 2D analysis
    '''

    def __init__(self,axial_mesh,xc_info):
        '''
        axial_mesh: mesh used for 1D analysis (often this is a finer
            discretization than the 1D mesh used for locating xcs)

        info2D = (xcs,mats,axial_pos_mesh,xc_orientations)
            xcs : list of 2D xdmf meshes for each cross-section
            mats : list of materials used corresponding to each XC
            axial_pos_mesh : the beam axis discretized into the number of
                elements with nodes at the spanwise locations of the XCs
            xc_orientations: list of vectors defining orientation of 
                horizontal axis used in xc analysis
        '''
        self.axial_mesh = axial_mesh
        [self.xcs, self.mats, self.axial_pos_mesh, self.orientations] = xc_info
        self.numxc = len(self.xcs)
        
        print("Orienting XCs along beam axis....")
        self.get_xc_orientations_for_1D()

        print("Getting XC Properties...")
        self.get_axial_props_from_xc()

        print("Initializing Axial Model (1D Analysis)")
        super().__init__(self.axial_mesh,self.k,self.o)

        print("Computing Elastic Energy...")
        self.elastic_energy()

    def get_xc_orientations_for_1D(self):
        #define orientation of the x2 axis w.r.t. the beam axis x1 to allow for 
        # matching the orientation of the xc with that of the axial mesh
        # (this must be done carefully and with respect to the location of the beam axis)
        self.O2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=3)
        self.o2 = Function(self.O2)
        self.o2.vector.array = np.array(self.orientations)
        self.o2.vector.destroy() #needed for PETSc garbage collection

        #interpolate these orientations into the finer 1D analysis mesh
        self.O = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=3)
        self.o = Function(self.O)
        self.o.interpolate(self.o2)

    
    def get_axial_props_from_xc(self):
        '''
        CORE FUNCTION FOR PROCESSING MULTIPLE 2D XCs TO PREPARE A 1D MODEL
        '''
        # (mesh1D_2D,(meshes2D,mats2D)) = info2D 
        # mesh1D_1D = info1D
        xcs = []
        # K_list = []
        
        def get_flat_sym_stiff(K_mat):
            K_flat = np.concatenate([K_mat[i,i:] for i in range(6)])
            return K_flat
        
        sym_cond = False #there is an issue with symmetric tensor fxn spaces in dolfinx at the moment
        K2 = TensorFunctionSpace(self.axial_pos_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        k2 = Function(K2)
        #TODO:same process for mass matrix
        A2 = FunctionSpace(self.axial_pos_mesh,('CG',1))
        a2 = Function(A2)
        C2 = VectorFunctionSpace(self.axial_pos_mesh,('CG',1),dim=2)
        c2 = Function(C2)
        
        for i,[mesh2d,mat2D] in enumerate(zip(self.xcs,self.mats)):
            print('    computing properties for XC '+str(i+1)+'/'+str(self.numxc)+'...')
            #instantiate class for cross-section i
            xcs.append(CrossSection(mesh2d,mat2D))
            #analyze cross section
            xcs[i].getXCStiffnessMatrix()

            #output stiffess matrix
            if sym_cond==True:
                #need to add fxn
                print("symmetric mode not available yet")
                exit()
                k2.vector.array[21*i,21*(i+1)] = xcs[i].K.flatten()
            elif sym_cond==False:
                k2.vector.array[36*i:36*(i+1)] = xcs[i].K.flatten()
                a2.vector.array[i] = xcs[i].A
                c2.vector.array[2*i:2*(i+1)] = [xcs[i].yavg,xcs[i].zavg]
        print("Done computing cross-sectional properties...")
        # if sym_cond==True:
        #     K_entries = np.concatenate([get_flat_sym_stiff(K_list[i]) for i in range(self.num_xc)])
        # elif sym_cond == False:
        #     K_entries = np.concatenate([K_list[i].flatten() for i in range(self.num_xc)])
        
        # k2.vector.array = K_entries
        print("Interpolating cross-sectional properties to axial mesh...")
        #interpolate from axial_pos_mesh to axial_mesh 
        self.K = TensorFunctionSpace(self.axial_mesh,('CG',1),shape=(6,6),symmetry=sym_cond)
        self.k = Function(self.K)
        self.k.interpolate(k2)

        self.A = FunctionSpace(self.axial_mesh,('CG',1))
        self.a = Function(self.A)
        self.a.interpolate(a2)

        self.V = VectorFunctionSpace(self.axial_mesh,('CG',1),dim=2)
        self.c = Function(self.V)
        self.c.interpolate(c2)
        # see: https://fenicsproject.discourse.group/t/yaksa-warning-related-to-the-vectorfunctionspace/11111
        k2.vector.destroy()     #need to add to prevent PETSc memory leak 
        a2.vector.destroy()
        c2.vector.destroy()
        print("Done interpolating cross-sectional properties to axial mesh...")



