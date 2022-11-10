# computing cross sectional properties 

#cell 1:

from dolfinx.fem import assemble,Constant, FunctionSpace, Function
from ufl import Measure, SpatialCoordinate, TestFunction, TrialFunction
from dolfinx.io import XDMFFile
from dolfinx.mesh import Mesh
import meshio

import matplotlib.pyplot as plt

from mpi4py import MPI
mesh = Mesh()
with XDMFFile("box_girder_bridge.xdmf") as in_file:
    in_file.read(mesh)
# fileName = "../mesh/box_girder_bridge.msh"
# mesh = meshio.read(fileName)
# with XDMFFile(MPI.COMM_WORLD, f"../mesh/box_girder_bridge.xdmf", "w") as file:
#     file.write_mesh(msh)

# with XDMFFile(MPI.COMM_WORLD, f"../mesh/box_girder_bridge.xdmf", "r") as xdmf:
#     mesh = xdmf.read_mesh()

htop = max(mesh.coordinates()[:, 1])
hbot = min(mesh.coordinates()[:, 1])
wleft = min(mesh.coordinates()[:, 0])
wright = max(mesh.coordinates()[:, 0])
    
    
dx = Measure("dx", domain=mesh)
x = SpatialCoordinate(mesh)

A = assemble(Constant(1.)*dx)
y_G = assemble(x[0]*dx)/A
z_G = assemble(x[1]*dx)/A

z_0 = z_G
I_y = assemble((x[1]-z_0)**2*dx)

plt.figure()
plt.plot(mesh, linewidth=0.2)
plt.xlabel("$y$", fontsize=16)
plt.ylabel("$z$", fontsize=16)
ax = plt.gca()
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print("Area: {:.2f} m^2".format(A))
print("Centroid axis location: {:.2f} m".format(z_G))
print("Bending inertia: {:.2f} m^4".format(I_y))

# #cell 2

# V = FunctionSpace(mesh, "CG", 1)
# v = TestFunction(V)
# u = TrialFunction(V)
# T = Function(V, name="Temperature")
# T_amb = Constant(20.)
# T_fire = Constant(800.)

# a = dot(grad(u), grad(v))*dx
# L = Constant(0.)*v*dx
# bc = [DirichletBC(V, T_amb, "near(x[1],{}) && on_boundary".format(htop)), 
#       DirichletBC(V, T_fire, "near(x[1],{}) && on_boundary".format(hbot))]
# solve(a == L, T, bc)

# k = exp(-(T-20)/211)

# plt.figure(figsize=(10,4))
# plt.subplot(1, 2, 1)
# pl = plot(T, cmap="coolwarm")
# cbar = plt.colorbar(pl, orientation="horizontal")
# cbar.set_label('"Temperature [°C]')
# plt.subplot(1, 2, 2)
# pl = plot(k, cmap="plasma_r")
# cbar = plt.colorbar(pl, orientation="horizontal")
# cbar.set_label("Stiffness reduction factor [-]")


# #cell 3

# E = Constant(50.)

# print("Axial stiffness:")
# H_N_amb = float(E)*A
# H_N = assemble(k*E*dx)
# print(" H_N_amb = {:.2f} GN".format(H_N_amb))
# print(" H_N = {:.2f} GN".format(H_N))
# print(" relative = {:+.2f}%".format(100*(H_N/H_N_amb-1)))

# print("\nBending stiffness:")
# H_M_amb = float(E)*I_y
# H_M = assemble(k*E*(x[1]-z_0)**2*dx)
# print(" H_M_amb = {:.2f} GN.m^2".format(H_M_amb))
# print(" H_M = {:.2f} GN.m^2".format(H_M))
# print(" relative = {:+.2f}%".format(100*(H_M/H_M_amb-1)))

# print("\nCoupling stiffness:")
# H_NM_amb = float(E)*(z_G-z_0)
# H_NM = assemble(k*E*(x[1]-z_0)*dx)
# print(" H_NM_amb = {:.2f} GN.m".format(H_NM_amb))
# print(" H_NM = {:.2f} GN.m".format(H_NM))

# #cell 4

# nu = Constant(0.2)
# mu_amb = E/2/(1+nu)
# mu = k*mu_amb

# Ve = FiniteElement("CG", mesh.ufl_cell(), 1)
# Re = FiniteElement("R", mesh.ufl_cell(), 0)
# W = FunctionSpace(mesh, MixedElement([Ve, Re]))

# v, lamb_ = TestFunctions(W)
# u, lamb = TrialFunctions(W)

# V_z = Constant(1.)
# Delta_H = H_N*H_M-H_NM**2
# a_shear = dot(mu*grad(u), grad(v))*dx 
# a_tot = a_shear + (u*lamb_+v*lamb)*dx
# f = (H_N*(x[1]-z_0)-H_NM)/Delta_H*k*E*V_z
# L_shear = f*v*dx

# print("Average f:", assemble(f*dx))


# #cell 5


# w = Function(W)
# solve(a_tot == L_shear, w)

# u = w.sub(0, True)
# u.rename("Displacement", "")

# sig_xy = Function(V, name="Shear stress xy")
# sig_xy.assign(project(mu*u.dx(0), V))
# sig_xz = Function(V, name="Shear stress xz")
# sig_xz.assign(project(mu*u.dx(1), V))

# with XDMFFile("results.xdmf") as out_file:
#     out_file.parameters["functions_share_mesh"]=True
#     out_file.write(u, 0)
#     out_file.write(sig_xy, 0)
#     out_file.write(sig_xz, 0)

# #cell 6

# energy = 0.5*assemble(ufl.energy_norm(a_shear, w))
# H_V = assemble(mu*dx)
# kappa = float(V_z)**2/2/H_V/energy
# print("kappa_z = {:.3f}".format(kappa))
# print(" shear stiffness relative reduction: {:+.2f}%".format((kappa*H_V/float(mu_amb)/A-1)*100))


