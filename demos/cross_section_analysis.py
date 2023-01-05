#!/usr/bin/env python
# coding: utf-8

# # Axial, bending and shear stiffnesses of generic beam cross-sections
# ### FEniCS implementation
# 
# We illustrate the computation of these various moduli for a typical box girder bridge cross-section subject to fire conditions. We consider here only the concrete cross-section and neglect the presence of steel rebars and pres-stress cables in the computation. The cross-section geometry is defined and meshed using `gmsh`. We readily compute the cross-section area and the centroid position.

from dolfin import *
import ufl
import matplotlib.pyplot as plt
import gmsh

mesh = Mesh()
with XDMFFile("../mesh/box_girder_bridge.xdmf") as in_file:
    in_file.read(mesh)
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
plot(mesh, linewidth=0.2)
plt.xlabel("$y$", fontsize=16)
plt.ylabel("$z$", fontsize=16)
ax = plt.gca()
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.show()

print("Area: {:.2f} m^2".format(A))
print("Centroid axis location: {:.2f} m".format(z_G))
print("Bending inertia: {:.2f} m^4".format(I_y))


# Considering that the cross-section is made of a concrete material with $E=50$ GPa at ambient temperature, we assume that the bottom surface is subject to an imposed temperature $T_{fire}$ while the top surface remains at the ambient temperature $T_{amb} = 20\deg$. The remaining surfaces are assumed to have zero flux condition. We solve here a stationary heat transfer problem in order to determine the temperature field on the cross-section. This temperature field will decrease the value of the Young and shear moduli through a common stiffness reduction factor, the expression of which is assumed to be:
# \begin{equation}
# k(T[\deg]) \approx 
# \exp\left(-\frac{T-20}{211}\right), \quad \text{for } T \geq 20^\circ\text{C}
# \end{equation}
# that is:
# $$E(y,z)=k(T(y,z))E_{amb}, \quad \mu(y,z)=k(T(y,z))\mu_{amb}$$ 
# where $E_{amb}$, $\mu_{amb}$ are the stiffness moduli at ambiant temperature.


V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
T = Function(V, name="Temperature")
T_amb = Constant(20.)
T_fire = Constant(800.)

a = dot(grad(u), grad(v))*dx
L = Constant(0.)*v*dx
bc = [DirichletBC(V, T_amb, "near(x[1],{}) && on_boundary".format(htop)), 
      DirichletBC(V, T_fire, "near(x[1],{}) && on_boundary".format(hbot))]
solve(a == L, T, bc)

k = exp(-(T-20)/211)

plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
pl = plot(T, cmap="coolwarm")
cbar = plt.colorbar(pl, orientation="horizontal")
cbar.set_label('"Temperature [°C]')
plt.subplot(1, 2, 2)
pl = plot(k, cmap="plasma_r")
cbar = plt.colorbar(pl, orientation="horizontal")
cbar.set_label("Stiffness reduction factor [-]")


# We see that the bottom of the cross-section is strongly affected by the stiffness reduction. We now compute the various moduli at ambient temperature and the corresponding values for this temperature field.

E = Constant(50.)

print("Axial stiffness:")
H_N_amb = float(E)*A
H_N = assemble(k*E*dx)
print(" H_N_amb = {:.2f} GN".format(H_N_amb))
print(" H_N = {:.2f} GN".format(H_N))
print(" relative = {:+.2f}%".format(100*(H_N/H_N_amb-1)))

print("\nBending stiffness:")
H_M_amb = float(E)*I_y
H_M = assemble(k*E*(x[1]-z_0)**2*dx)
print(" H_M_amb = {:.2f} GN.m^2".format(H_M_amb))
print(" H_M = {:.2f} GN.m^2".format(H_M))
print(" relative = {:+.2f}%".format(100*(H_M/H_M_amb-1)))

print("\nCoupling stiffness:")
H_NM_amb = float(E)*(z_G-z_0)
H_NM = assemble(k*E*(x[1]-z_0)*dx)
print(" H_NM_amb = {:.2f} GN.m".format(H_NM_amb))
print(" H_NM = {:.2f} GN.m".format(H_NM))
#print(" relative = {:+.2f}%".format(100*(H_NM/H_NM_amb-1)))


# ## Computing shear correction coefficients
# ### FEniCS implementation
# We now define and solve the corresponding variational problem for the reduced shear modulus $\mu(y,z)=k(y,z)\mu_{amb}$. We use a mixed function space involving a scalar Lagrange multiplier to enforce a zero-average condition on $u$ as [discussed here](https://fenicsproject.org/olddocs/dolfin/latest/python/demos/neumann-poisson/demo_neumann-poisson.py.html). We also perform a sanity check that the source term is indeed of zero average.

nu = Constant(0.2)
mu_amb = E/2/(1+nu)
mu = k*mu_amb

Ve = FiniteElement("CG", mesh.ufl_cell(), 1)
Re = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]))

v, lamb_ = TestFunctions(W)
u, lamb = TrialFunctions(W)

V_z = Constant(1.)
Delta_H = H_N*H_M-H_NM**2
a_shear = dot(mu*grad(u), grad(v))*dx 
a_tot = a_shear + (u*lamb_+v*lamb)*dx
f = (H_N*(x[1]-z_0)-H_NM)/Delta_H*k*E*V_z
L_shear = f*v*dx

print("Average f:", assemble(f*dx))


# We now solve the problem and extract the displacement function. The stress components are then obtained and fields are saved to a `.xdmf` file.

w = Function(W)
solve(a_tot == L_shear, w)

u = w.sub(0, True)
u.rename("Displacement", "")

sig_xy = Function(V, name="Shear stress xy")
sig_xy.assign(project(mu*u.dx(0), V))
sig_xz = Function(V, name="Shear stress xz")
sig_xz.assign(project(mu*u.dx(1), V))

with XDMFFile("results.xdmf") as out_file:
    out_file.parameters["functions_share_mesh"]=True
    out_file.write(u, 0)
    out_file.write(sig_xy, 0)
    out_file.write(sig_xz, 0)


# Finally, the shear reduction coefficient and the corresponding shear stiffness are computed.

energy = 0.5*assemble(ufl.energy_norm(a_shear, w))
H_V = assemble(mu*dx)
kappa = float(V_z)**2/2/H_V/energy
print("kappa_z = {:.3f}".format(kappa))
print(" shear stiffness relative reduction: {:+.2f}%".format((kappa*H_V/float(mu_amb)/A-1)*100))

