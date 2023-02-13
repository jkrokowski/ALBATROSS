from dolfin import *
import ufl
import matplotlib.pyplot as plt

mesh = Mesh()
with XDMFFile("output/box_XC.xdmf") as in_file:
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



V = FunctionSpace(mesh, "CG", 1)
v = TestFunction(V)
u = TrialFunction(V)
T = Function(V, name="Temperature")
T_amb = Constant(20.)
T_fire = Constant(800.)


# ## Computing shear correction coefficients

E = Constant(50.)
H_N = assemble(E*dx)
H_M = assemble(E*(x[1]-z_0)**2*dx)
H_NM = assemble(E*(x[1]-z_0)*dx)

nu = Constant(0.2)
mu = E/2/(1+nu)

Ve = FiniteElement("CG", mesh.ufl_cell(), 1)
Re = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]))

v, lamb_ = TestFunctions(W)
u, lamb = TrialFunctions(W)

V_z = Constant(1.)
Delta_H = H_N*H_M-H_NM**2
a_shear = dot(mu*grad(u), grad(v))*dx 
a_tot = a_shear + (u*lamb_+v*lamb)*dx
f = (H_N*(x[1]-z_0)-H_NM)/Delta_H*E*V_z
L_shear = f*v*dx

print("Average f:", assemble(f*dx))

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
print(" shear stiffness relative reduction: {:+.2f}%".format((kappa*H_V/float(mu)/A-1)*100))

