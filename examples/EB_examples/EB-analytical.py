import numpy as np
from matplotlib import pyplot as plt

E = 70e3
L = 1.0
b = 0.1
h = 0.05
rho = 2.7e-3
g = 9.81

EI = (E*b*h**3)/12
A = h*b

q = -rho*A*g

x = np.linspace(0,1,10000)

def w(x,support='ss'):
    if support == 'p-p':
        return 1/EI * ( (q*x**4)/24 -(q*L*x**3)/12 +(q*(L**3)*x)/24 )
    elif support == 'f-f':
        return 1/EI * ( (q*x**4)/24 -(q*L*x**3)/12 +(q*(L**2)*x**2)/24 )
    elif support == 'f-p':
        #TODO: update for correct equation
        return 1/EI * ( (q*x**4)/24 -(5*q*L*x**3)/48 +(q*(L**2)*x**2)/16)
    elif support == 'cantilever':
        return 1/EI * ( (q*x**4)/24 -(q*L*x**3)/6 +(q*(L**2)*x**2)/4 )
    else:
        print("ERROR: improper boundary conditions")
        exit()
        return

figure, axis = plt.subplots(2, 2)

#fixed-fixed
w_ff = w(x,'f-f')
print("Maximum magnitude displacement (fixed-fixed) is: %.2e" % np.min(w_ff))
axis[0, 0].plot(x, w_ff)
axis[0, 0].set_title("Fixed-Fixed")
axis[0, 0].set(xlim=(0, 1), ylim=(-2.5e-4, 0))
axis[0, 0].text(.2,-2e-4,'max disp: %.2e' % np.min(w_ff))

#pinned-pinned
w_pp = w(x,'p-p')
print("Maximum magnitude displacement (pinned-pinned) is: %.2e" % np.min(w_pp))
axis[0, 1].plot(x, w_pp)
axis[0, 1].set_title("Pinned-Pinned")
axis[0, 1].set(xlim=(0, 1), ylim=(-2.5e-4, 0))
axis[0, 1].text(.2,-2e-4,'max disp: %.2e' % np.min(w_pp))

#fixed-pinned
w_fp = w(x,'f-p')
print("Maximum magnitude displacement (fixed-pinned) is: %.2e" % np.min(w_fp))
axis[1, 0].plot(x, w_fp)
axis[1, 0].set_title("Fixed-Pinned")
axis[1, 0].set(xlim=(0, 1), ylim=(-2.5e-4, 0))
axis[1, 0].text(.2,-2e-4,'max disp: %.2e' % np.min(w_fp))

#cantilever
w_cl = w(x,'cantilever')
print("Maximum magnitude displacement (cantilever) is: %.2e" % np.min(w_cl))
axis[1, 1].plot(x, w_cl)
axis[1, 1].set_title("Cantilever")
axis[1, 1].set(xlim=(0, 1), ylim=(-2.5e-4, 0))
axis[1, 1].text(.2,-2e-4,'max disp: %.2e' % np.min(w_cl))

plt.show()