import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv, q, energy

SAVE = 0
PLOT = 1

nt = 10000

T = 40
a = 0.0  # black hole angular momentum

# t, r, theta, phi, p_r, p_theta = y

# units of length are half the schwarzschild radius (also note c = 1)
t0 = 0
r0 = 10
theta0 = np.pi/2
phi0 = 0

# these are componenets of the covariant 4-momentum (= 4-velocity, as mu = 1)
p_r0 = 0
p_theta0 = 0
p_phi = 0

y_0 = np.array([t0, r0, theta0, phi0, p_r0, p_theta0])

# these are functions of _0:
# angular momentum (= r * p^phi)
b = p_phi
# energy at infinity
E = energy(y_0, a, b)
# Carter's constant
_q = q(theta0, p_theta0, a, E, b)

zeta = np.linspace(0, T, nt + 1)
orbit = np.zeros((nt + 1, 6))

orbit = spi.odeint(deriv, y_0, zeta, (a,E,b,_q), atol = 1e-10)

t = orbit[:, 0]
r = orbit[:, 1]
theta = orbit[:, 2]
phi = orbit[:, 3]
pr = orbit[:, 4]
ptheta = orbit[:, 5]

orbit_x = np.sqrt(r**2 + a * a) * \
    np.sin(theta) * np.cos(phi)
orbit_y = np.sqrt(r**2 + a * a) * \
    np.sin(theta) * np.sin(phi)
orbit_z = r * np.cos(theta)

# theoretical proper time time to fall from r0 to r=2 (event horizon)
# http://www.reed.edu/physics/courses/Physics411/html/411/page2/files/Lecture.31.pdf
# antideriv = lambda r: 1/3 * math.sqrt(2) * (r**1.5)
# infalltau = antideriv(r0) - antideriv(2)

# theoretical proper time on radially infalling path
# http://www.reed.edu/physics/courses/Physics411/html/411/page2/files/Lecture.31.pdf
#tau = 1/3 * math.sqrt(2) * (r0**1.5 - r**1.5)
#_t = -E/3 * math.sqrt(2) * (np.sqrt(r)*(6+r) - math.sqrt(r0)*(6+r0)) +\
#    -2*E*(np.log( (1-np.sqrt(2/r))/(1+np.sqrt(2/r)) ) - \
#         math.log( (1-np.sqrt(2/r0))/(1+np.sqrt(2/r0)) ))


# which disagrees with https://digitalcommons.usu.edu/cgi/viewcontent.cgi?article=1030&context=phys_capstoneproject
infalltau = r0*(math.sqrt(r0/2)*(np.pi/2 - math.asin(math.sqrt(2/r0))) + math.sqrt(1-2/r0))
print("Theoretical Infall Time:", infalltau)

# I integrated something from the paper in mathematica and got this
tau = np.imag(2*1j*np.sqrt(r0*r*(r0-r)) + (r0**1.5)*np.log(-1 + 2/r0 * (r+1j*np.sqrt(r*(r0-r)))))/(2*math.sqrt(2))
# but it does work perfectly

if SAVE:
    np.save("renderdata", np.dstack((orbit_x, orbit_y, orbit_z)))
if PLOT:
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_zlim(-5,5)
#    ax.plot(orbit_x, orbit_y, zs=orbit_z)
    
    plt.figure()
    plt.title("Proper Infall Time | r_0 = {}".format(r0))
    plt.plot(zeta, r, 'k', linewidth=0.5, label="simulated")
    plt.plot(tau, r, 'k--', linewidth=0.5, label="theoretical")
    plt.ylabel("r")
    plt.xlabel("tau")
    plt.legend()
    
#    plt.figure()
#    plt.title("Coordinate Infall Time | r_0 = {}".format(r0))
#    plt.plot(t, r, 'k', linewidth=0.5, label="simulated")
#    plt.plot(_t, r, 'k--', linewidth=0.5, label="theoretical")
#    plt.ylabel("r")
#    plt.xlabel("t")
#    plt.legend()

    plt.show()
