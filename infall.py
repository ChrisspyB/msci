import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv, q, energy

# ----
# Roughly reproduces FIG. 2 of Levin 2008
# ----

SAVE = 0
PLOT = 1

nt = 10000

T = 1200
a = 0.0  # black hole angular momentum

# t, r, theta, phi, p_r, p_theta = y

# units of length are half the schwarzschild radius (also note c = 1)
t0 = 0
r0 = 100
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
#E =  0.973101
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

# theoretical time to fall from r0 to r=2 (event horizon)
antideriv = lambda r: 4/3*(r/2)**1.5
infalltau = antideriv(r0) - antideriv(2)
print("Theoretical Infall Time:", infalltau)

# theoretical proper time on radially infalling path
tau = -1/3 * math.sqrt(2) * (r**1.5 - r0**1.5)

if SAVE:
    np.save("renderdata", np.dstack((orbit_x, orbit_y, orbit_z)))
if PLOT:
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_zlim(-5,5)
#    ax.plot(orbit_x, orbit_y, zs=orbit_z)
    
    plt.figure()
    plt.title("Infall Time | r_0 = {}".format(r0))
    plt.plot(zeta, r, 'k', linewidth=0.5, label="simulated")
    plt.plot(tau, r, 'k--', linewidth=0.5, label="theoretical")
    plt.ylabel("r")
    plt.xlabel("tau")
    plt.legend()

    plt.show()
