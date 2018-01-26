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

T = 800000

a = 0.0 # black hole angular momentum

# t, r, theta, phi, p_r, p_theta = y

# units of length are half the schwarzschild radius (also note c = 1)
t0 = 0
r0 = 27.75
theta0 = np.pi/2
phi0 = 0

# these are componenets of the covariant 4-momentum (= 4-velocity, as mu = 1)
p_r0 = 0
p_theta0 = 0.0
p_phi = 3.980393

y_0 = np.array([t0, r0, theta0, phi0, p_r0, p_theta0])

# these are functions of y_0:
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

orbit_x = np.sqrt(orbit[:, 1]**2 + a * a) * \
    np.sin(orbit[:, 2]) * np.cos(orbit[:, 3])
orbit_y = np.sqrt(orbit[:, 1]**2 + a * a) * \
    np.sin(orbit[:, 2]) * np.sin(orbit[:, 3])
orbit_z = orbit[:, 1] * np.cos(orbit[:, 2])

if SAVE:
    np.save("renderdata", np.dstack((orbit_x, orbit_y, orbit_z)))
if PLOT:
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.set_zlim(-5,5)
#    ax.plot(orbit_x, orbit_y, zs=orbit_z)
    
    plt.figure(figsize=(8,8))
    plt.plot(orbit_x, orbit_y, 'k', linewidth=0.5)
    plt.title("r_0 = {}, L = {}, E = {}".format(r0,b,E))

    plt.show()
