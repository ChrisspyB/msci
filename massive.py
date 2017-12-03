import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from newderiv_funcs_massive import deriv_massive, q

SAVE = 0
PLOT = 1

n_orbits = 1

nt = 100

T = 25 # units of affine parameter?

a = 0.99999999  # black hole angular momentum

# r, theta, phi, p_r, p_theta, t = y

orbits_0 = np.zeros((n_orbits, 6))
r0, theta0, phi0, t0 = 4, 0.1 * np.pi, 0.5 * np.pi, 0
p_r0, p_theta0 = 0.2,0.2
y_0 = np.array([r0,theta0,phi0,p_r0,p_theta0,t0])
orbits_0[:] = y_0

zeta = np.linspace(0, T, nt + 1)
orbits = np.zeros((n_orbits, nt + 1, 6))

# integrate momenta and positions
for i in range(n_orbits):
    
    # here units of length are half the schwarzschild radius (also note c = 1)
    b = 0.2 # p_phi
    E = 1 # energy at infinity
    
    _q = q(y_0, p_theta0, a, E, b)
    
    # TODO: CHECK b,p_theta,p_r,E are physically allowed
    orbits[i] = spi.odeint(deriv_massive, orbits_0[i], zeta, (a,E,b,_q))

orbits_x = np.sqrt(orbits[:, :, 1]**2 + a * a) * \
    np.sin(orbits[:, :, 2]) * np.cos(orbits[:, :, 3])
orbits_y = np.sqrt(orbits[:, :, 1]**2 + a * a) * \
    np.sin(orbits[:, :, 2]) * np.sin(orbits[:, :, 3])
orbits_z = orbits[:, :, 1] * np.cos(orbits[:, :, 2])

if SAVE:
    np.save("renderdata", np.dstack((orbits_x, orbits_y, orbits_z)))
if PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1,1)
    for i in range(n_orbits):
        ax.plot(orbits_x[i, :], orbits_y[i, :], zs=orbits_z[i, :])

    plt.show()
