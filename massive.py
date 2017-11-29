import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv_massive, q

SAVE = 0
PLOT = 1

n_orbits = 12

nt = 1000

T = 25000 # units of affine parameter?

a = 0.0000001  # black hole angular momentum

# [ray_0, ... , ray_n-1]
# [r, theta, phi, p_r, p_theta, p_phi]
orbits_0 = np.zeros((n_orbits, 4))

start_pos = np.array([100, 0.5 * np.pi, 0])
# note the minus signs above; rays are integrated backwards in time
# and we would like to specify theta_0, phi_0 in the direction of
# integration of the rays

y_0 = np.concatenate((np.zeros(1), start_pos))
orbits_0[:] = y_0

zeta = np.linspace(0, T, nt + 1)

orbits = np.zeros((n_orbits, nt + 1, 4))

# integrate momenta and positions
for i in range(n_orbits):
    mu = 1
    p_theta = 0.0000001
    
    # here units of length are half the schwarzschild radius (also note c = 1)
    b = 0.4 # p_phi
    E = 1 # energy at infinity
    
    _q = q(y_0, p_theta, a, mu*mu, E, b)
    
    orbits[i] = spi.odeint(deriv_massive, orbits_0[i], zeta, (a,mu,E,b,_q))

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
