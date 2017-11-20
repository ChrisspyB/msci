import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv_massive, E_f, rho, Delta, pomega

SAVE = 0
PLOT = 1

n_rays = 50

nt = 1000

a = 0.999  # black hole angular momentum

# [ray_0, ... , ray_n-1]
# [r, theta, phi, p_r, p_theta, p_phi]
rays_0 = np.zeros((n_rays, 6))

# camera positions (r, theta, phi)
cam_pos = np.array([10, 0.5 * np.pi, 0])

# t=0 (negative) ray directions in camera's spherical coordinates (aligned
# to FIDO)
theta_0 = np.ones(n_rays) * math.pi / 2
phi_0 = np.linspace(0.7 * np.pi, 1.3 * np.pi, n_rays)

# unit vectors in tangent space to FIDO, up is along e_theta
n_0 = np.zeros((n_rays, 3))  # (r,theta,phi)
n_0[:, 0] = -np.sin(theta_0) * np.cos(phi_0)  # e_x ~ e_r
n_0[:, 2] = -np.sin(theta_0) * np.sin(phi_0)  # e_y ~ e_phi
n_0[:, 1] = -np.cos(theta_0)  # e_z ~ e_theta

# note the minus signs above; rays are integrated backwards in time
# and we would like to specify theta_0, phi_0 in the direction of
# integration of the rays

# initialise ray momenta and positions
# Note: E_f doesnt depend on momentum
y_0 = np.concatenate((cam_pos, np.zeros(3)))

for i in range(n_rays):
    rays_0[i, 3] = n_0[i, 0] * \
        E_f(y_0, n_0[i, 2], a) * rho(y_0, a) / math.sqrt(Delta(y_0, a))
    rays_0[i, 4] = n_0[i, 1] * E_f(y_0, n_0[i, 2], a) * rho(y_0, a)
    rays_0[i, 5] = n_0[i, 2] * \
        E_f(y_0, n_0[i, 2], a) * pomega(y_0, a)  # = b (conserved)

rays_0[:, 0:3] = cam_pos.copy()

zeta = np.linspace(0, -25, nt + 1)

rays = np.zeros((n_rays, nt + 1, 6))

# integrate momenta and positions
for i in range(n_rays):
    rays[i] = spi.odeint(deriv_massive, rays_0[i], zeta, (a,))

rays_x = np.sqrt(rays[:, :, 0]**2 + a * a) * \
    np.sin(rays[:, :, 1]) * np.cos(rays[:, :, 2])
rays_y = np.sqrt(rays[:, :, 0]**2 + a * a) * \
    np.sin(rays[:, :, 1]) * np.sin(rays[:, :, 2])
rays_z = rays[:, :, 0] * np.cos(rays[:, :, 1])

if SAVE:
    np.save("renderdata", np.dstack((rays_x, rays_y, rays_z)))

if PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-1,1)
    for i in range(n_rays):
        ax.plot(rays_x[i, :], rays_y[i, :], zs=rays_z[i, :])

    plt.show()
