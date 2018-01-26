import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_light import deriv, E_f, rho, Delta, pomega

SAVE = False
PLOT = True

n_rays = 2*250
nt = 250  # time steps (time points - 1)
# note: integration is adaptive, nt is just the number of points that get saved

a = 0.9999  # black hole angular momentum / M

x_dist = 1000000


# [ray_0, ... , ray_n-1]
# [r, theta, phi, p_r, p_theta, p_phi]
rays_0 = np.zeros((n_rays, 6))

# multiple camera positions (r, theta, phi)
cam_pos = np.zeros((n_rays, 3))
cam_y = np.concatenate(
        (np.linspace(-200, -7, n_rays/2),
         np.linspace(7, 200, n_rays/2)))  # impact parameters
cam_x = np.ones(n_rays) * x_dist
cam_pos[:, 0] = np.sqrt(cam_x * cam_x + cam_y * cam_y)
cam_pos[:, 1] = math.pi / 2
cam_pos[:, 2] = np.arctan2(cam_y, cam_x)

# initial (negative) ray directions in camera's cartesian coords
# unit vectors in tangent space to FIDO
n_0 = np.zeros((n_rays, 3))  # (e_r, e_theta, e_phi)
# e_x
n_0[:, 0] = np.cos(cam_pos[:, 2])
n_0[:, 1] = 0
n_0[:, 2] = -np.sin(cam_pos[:, 2])

# note the minus signs above; rays are integrated backwards in time
# and we would like to specify theta_0, phi_0 in the direction of
# integration of the rays

# initialise ray momenta and positions
# Note: E_f doesnt depend on momentum
for i in range(n_rays):
    y_0 = np.concatenate((cam_pos[i,:], np.zeros(3)))
    rays_0[i, 3] = n_0[i, 0] * \
        E_f(y_0, n_0[i, 2], a) * rho(y_0, a) / math.sqrt(Delta(y_0, a))
    rays_0[i, 4] = n_0[i, 1] * E_f(y_0, n_0[i, 2], a) * rho(y_0, a)
    rays_0[i, 5] = n_0[i, 2] * \
        E_f(y_0, n_0[i, 2], a) * pomega(y_0, a)  # = b (conserved)

rays_0[:, 0:3] = cam_pos.copy()

zeta = np.linspace(0, -2*x_dist, nt + 1)

rays = np.zeros((n_rays, nt + 1, 6))

deflec = np.zeros(n_rays)
# integrate momenta and positions
for i in range(n_rays):
    rays[i] = spi.odeint(deriv, rays_0[i], zeta, (a,))

rays_x = np.sqrt(rays[:,:, 0]**2 + a * a) * \
    np.sin(rays[:,:, 1]) * np.cos(rays[:,:, 2])
rays_y = np.sqrt(rays[:,:, 0]**2 + a * a) * \
    np.sin(rays[:,:, 1]) * np.sin(rays[:,:, 2])
rays_z = rays[:,:, 0] * np.cos(rays[:,:, 1])

dx = rays_x[:, -1] - rays_x[:, -2]
dy = rays_y[:, -1] - rays_y[:, -2]
deflec = np.arctan2(-dy, -dx)

if SAVE:
    np.save("renderdata", np.dstack((rays_x, rays_y, rays_z)))

if PLOT:
    plt.figure()
    for i in range(n_rays):
        plt.plot(rays_x[i,:], rays_y[i,:], 'b')
    plt.title("Ray Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.figure()
    plt.plot(cam_y, np.abs(deflec), label="Traced Rays", linewidth = 1.0)
    b = np.abs(cam_y)
    s = np.sign(-cam_y) # + = prograde, - = retrograde
    # kerr deflection in equatorial plane
    plt.plot(cam_y,
             4/b \
             + (15*math.pi/4 - 12*s*a) * 1/(b*b) \
             + (128/3 - 10*math.pi*s*a + 4*a*a) * 1/(b*b*b)
             ,'--', label="Theory (Cubic Order)", linewidth = 1.0)
    #plt.title("Deflection of rays with different starting Y")
    plt.xlabel("Impact Parameter")
    plt.ylabel("Deflection Angle")
    plt.legend()
    plt.grid()
    plt.show()
