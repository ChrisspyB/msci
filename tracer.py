import math
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SAVE = True
PLOT = False

n_rays = 50
nt = 10000  # time steps (time points - 1)

a = 0.999  # black hole angular momentum
r_1 = 2 + 2*math.cos(2*math.acos(-a)/3)
r_2 = 2 + 2*math.cos(2*math.acos(a)/3)

def b_0(r):
    return -(r*r*r - 3*r*r + a*a*r + a*a)/(a*(r-1))

# write this as a function of b in the range [r_1, r_2]
# we should be able to invert b_0(r) in this range
def q_0(r):
    return -r*r*r*(r*r*r - 6*r*r + 9*r - 4*a*a)/(a*a*(r-1)*(r-1))

def rho(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)


def rho_sqr(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost


def Sigma(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    sint = math.sin(theta)
    rr_plus_aa = r * r + a * a
    return math.sqrt(rr_plus_aa * rr_plus_aa - a * a * sint * sint)


def Delta(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    return r * r - 2 * r + a * a


def omega(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    sig = Sigma(y)
    return 2 * a * r / (sig * sig)


def pomega(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    return Sigma(y) * math.sin(theta) / rho(y)


def alpha(y):
    return rho(y) * math.sqrt(Delta(y)) / Sigma(y)


def E_f(y, n_phi):
    r, theta, phi, p_r, p_theta, p_phi = y
    return 1 / (alpha(y) + omega(y) * pomega(y) * n_phi)


def q(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (p_phi * p_phi / (sint * sint) - a * a)


def P(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    return r * r + a * a - a * p_phi


def R(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    p = P(y)
    p_minus_a = (p_phi - a)
    return p * p - Delta(y) * (p_minus_a * p_minus_a + q(y))


def Theta(y):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q(y) - cost * cost * (p_phi * p_phi / (sint * sint) - a * a)


def deriv(y, zeta):
    r, theta, phi, p_r, p_theta, p_phi = y
    _delta = Delta(y)
    _rho_sqr = rho_sqr(y)
    _rho_qua = _rho_sqr * _rho_sqr
    _P = P(y)
    _R = R(y)
    _Theta = Theta(y)
    _b = p_phi
    _q = q(y)

    cot = -math.tan(theta - math.pi / 2.0)
    aasincos = a * a * math.sin(theta) * math.cos(theta)

    dr = p_r * _delta / _rho_sqr
    dtheta = p_theta * 1.0 / _rho_sqr
    dphi = (a * _P / _delta + _b - a + _b * cot * cot) / _rho_sqr
    dp_r = p_r * p_r * (_delta * r / _rho_qua - (r - 1) / _rho_sqr) + \
        p_theta * p_theta * r / _rho_qua + \
        ((2 * r * _P - (r - 1) * ((_b - a) * (_b - a) + _q)) * _delta * _rho_sqr -
         _R * ((r - 1) * _rho_sqr + _delta * r)) / (_delta * _delta * _rho_qua) - \
        r * _Theta / _rho_qua
    dp_theta = -aasincos * _delta * p_r * p_r / _rho_qua - \
        aasincos * p_theta * p_theta / _rho_qua + \
        (_R * aasincos - _delta * _rho_sqr * (aasincos - _b * _b * (cot + cot * cot * cot))) \
         / (_delta * _rho_qua) + \
        _Theta * aasincos / _rho_qua
    dp_phi = 0

    return np.array([dr, dtheta, dphi, dp_r, dp_theta, dp_phi])

# [ray_0, ... , ray_n-1]
# [r, theta, phi, p_r, p_theta, p_phi]
rays_0 = np.zeros((n_rays, 6))

# camera position (r, theta, phi)
cam_pos = np.array([10, 0.5 * np.pi, 0])

#  t=0 (negative) ray directions in camera's spherical coordinates (aligned to FIDO)
theta_0 = np.linspace(np.pi/2, np.pi/2, n_rays)
phi_0 = np.linspace(0.7 * np.pi, 1.3 * np.pi, n_rays)

# unit vectors in tangent space to FIDO, up is along e_theta
n_0 = np.zeros((n_rays, 3))  # (r,theta,phi)
n_0[:, 0] = -np.sin(theta_0) * np.cos(phi_0) # e_x ~ e_r
n_0[:, 2] = -np.sin(theta_0) * np.sin(phi_0) # e_y ~ e_phi
n_0[:, 1] = -np.cos(theta_0) # e_z ~ e_theta

# note the minus signs above; rays are integrated backwards in time
# and we would like to specify theta_0, phi_0 in the direction of integration of the rays

# initialise ray momenta and positions
# Note: E_f doesnt depend on momentum
y_0 = np.concatenate((cam_pos, np.zeros(3))) 

for i in range(n_rays):
    rays_0[i, 3] = n_0[i, 0] * E_f(y_0, n_0[i, 2]) * rho(y_0) / math.sqrt(Delta(y_0))
    rays_0[i, 4] = n_0[i, 1] * E_f(y_0, n_0[i, 2]) * rho(y_0)
    rays_0[i, 5] = n_0[i, 2] * E_f(y_0, n_0[i, 2]) * pomega(y_0)  # = b (conserved)

rays_0[:, 0:3] = cam_pos.copy()

zeta = np.linspace(0, -25, nt + 1)

rays = np.zeros((n_rays, nt + 1, 6))

# integrate momenta and positions
for i in range(n_rays):
    rays[i] = spi.odeint(deriv, rays_0[i], zeta)

rays_x = np.sqrt(rays[:,:, 0]**2 + a * a) * \
    np.sin(rays[:,:, 1]) * np.cos(rays[:,:, 2])
rays_y = np.sqrt(rays[:,:, 0]**2 + a * a) * \
    np.sin(rays[:,:, 1]) * np.sin(rays[:,:, 2])
rays_z = rays[:,:, 0] * np.cos(rays[:,:, 1])

if SAVE:
    np.save("renderdata", np.dstack((rays_x, rays_y, rays_z)))

if PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(n_rays):
        ax.plot(rays_x[i,:], rays_y[i,:], zs=rays_z[i,:])

    plt.show()
