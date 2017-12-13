import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv, q, energy

SAVE = 0
PLOT = 1

nt = 10000

T = 1000000

a = 0.0  # black hole angular momentum

# t, r, theta, phi, p_r, p_theta = y

# units of length are half the schwarzschild radius (also note c = 1)
t0 = 0
r0 = 1000
theta0 = np.pi/2
phi0 = 0

# these are componenets of the covariant 4-momentum (= 4-velocity, as mu = 1)
p_r0 = 0
p_theta0 = 0.0
p_phi = 16

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

def maxima(deriv):
    inds = []
    for i in range(len(deriv) - 1):
        # turning point is where deriv crosses zero
        if deriv[i+1] < 0 and deriv[i] >= 0:
            di = abs(deriv[i])
            di1 = abs(deriv[i+1])
            # find closest point to zero
            if di > di1:
                inds.append(i+1)
            else:
                inds.append(i)
    return inds

def minima(deriv):
    inds = []
    for i in range(len(deriv) - 1):
        # turning point is where deriv crosses zero
        if deriv[i+1] > 0 and deriv[i] <= 0:
            di = abs(deriv[i])
            di1 = abs(deriv[i+1])
            # find closest point to zero
            if di > di1:
                inds.append(i+1)
            else:
                inds.append(i)
    return inds

imaxs = maxima(pr)
imins = minima(pr)

deltaphi = (phi[imaxs][1] - phi[imaxs][0] - 2*np.pi*np.ones(len(t)))[0]
print("Precession per Orbit:", deltaphi)
# semi major axis
a = (r[imaxs][0] + r[imins][0])/2
period = t[imaxs][1] - t[imaxs][0]
# ellipticity
e = (r[imaxs][0] - r[imins][0])/(r[imaxs][0] + r[imins][0])
# theoretical precession angle - Einstein
thdeltaphi = 24 * np.pi**3 * a * a / (period*period * (1 - e*e))
print("Theoretical Value:", thdeltaphi)

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