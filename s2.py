import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import maxima, minima
from deriv_funcs_massive import deriv, q, metric

SAVE = 0
PLOT = 1

nt = 100000
# TODO: have more points near periapsis
# S2 is very fast here, and zooming in shows nt=100000 is too low to measure
# precession angle well
# this should be easy, since t is barely different from proper time (zeta)

T = 80000000 # units of 1/2 r_s / c

a = 0.99 # black hole angular momentum

# black hole mass
M = 4.28e6 # solar masses
# distance from earth
R_0 = 8.32 # kpc

# definitions as in Grould et al. 2017 (relation to Earth observer frame)
# orbital elements
sma = 0.1255 # arcseconds in Earth's sky
ecc = 0.8839
incl = 134.18 # degrees
long_asc = 226.94 # degrees
arg_peri = 65.51 # degrees
period = 16.0 # years

# convert to natural units
# degrees -> radians
# sma/arcseconds -> half Schwarzschild radii
# years -> 1/2 r_s / c

incl *= np.pi/180
long_asc *= np.pi/180
arg_peri *= np.pi/180

YEAR = 86400 * 365.25 # seconds
SGP_SUN = 1.32712440018e20 # solar standard gravitational parameter / m^3 s^-2
SOL = 299792458 # speed of light / ms^-1
AU = 149597870700 # astronomical unit / m
half_rs = SGP_SUN * M / (SOL*SOL) # 1/2 schawrzschild radius

sma *= 1000 * R_0 * AU / (half_rs)
# sma *= 1/3600 * np.pi/180 * 1000 *  R_0 * 648000/np.pi * AU / half_rs
period *= YEAR * SOL / half_rs

# cartesian coordinates of apoapsis in orbital plane
ecc_anom = np.pi # 0 at periapsis, pi at apoapsis
x_orb = sma * np.array([np.cos(ecc_anom)-ecc,
                        np.sqrt(1-ecc*ecc)*np.sin(ecc_anom),
                        0])

# looking from earth, S2 orbit is clockwise
# so sign of y here must agree (currently does)
# anti-clockwise in orbital plane
v_orb = 2*np.pi*sma*sma/(np.linalg.norm(x_orb)*period) * \
        np.array([-np.sin(ecc_anom),
                  np.cos(ecc_anom)*np.sqrt(1-ecc*ecc), 0])

# rotation matrices
R_arg = np.array([[np.cos(arg_peri), -np.sin(arg_peri), 0],
                   [np.sin(arg_peri), np.cos(arg_peri), 0],
                   [0, 0, 1]])
R_incl = np.array([[1, 0, 0],
                   [0, np.cos(incl), -np.sin(incl)],
                   [0, np.sin(incl), np.cos(incl)]])
R_long = np.array([[np.cos(long_asc), -np.sin(long_asc), 0],
                   [np.sin(long_asc), np.cos(long_asc), 0],
                   [0, 0, 1]])

# orbital plane to
# cartesian coords in observer frame
# z axis points toward earth
# x axis is north (declination)
# y axis is east (right ascension)
obs_from_orb = R_long @ R_incl @ R_arg
orb_from_obs = obs_from_orb.transpose() # property of orthogonal matrix

# spin orientation in observer frame
# (using same parameters as for orbit)
spin_phi = 0 # anti clockwise from x in x-y plane
spin_theta = np.pi # angle made to z axis

R_spin_phi = np.array([[np.cos(spin_phi), np.sin(spin_phi), 0],
                      [-np.sin(spin_phi), np.cos(spin_phi), 0],
                      [0, 0, 1]])

R_spin_theta = np.array([[-np.cos(spin_theta), 0, np.sin(spin_theta)],
                         [0, 1, 0],
                         [-np.sin(spin_theta), 0, -np.cos(spin_theta)]])

# BH frame has spin in -z direction
bh_from_obs = R_spin_theta @ R_spin_phi
obs_from_bh = bh_from_obs.transpose()

x_bh = bh_from_obs @ obs_from_orb @ x_orb
v_bh = bh_from_obs @ obs_from_orb @ v_orb
#x_bh = obs_from_orb @ x_orb
#v_bh = obs_from_orb @ v_orb


# verified - orbit is close to this
#test_ea = np.linspace(0, 2*np.pi, 50)
#test_ellipse = np.zeros((50,3))
#test_ellipse_obs = np.zeros((50,3))
#for i in range(50):
#        test_ellipse[i,0] = sma * (np.cos(test_ea[i]) - ecc)
#        test_ellipse[i,1] = sma * np.sqrt(1-ecc*ecc) * np.sin(test_ea[i])
#        
#        test_ellipse_obs[i,:] = obs_from_orb @ test_ellipse[i,:]
        

#x_obs = x_orb
#v_obs = v_orb

# find Boyer Lindquist position
x0 = x_bh[0]
y0 = x_bh[1]
z0 = x_bh[2]

phi0 = np.arctan2(y0,x0)

a_xyz = a*a-x0*x0-y0*y0-z0*z0
r0 = np.sqrt(-0.5*a_xyz + 0.5*np.sqrt(a_xyz*a_xyz + 4*a*a*z0*z0))

theta0 = np.arccos(z0/r0)

# verified - change back to get starting point
# x = np.sqrt(r0*r0 + a*a)*np.sin(theta0)*np.cos(phi0)
# y = np.sqrt(r0*r0 + a*a)*np.sin(theta0)*np.sin(phi0)
# z = r0*np.cos(theta0)

t0 = 0

# we need contravariant 4-momentum (= 4-velocity, as m=1)
# which is d/dtau of t,r,theta,phi

# first find d/dt of t,r,theta,phi
_u_dt = np.zeros(4)

# v_obs is d/dt of x,y,z

mat = np.array([
        [r0*x0/(r0*r0 + a*a), -x0*np.tan(theta0 - np.pi/2), -y0],
        [r0*y0/(r0*r0 + a*a), -y0*np.tan(theta0 - np.pi/2), x0],
        [z0/r0, -r0*np.sin(theta0), 0]
        ])

_u_dt[0] = 1
_u_dt[1:4] = np.linalg.solve(mat, v_bh)

# change to proper time derivative
metric0 = metric(np.array([0, r0, theta0, 0, 0, 0]), a) # only depends on r, theta
dt_dtau = 1/np.sqrt(-(metric0 @ _u_dt) @ _u_dt)

_p = dt_dtau * _u_dt

# multiply by metric for covariant 4-momentum
p_cov = metric0 @ _p

E = -p_cov[0] # by definition, for any stationary observer
p_r0 = p_cov[1]
p_theta0 = p_cov[2]
p_phi = p_cov[3]

orbit0 = np.array([t0, r0, theta0, phi0, p_r0, p_theta0])

# these are functions of orbit0:
# angular momentum (= r * p^phi for large r)
b = p_phi

# Carter's constant
_q = q(theta0, p_theta0, a, E, b)

zeta = np.linspace(0, T, nt + 1)
orbit = np.zeros((nt + 1, 6))
orbit_xyz = np.zeros((nt + 1, 3))

orbit = spi.odeint(deriv, orbit0, zeta, (a,E,b,_q), atol = 1e-12)

orbit_xyz[:, 0] = np.sqrt(orbit[:, 1]**2 + a * a) * \
    np.sin(orbit[:, 2]) * np.cos(orbit[:, 3])
orbit_xyz[:, 1] = np.sqrt(orbit[:, 1]**2 + a * a) * \
    np.sin(orbit[:, 2]) * np.sin(orbit[:, 3])
orbit_xyz[:, 2] = orbit[:, 1] * np.cos(orbit[:, 2])

# transform back into obs and orb frames
orbit_obs = np.zeros((nt + 1, 3))
orbit_orb = np.zeros((nt + 1, 3))
for i in range(nt+1):
    orbit_obs[i,:] = obs_from_bh @ orbit_xyz[i,:]
#    orbit_obs[i,:] = orbit_xyz[i,:]
    orbit_orb[i,:] = orb_from_obs @ orbit_obs[i,:]

# find precession angle
t = orbit[:, 0]
pr = orbit[:, 4]

imaxs = maxima(pr)
imins = minima(pr)

phase = np.arctan2(orbit_orb[:, 1], orbit_orb[:, 0])
# different from keplerian fit
simul_period = t[imaxs][1] - t[imaxs][0]
simul_sma = (orbit_orb[imins, 0][0] - orbit_orb[imaxs, 0][0])/2

deltaphase = phase[imaxs][1] + np.pi
print("Precession per Orbit:", deltaphase)
# semi major axis
# theoretical precession angle - Einstein
#thdeltaphase = 24 * np.pi**3 * sma * sma \
#    / (period*period * (1 - ecc*ecc))
# Gillesen 2017
thdeltaphase = 6 * np.pi / (sma * (1 - ecc*ecc))
print("Theoretical Value (no spin, small angle):", thdeltaphase)

if PLOT:
    plt.close('all')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0],[0])
    ax.plot(orbit_obs[:, 0], orbit_obs[:, 1], zs=orbit_obs[:, 2])
#    check orbit is close to elliptical fit
#    ax.plot(test_ellipse_obs[:, 0], test_ellipse_obs[:, 1], zs=test_ellipse_obs[:, 2])
#    check orbit is 'flat' after inverse transform
#    ax.plot(orbit_xyplane[:, 0], orbit_xyplane[:, 1], zs=orbit_xyplane[:, 2])
    
    # plot in orbital plane
    plt.figure(figsize=(8,8))
    plt.plot(orbit_orb[:, 0], orbit_orb[:, 1], 'k', linewidth=0.5)
    plt.scatter([0],[0], c='k', marker='x')
    plt.title("r_0 = {}, L = {}, E = {}".format(r0,b,E))
    
    # view from Earth's sky
    # (west, north)
    plt.figure(figsize=(8,8))
    to_arcsec = half_rs / (1000 * R_0 * AU)
    plt.plot(-orbit_obs[:, 1]*to_arcsec, orbit_obs[:, 0]*to_arcsec, 'k', linewidth=0.5)
    plt.xlabel("-alpha") # - right ascension
    plt.ylabel("delta") # declination
    plt.scatter([0],[0], c='k', marker='x')
    plt.title("r_0 = {}, L = {}, E = {}".format(r0,b,E))

    plt.show()
