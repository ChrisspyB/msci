import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv, q, metric, inv_metric
from deriv_funcs_light import E_f, rho, Delta, pomega, ray0_b_from_pos0_n0
from deriv_funcs_light import deriv as deriv_l
from deriv_funcs_light import metric as metric_l
from deriv_funcs_light import inv_metric as inv_metric_l
from utils import maxima, minima, xyz_to_bl, bl_to_xyz, \
                                  deriv_rtp_to_xyz, deriv_xyz_to_rtp

SAVE = False
PLOT = True

nt = 100000
# TODO: have more points near periapsis
# S2 is very fast here, and zooming in shows nt=100000 is too low to measure
# precession angle well
# this should be easy, since t is barely different from proper time (zeta)

a = 0.0 # black hole angular momentum

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
half_rs = SGP_SUN * M / (SOL*SOL) # 1/2 schawrzschild radius / m

to_arcsec = half_rs / (1000 * R_0 * AU)
from_arcsec = 1000 * R_0 * AU / half_rs

sma *= from_arcsec
# sma *= 1/3600 * np.pi/180 * 1000 *  R_0 * 648000/np.pi * AU / half_rs
period *= YEAR * SOL / half_rs

T = 1.1 * period # units of 1/2 r_s / c

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
# z axis points away from earth
# x axis is north (declination)
# y axis is east (right ascension)
obs_from_orb = R_long @ R_incl @ R_arg
orb_from_obs = obs_from_orb.transpose() # property of orthogonal matrix

# spin orientation in observer frame
# (using same parameters as for orbit)
spin_phi = 0 # anti clockwise from x in x-y plane
spin_theta = 0 # angle made to z axis

R_spin_phi = np.array([[np.cos(spin_phi), np.sin(spin_phi), 0],
                      [-np.sin(spin_phi), np.cos(spin_phi), 0],
                      [0, 0, 1]])

R_spin_theta = np.array([[np.cos(spin_theta), 0, -np.sin(spin_theta)],
                         [0, 1, 0],
                         [np.sin(spin_theta), 0, np.cos(spin_theta)]])

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
pos_bl = xyz_to_bl(x_bh, a)

r0, theta0, phi0 = pos_bl
t0 = 0

# verified - change back to get starting point
# x = np.sqrt(r0*r0 + a*a)*np.sin(theta0)*np.cos(phi0)
# y = np.sqrt(r0*r0 + a*a)*np.sin(theta0)*np.sin(phi0)
# z = r0*np.cos(theta0)


# we need contravariant 4-momentum (= 4-velocity, as m=1)
# which is d/dtau of t,r,theta,phi

# first find d/dt of t,r,theta,phi
_x_dt = np.zeros(4)

# v_obs is d/dt of x,y,z

mat = deriv_xyz_to_rtp(x_bh, pos_bl, a)

_x_dt[0] = 1
_x_dt[1:4] = mat @ v_bh

# change to proper time derivative
metric0 = metric(np.array([0, r0, theta0, 0, 0, 0]), a) # only depends on r, theta
dt_dtau = 1/np.sqrt(-(metric0 @ _x_dt) @ _x_dt)

_p = dt_dtau * _x_dt

# multiply by metric for covariant 4-momentum
p_cov = metric0 @ _p

E = -p_cov[0] # by definition, for any stationary observer
p_r0 = p_cov[1]
p_theta0 = p_cov[2]
p_phi = p_cov[3]

orbit0 = np.array([t0, r0, theta0, phi0, p_r0, p_theta0])

# these are conserved:
# angular momentum (= r * p^phi for large r)
b = p_phi

# Carter's constant
_q = q(theta0, p_theta0, a, E, b)

zeta = np.linspace(0, T, nt + 1)
orbit = np.zeros((nt + 1, 6))
orbit_xyz = np.zeros((nt + 1, 3))

orbit = spi.odeint(deriv, orbit0, zeta, (a,E,b,_q))

orbit_xyz = bl_to_xyz(orbit[:, 1:4], a)

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

# TODO: fix numerics when obs -> bh transformation is non trivial
def cast_ray(xyz0, n0, zeta, a):
    # initial position of ray in BL coords
    pos0 = xyz_to_bl(xyz0, a)

    mat = deriv_xyz_to_rtp(xyz0, pos0, a)
    # initial [r, theta, phi, pr, pt]
    ray0 = np.concatenate((pos0, np.zeros(2)))

    metric0 = metric_l(ray0, a)

    _p = np.zeros(4)

    # _p[1:4] = np.linalg.solve(mat, n0)
    
    _p[1:4] = mat @ n0

    # from definition of energy E = -p^a u_a
    _p[0] = (-1 - _p[3] * metric0[0, 3])/metric0[0,0]

    _p_cov = metric0 @ _p

    ray0[3:5] = _p_cov[1:3]
    b = _p_cov[3]
    
    return spi.odeint(deriv_l, ray0, zeta, (a,b))
        

# returns ray trajectory, redshift f/f_e
# f = doppl * grav * f_emitted
def cast_ray_freqshift(xyz0, n0, zeta, star_vel_bh, a):
    # initial position of ray in BL coords
    pos0 = xyz_to_bl(xyz0, a)

    mat = deriv_xyz_to_rtp(xyz0, pos0, a)

    # initial [r, theta, phi, pr, pt]
    ray0 = np.concatenate((pos0, np.zeros(2)))

    metric0 = metric_l(ray0, a)

    _p = np.zeros(4)

    _p[1:4] = mat @ n0

    # from definition of energy E = -p^a u_a
    _p[0] = (-1 - _p[3] * metric0[0, 3])/metric0[0,0]

    _p_cov = metric0 @ _p

    ray0[3:5] = _p_cov[1:3]
    b = _p_cov[3]

    ray = spi.odeint(deriv_l, ray0, zeta, (a,b))

    # find redshift

    # fully GR

    # metric at emission
    metric1 = metric_l(ray[-1], a)
    inv_metric1 = inv_metric_l(ray[-1], a)
    # covariant four-momentum at emission
    p_cov1 = np.array([-1, ray[-1, 3], ray[-1, 4], b])
    p1 = inv_metric1 @ p_cov1

    mat1 = deriv_rtp_to_xyz(bl_to_xyz(ray[-1,0:3], a), ray[-1,0:3], a)
    mat1_inv = deriv_xyz_to_rtp(bl_to_xyz(ray[-1,0:3], a),ray[-1,0:3], a)

    # find ray direction at emission
    # cartesian dx/dt
    n1 = mat1 @ p1[1:4]
    # (should be normalised anyway)do
    n1 = n1 / np.linalg.norm(n1)

    # observer dx/dt
    # project star velocity onto ray direction
    # observer moves with star
    u1_dt_xyz = np.concatenate(([1], (star_vel_bh @ n1) * n1))

    u1_dt = np.ones(4)
    u1_dt[1:4] = mat1_inv @ u1_dt_xyz[1:4]
    # to change dx/dt to 4-velocity
    dt_dtau1 = 1/np.sqrt(-(metric1 @ u1_dt) @ u1_dt)
    u1 = dt_dtau1 * u1_dt

    # energy at emission
    E1 = - p_cov1 @ u1

    # same for observer
    # trivial when four-velocity of observer is time-only
    E2 = 1/np.sqrt(-metric0[0,0])

    freqshift_full = E2/E1

    # grav plus SR doppler
    # (for verification - should be very close if not the same)
    _grav = np.sqrt(-metric1[0,0])/np.sqrt(-metric0[0,0])
    beta = star_vel_bh @ n1 # radial veloctiy
    _doppler = np.sqrt((1 + beta)/(1 - beta))
    freqshift_full_alt = _doppler * _grav

    return freqshift_full, _doppler

def cast_earth_ray_freqshift(x, y, zeta_end, star_vel_bh, a):
    # some large number compared to r_s/2, and closest distance of star to Earth
    z_inf = -1e7

    # initial position(x0,y0,z0) in obs frame
    xyz0 = bh_from_obs @ np.array([x,y,z_inf])

    # rays coming to Earth from star
    n0 = bh_from_obs @ np.array([0, 0, -1])

    # only need start and end points
    zeta = np.array([0, zeta_end])

    return cast_ray_freqshift(xyz0, n0, zeta, star_vel_bh, a)


def cast_earth_ray(x, y, pos_bh, nt, a):
    # some large number compared to r_s/2, and closest distance of star to Earth
    z_inf = -1e7

    # initial position(x0,y0,z0) in obs frame
    xyz0 = bh_from_obs @ np.array([x,y,z_inf])

    # rays coming to Earth from star
    n0 = bh_from_obs @ np.array([0, 0, -1])

    # time taken by ray travelling to star along straight line
    # in units of 1/2 r_s / c
    str_dist = np.sqrt((pos_bh - xyz0) @ (pos_bh - xyz0))

    _zeta_0 = np.zeros(1)
    _zeta_1 = np.linspace(-str_dist+10, -str_dist-10, nt + 1)
    zeta = np.concatenate((_zeta_0, _zeta_1))

    return cast_ray(xyz0, n0, zeta, a), zeta

# minimum distance to pos from ray originating at (x,y,infinity)
# i.e. rays from Earth
def minimum_distance_sqr(x, y, pos, nt, a):
    pos_bh = bh_from_obs @ pos

    ray, zeta = cast_earth_ray(x, y, pos_bh, nt, a)

    ray_xyz = bl_to_xyz(ray[:, 0:3], a)

    dist_sqr_min = 1e50 # further than possible start
    i_min = len(ray) + 1
    for i in range(len(ray)):
        disp = ray_xyz[i] - pos_bh
        dist_sqr = disp @ disp
        if dist_sqr < dist_sqr_min:
            dist_sqr_min = dist_sqr
            i_min = i

    return dist_sqr_min, zeta[i_min]

# returns an array of lensed positions and redshifts f/f_em
# (note that the doppler and gravitational shifts are not entirely separable,
# as the lensing means the velocity is not actually projected radially
# when calculating the doppler shift)
# TODO: time delays from path length differences (probably negligible?)
def lensed_pos_and_freqshift(orbit, orbit_E, orbit_b, a):
    orbit_t = orbit[:, 0]
    xyzs_bh = bl_to_xyz(orbit[:, 1:4], a)

    pos_arr = np.zeros((len(orbit), 2)) # lensed x,y
    fs = np.zeros(len(orbit)) # full freq shift
    dopp = np.zeros(len(orbit)) # doppler only
    arrival_time = np.zeros(len(orbit)) # time difference from first ray
    
    for i in range(len(orbit)):
        print(i)
        xyz_obs = obs_from_bh @ xyzs_bh[i]

        min_dist_sqr_f = lambda xs: minimum_distance_sqr(xs[0], xs[1], xyz_obs, 256, a)[0]
        res = spo.minimize(min_dist_sqr_f,
                           xyz_obs[:2],
                           method='Nelder-Mead',
                           options={'fatol':1e-8,'xatol':1e-8})

        pos_arr[i, :] = res.x
        x, y = res.x

        # zeta needed to take ray to star
        zeta_end = minimum_distance_sqr(x, y, xyz_obs, 256, a)[1]

        inv_metric_here = inv_metric(orbit[i], a)
        star_p = inv_metric_here @ np.array([-orbit_E, orbit[i, 4], orbit[i, 5], b])
        star_vel_bh = deriv_rtp_to_xyz(xyzs_bh[i], orbit[i, 1:4], a) @ star_p[1:4]

        fs[i], dopp[i] = cast_earth_ray_freqshift(x, y, zeta_end, star_vel_bh, a)
        arrival_time[i] = orbit_t[i] + zeta_end
        
    arrival_time = arrival_time - arrival_time[0]
    return pos_arr, fs, dopp, arrival_time


sub_orbit = orbit[::(nt+1)//128]
sub_t = sub_orbit[:, 0]

lensed_xy, freqshift, doppler, delayed_t = lensed_pos_and_freqshift(sub_orbit, E, b, a)

unlensed_bh = bl_to_xyz(sub_orbit[:, 1:4], a)
unlensed_obs = np.zeros_like(unlensed_bh)
for i in range(len(unlensed_bh)):
    unlensed_obs[i] = obs_from_bh @ unlensed_bh[i]

deflec = np.linalg.norm(lensed_xy - unlensed_obs[:,:2], axis=1)
delayed_t *=  half_rs / (YEAR * SOL)

## find distance of rays to periapsis
#
#peri_obs = orbit_obs[imins][0]
##peri_obs = obs_from_bh @ bl_to_xyz(sub_orbit[30, 1:4], a)
#
## takes a long time; adjust to speed up
#nx = 32
#ny = nx
#
#width = 10
#xspace = np.linspace(peri_obs[0]-width, peri_obs[0]+width, nx)
#yspace = np.linspace(peri_obs[1]-width, peri_obs[1]+width, ny)
#
#min_dist = np.zeros((ny,nx))
#
#for i in range(nx):
#    for j in range(nx):
#        _x = xspace[i]
#        _y = yspace[j]
#        min_dist[j, i] = np.sqrt(minimum_distance_sqr(_x, _y, peri_obs, 1024, a)[0])
#        if j == 0 and i % 4 == 0:
#            print(i)
##        import sys
##        sys.stdout.flush()
##        import time
##        time.sleep(0.5)
##        print(i,j)
#
#min_dist_sqr_f = lambda xs: minimum_distance_sqr(xs[0], xs[1], peri_obs, 1024, a)[0]
#res = spo.minimize(min_dist_sqr_f,
#                   peri_obs[:2],
#                   method='Nelder-Mead',
#                           options={'fatol':1e-8,'xatol':1e-8})
#lensed_peri = res.x
#print('Closest Ray: ', np.sqrt(res.fun))
##
## cast test ray
#ray, _ = cast_earth_ray(lensed_peri[0], lensed_peri[1], bh_from_obs @ peri_obs, 1024, a)
#ray_xyz = bl_to_xyz(ray[:, 0:3], a)
#ray_obs = np.zeros_like(ray_xyz)
#for i in range(len(ray)):
#    ray_obs[i,:] = obs_from_bh @ ray_xyz[i,:]

if PLOT:
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0],[0])
    ax.plot(orbit_obs[:, 0], orbit_obs[:, 1], zs=orbit_obs[:, 2])

#    # check orbit is close to elliptical fit
#    ax.plot(test_ellipse_obs[:, 0], test_ellipse_obs[:, 1], zs=test_ellipse_obs[:, 2])
#    # check orbit is 'flat' after inverse transform
#    ax.plot(orbit_xyplane[:, 0], orbit_xyplane[:, 1], zs=orbit_xyplane[:, 2])
    
    # plot test ray
#    ax.scatter(ray_obs[:, 0], ray_obs[:, 1], zs=ray_obs[:, 2], marker='x')
#    ax.set_zlim(ray_obs[-1, 2] - 1, ray_obs[-1025, 2] + 1)
#    ax.set_xlim(ray_obs[-1, 0] - 10, ray_obs[-1025, 0] + 10)
#    ax.set_ylim(ray_obs[-1, 1] - 10, ray_obs[-1025, 1] + 10)

    # check spin transformation
    _s = np.array([0,0,-10000])
    s = obs_from_bh @ _s

    ax.plot([0,s[0]], [0,s[1]], zs=[0,s[2]])
#
#    # plot in orbital plane
#    plt.figure(figsize=(8,8))
#    plt.plot(orbit_orb[:, 0], orbit_orb[:, 1], 'k', linewidth=0.5)
#    plt.scatter([0],[0], c='k', marker='x')
#    plt.title("r_0 = {}, L = {}, E = {}".format(r0,b,E))
#
#    # view from Earth's sky
#    # (west, north)
#    # (-R.A., Decl.)
#    plt.figure(figsize=(8,8))
#    plt.plot(-orbit_obs[:, 1]*to_arcsec, orbit_obs[:, 0]*to_arcsec, 'k', linewidth=0.5)
#    plt.xlabel("-α (\'\')") # - right ascension
#    plt.ylabel("δ (\'\')") # declination
#    plt.scatter([0],[0], c='k', marker='x')
#    #plt.title("r_0 = {}, L = {}, E = {}".format(r0,b,E))

    # distance of rays to periapsis
#    plt.figure(figsize=(8,8))
#    cs = plt.contourf(xspace*to_arcsec*1e6, yspace*to_arcsec*1e6, min_dist,
#                      50, cmap='viridis')
#    plt.colorbar(cs, orientation='vertical')
#    
#    plt.scatter(peri_obs[0]*1e6*to_arcsec, peri_obs[1]*1e6*to_arcsec,
#                c='w', marker='x')
#    plt.scatter(lensed_peri[0]*1e6*to_arcsec, lensed_peri[1]*1e6*to_arcsec,
#                c='w', marker='o')
#    plt.xlabel("-α (μas)") # - right ascension
#    plt.ylabel("δ (μas)") # declination
    
    # lensing and redshift
    fig, ax1 = plt.subplots()
    
    for i in imins:
        plt.axvline(t[i] * half_rs / (YEAR * SOL), color='k',
                    linestyle='--', linewidth=1)
        
    for i in imaxs[1:]:
        plt.axvline(t[i] * half_rs / (YEAR * SOL), color='k',
                    linestyle=':', linewidth=1)
    
    ax1.set_ylabel("Deflection Angle / μas", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.plot(delayed_t, deflec*to_arcsec*1e6, color='b')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("f / f_e", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.plot(delayed_t, freqshift, color='r')
    
    # 2D lensing
    plt.figure(figsize=(8,8))
    
    xy = (lensed_xy - unlensed_obs[:,:2])*to_arcsec*1e6
    plt.plot(xy[:, 1], xy[:, 0])
    
    peri_obs = orbit_obs[imins][0]
    apo_obs = orbit_obs[imaxs][0]
    
    min_peri = lambda xs: minimum_distance_sqr(xs[0], xs[1], peri_obs, 1024, a)[0]
    min_apo = lambda xs: minimum_distance_sqr(xs[0], xs[1], apo_obs, 1024, a)[0]
    res_peri = spo.minimize(min_peri,
                            peri_obs[:2],
                            method='Nelder-Mead',
                            options={'fatol':1e-8,'xatol':1e-8})
    lensed_peri = res_peri.x
    res_apo = spo.minimize(min_apo,
                           peri_obs[:2],
                           method='Nelder-Mead',
                           options={'fatol':1e-8,'xatol':1e-8})
    lensed_apo = res_apo.x
    deflec_peri = lensed_peri - peri_obs[:2]
    deflec_apo = lensed_apo - apo_obs[:2]
    
    plt.scatter(deflec_peri[1]*1e6*to_arcsec, deflec_peri[0]*1e6*to_arcsec,
                c='k', marker='x')
    plt.scatter(deflec_apo[1]*1e6*to_arcsec, deflec_apo[0]*1e6*to_arcsec,
                    c='k', marker='o')
    
    plt.show()
