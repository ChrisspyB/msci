import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from geodesic_solver import BlackHole, Orbit

PLOT = True

bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2, spin_theta=0, spin_phi=0)

zeta = np.linspace(0, bh.from_years(18), 100000)
#zeta = np.concatenate((np.linspace(0, bh.from_years(0.1), 2),
#                       np.linspace(bh.from_years(16.3), bh.from_years(16.4), 10000)))

s2 = Orbit(bh=bh,
           sma=0.1255,
           ecc=0.8839,
           incl=134.18,
           long_asc=226.94,
           arg_peri=65.51,
           period=16.0,
           zeta=zeta)

t = s2.orbit[:, 0]
xyz = s2.xyz
#obs_t, deflec, fshift, dopp, grav = s2.earth_obs(100)

if PLOT:
    plt.close('all')
    
    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([0],[0])
    ax.plot(xyz[:, 0], xyz[:, 1], zs=xyz[:, 2])

    # check spin transformation
    _s = np.array([0,0,-10000])
    s = bh.obs_from_bh(_s)

    ax.plot([0,s[0]], [0,s[1]], zs=[0,s[2]])

    # view from Earth's sky
    # (west, north)
    # (-R.A., Decl.)
    plt.figure(figsize=(8,8))
    plt.plot(bh.to_arcsec(xyz[:, 1]), bh.to_arcsec(xyz[:, 0]),
             'k', linewidth=0.5)
    plt.xlabel("α (\'\')") # - right ascension
    plt.ylabel("δ (\'\')") # declination
    plt.scatter([0],[0], c='k', marker='x')

    # lensing and redshift
    fig, ax1 = plt.subplots()

    for i in s2.i_periapses:
        plt.axvline(bh.to_years(t[i]), color='k',
                    linestyle='--', linewidth=1)

    for i in s2.i_apoapses[1:]:
        plt.axvline(bh.to_years(t[i]), color='k',
                    linestyle=':', linewidth=1)

    ax1.set_ylabel("Deflection Angle (μas)", color='b')
    ax1.set_xlabel("t (yr)")
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.plot(bh.to_years(obs_t), bh.to_arcsec(deflec*1e6), color='b')

    ax2 = ax1.twinx()
    ax2.set_ylabel("v_r (km/s)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    radial_vel = (1 - fshift*fshift)/(1 + fshift*fshift)
    ax2.plot(bh.to_years(obs_t), bh.to_kms(radial_vel), color='r')