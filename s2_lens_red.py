import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from geodesic_solver import BlackHole, Orbit

PLOT = True

bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2, spin_theta=0, spin_phi=0)

zeta = np.concatenate(([0], np.linspace(bh.from_years(7.5), bh.from_years(9), 100000)))
#zeta = np.concatenate(([0], np.linspace(bh.from_years(8), bh.from_years(8.4), 100000)))

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
obs_t, deflec, fshift, dopp, grav = s2.earth_obs(8)
fshift *= bh.doppler

obs_t = obs_t[1:]
deflec = deflec[1:]
fshift = fshift[1:]
dopp = dopp[1:]
grav = grav[1:]

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
    plt.gca().invert_xaxis()

    # lensing and redshift
    fig, ax1 = plt.subplots()
    fig.set_size_inches(10,6)

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
    
#    ax2 = ax1.twinx()
#    ax2.set_ylabel("Frequency Shift (km/s)", color='r')
#    ax2.tick_params(axis='y', labelcolor='r')
#    frequency shift in km/s
#    kms = bh.to_kms(fshift - 1)
#    ax2.plot(bh.to_years(obs_t), kms, color='r')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Δz due to Gravitational Effects (km/s)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    kms = bh.to_kms(1/dopp - 1/fshift)
    ax2.plot(bh.to_years(obs_t), kms, color='r')

#    ax2 = ax1.twinx()
#    ax2.set_ylabel("Apparent v_r due to gravitational redshift (km/s)", color='r')
#    ax2.tick_params(axis='y', labelcolor='r')
#    f = grav
#    radial_vel = (1 - f*f)/(1 + f*f)
#    ax2.plot(bh.to_years(obs_t), bh.to_kms(radial_vel), color='r')