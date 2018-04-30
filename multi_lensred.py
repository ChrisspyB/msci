import numpy as np
import matplotlib.pyplot as plt

from geodesic_solver import BlackHole, Orbit

def read_params(filename): # if name == "", read all
    # assume first line is headings
    orbits = []
    with open(filename,"r") as f:
        f.readline() # skip first line
        for line in f:
            val = line.split()
            val[1:14] = [float(n) for n in line.split()[1:14]]
            orbit = [
                    val[0],
                    val[1],
                    val[3],
                    val[5],
                    val[7],
                    val[9],
                    val[11],
                    val[13],
                    ]
            orbits.append(orbit)
        return orbits

paramss = read_params('gillessen_orbits.txt')

bh = BlackHole(a=0.0, M=4.28e6, R_0=8.32, v_r=14.2, spin_theta=0, spin_phi=0)

obs = []

for params in paramss:
    name, sma, ecc, incl, long_asc, arg_peri, p_date, period = params
    
    if name in ['S14', 'S38', 'S55']:
        print(name)
        print(sma, ecc, p_date, period)
        if p_date < 2017:
            print(bh.from_arcsec(sma*(1-ecc))/2, p_date + period)
        else:
            print(bh.from_arcsec(sma*(1-ecc))/2, p_date)
            
        zeta = np.linspace(0, bh.from_years(period*1.1), 10000)
        orb = Orbit(bh, sma, ecc, incl, long_asc, arg_peri, period, zeta)
        
        obs_t, deflec, fshift, dopp, grav = orb.earth_obs(256)
        
        obs.append((obs_t, deflec, fshift, dopp, grav, p_date, period))
    
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
fig.set_size_inches(10,8)


ax1.title.set_text('S14')
# translate 0 to apoapsis
t = bh.to_years(obs[0][0]) + obs[0][5] + obs[0][6]/2
deflec = obs[0][1]
fshift = obs[0][2]
dopp = obs[0][3]

ax1.axvline(obs[0][5] + obs[0][6], color='k', linestyle='--', linewidth=1)

ax1.set_ylabel("Deflection Angle (μas)", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.plot(t, bh.to_arcsec(deflec*1e6), color='b')

ax11 = ax1.twinx()
ax11.set_ylabel("Δz grav (km/s)", color='r')
ax11.tick_params(axis='y', labelcolor='r')
kms = bh.to_kms(1/dopp - 1/fshift)
ax11.plot(t, kms, color='r')

ax2.title.set_text('S38')
t = bh.to_years(obs[1][0]) + obs[1][5] + obs[1][6]/2
deflec = obs[1][1]
fshift = obs[1][2]
dopp = obs[1][3]

ax2.axvline(obs[1][5] + obs[1][6], color='k', linestyle='--', linewidth=1)

ax2.set_ylabel("Deflection Angle (μas)", color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.plot(t, bh.to_arcsec(deflec*1e6), color='b')

ax22 = ax2.twinx()
ax22.set_ylabel("Δz grav (km/s)", color='r')
ax22.tick_params(axis='y', labelcolor='r')
kms = bh.to_kms(1/dopp - 1/fshift)
ax22.plot(t, kms, color='r')

ax3.title.set_text('S55')
t = bh.to_years(obs[2][0]) + obs[2][5] + obs[2][6]/2
deflec = obs[2][1]
fshift = obs[2][2]
dopp = obs[2][3]

ax3.axvline(obs[2][5] + obs[2][6], color='k', linestyle='--', linewidth=1)

ax3.set_ylabel("Deflection Angle (μas)", color='b')
ax3.set_xlabel("Date (yr)")
ax3.tick_params(axis='y', labelcolor='b')
ax3.plot(t, bh.to_arcsec(deflec*1e6), color='b')

ax33 = ax3.twinx()
ax33.set_ylabel("Δz grav (km/s)", color='r')
ax33.tick_params(axis='y', labelcolor='r')
kms = bh.to_kms(1/dopp - 1/fshift)
ax33.plot(t, kms, color='r')