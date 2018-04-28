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
                    val[13],
                    ]
            orbits.append(orbit)
        return orbits

paramss = read_params('gillessen_orbits.txt')

bh = BlackHole(a=0.0, M=4.28e6, R_0=8.32, v_r=14.2, spin_theta=0, spin_phi=0)
angs = []
th_angs = []

plt.close('all')
plt.figure(figsize=(8,8))

for params in paramss:
    name, sma, ecc, incl, long_asc, arg_peri, period = params
    
    print(name)
    
    zeta = np.linspace(0, bh.from_years(1.1*period), 500000)
    orb = Orbit(bh, sma, ecc, incl, long_asc, arg_peri, period, zeta)
    
    imaxs = orb.i_apoapses
    
    apo_i = orb.xyz[0]
    apo_f = orb.xyz[imaxs[-1]]
    
    # u dot v = ||u|| ||v|| cos(theta)
    
    norms = (np.linalg.norm(apo_i) * np.linalg.norm(apo_f))
    ang = np.arccos((apo_i @ apo_f) / norms)
    angs.append(ang)
    
    th_ang = 6 * np.pi / (bh.from_arcsec(sma) * (1 - ecc*ecc))
    th_angs.append(th_ang)
    
    plt.scatter(ang, abs(ang - th_ang), marker='4')
    plt.annotate(name, (ang, abs(ang - th_ang)))
    
plt.xlabel('Precession Angle')
plt.ylabel('Difference from Theory')