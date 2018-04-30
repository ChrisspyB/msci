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
            print(bh.from_arcsec(sma*(1-ecc))/2, p_date + period)
            
        zeta = np.linspace(0, bh.from_years(period*1.1), 10000)
        orb = Orbit(bh, sma, ecc, incl, long_asc, arg_peri, period, zeta)
        
        obs_t, deflec, fshift, dopp, grav = orb.earth_obs(128)
        
        obs.append((obs_t, deflec, fshift, dopp, grav))