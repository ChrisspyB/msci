import numpy as np
import matplotlib.pyplot as plt

from geodesic_solver import BlackHole, Orbit

def precession_angle(theta, phi):
    bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2,
                   spin_theta=theta, spin_phi=phi)

    zeta = np.linspace(0, bh.from_years(18), 100000)
#    zeta = np.concatenate((np.linspace(0, bh.from_years(0.2), 10000),
#                           np.linspace(bh.from_years(15), bh.from_years(17), 20000)))
    
    s2 = Orbit(bh=bh,
               sma=0.1255,
               ecc=0.8839,
               incl=134.18,
               long_asc=226.94,
               arg_peri=65.51,
               period=16.0,
               zeta=zeta)
    
    apo_i = s2.xyz[s2.i_apoapses[0]]
    apo_f = s2.xyz[s2.i_apoapses[1]]
    
    # u dot v = ||u|| ||v|| cos(theta)
    
    norms = (np.linalg.norm(apo_i) * np.linalg.norm(apo_f))
    angle = np.arccos((apo_i @ apo_f) / norms)
    
    return angle

ntheta = 25
nphi = ntheta

theta = np.linspace(0, 180, ntheta)
phi = np.linspace(0, 360, nphi)

angle = np.zeros((ntheta, nphi))

for i in range(ntheta):
    for j in range(nphi):
        angle[i, j] = precession_angle(theta[i], phi[j])
        if j == (nphi-1): print(i)
        
cs = plt.contourf(phi, theta, angle,
                  50, cmap='viridis')
plt.colorbar(cs, orientation='vertical')