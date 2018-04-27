import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from geodesic_solver import BlackHole, Orbit

PLOT = True

bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2, spin_theta=0, spin_phi=0)

def orbit(tol):
    zeta = np.linspace(0, bh.from_years(2*16), 32)
    
    s2 = Orbit(bh=bh,
               sma=0.1255,
               ecc=0.8839,
               incl=134.18,
               long_asc=226.94,
               arg_peri=65.51,
               period=16.0,
               zeta=zeta,
               tol=tol)
    
    return s2.xyz

xyz_ref = orbit(1e-11)

tols = np.logspace(-11, -2, 16)
errs = np.zeros_like(tols)

for i in range(len(tols)):
    xyz = orbit(tols[i])
    errs[i] = np.linalg.norm(xyz - xyz_ref)/np.linalg.norm(xyz_ref)

if PLOT:
    plt.close('all')
    plt.figure(figsize=(8,8))
    plt.plot(xyz_ref[:, 0], bh.to_arcsec(xyz_ref[:, 1]),
             'k', linewidth=0.5)
    plt.xlabel("x") # - right ascension
    plt.ylabel("y") # declination
    plt.scatter([0],[0], c='k', marker='x')
    plt.gca().invert_xaxis()
    
    plt.figure(figsize=(8,8))
    plt.loglog(tols, errs)
    plt.xlabel('Tolerance')
    plt.ylabel('Error')