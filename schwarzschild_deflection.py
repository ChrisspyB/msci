import numpy as np
import matplotlib.pyplot as plt

from geodesic_solver.black_hole import BlackHole
from geodesic_solver.ray import Ray

# -- Customise --
PLOT = True
n_rays = 250
nt = 250  # time steps (time points - 1)
x0 = -10000 # initial distance from bh
y0_min, y0_max = 10,200 # min/max initial y coord of rays 
bh = BlackHole(a=0., M=1, R_0=1, v_r=0, incl=0, spin_theta=0, spin_phi=0)
# -- -------- --

zeta = np.linspace(0,np.abs(x0)*2,nt)
deflecs = []
rays = []
ys = np.linspace(y0_min,y0_max,n_rays)

for y in ys:
    r = Ray(bh,np.array([x0,y,0]),np.array([1,0,0]),zeta)
    dx = r.x[-1] - r.x[-2] 
    dy = r.y[-1] - r.y[-2]
    deflecs.append(np.arctan2(dy, dx))

    rays.append(r)

if PLOT:
    plt.figure()
    for i in range(n_rays):
        plt.plot(rays[i].x,rays[i].y, 'b')
    plt.title("Ray Trajectories")
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.figure()
    plt.plot(ys, np.abs(deflecs), label="From Traced Rays", linewidth = 1.0)
    plt.plot(ys, 4/ys, '--', label="Theory", linewidth = 1.0)
    plt.xlabel("Impact Parameter")
    plt.ylabel("Deflection Angle")
    plt.legend()
    plt.grid()
    plt.show()
