import numpy as np
import matplotlib.pyplot as plt

from geodesic_solver import BlackHole, Ray

# -- Customise --
PLOT = True
n_rays = 250
nt = 250  # time steps (time points - 1)
x0 = -10000 # initial distance from bh
y0_min, y0_max = 7,200 # min/max initial y coord of rays
a=0.99
bh = BlackHole(a=a, M=1, R_0=1, v_r=0, spin_theta=0, spin_phi=0)
# -- -------- --

deflecs = []
rays = []
theory = []
zeta = np.linspace(0,np.abs(x0)*2,nt)
ys = np.concatenate((
        np.linspace(-y0_max,-y0_min,n_rays//2), np.linspace(y0_min,y0_max,n_rays//2)))

for y in ys:
    r = Ray(bh,np.array([x0,y,0]),np.array([1,0,0]),zeta)
    dx = r.x[-1] - r.x[-2] 
    dy = r.y[-1] - r.y[-2]
    deflecs.append(np.arctan2(dy, dx))
    rays.append(r)

    b = np.abs(y)
    s = np.sign(y)
    theory.append(4/b \
             + (15*np.pi/4 - 12*s*a) * 1/(b*b) \
             + (128/3 - 10*np.pi*s*a + 4*a*a) * 1/(b*b*b))

if PLOT:
    plt.figure()
    for i in range(n_rays):
        plt.plot(rays[i].x,rays[i].y, 'b')
    plt.title("Ray Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure()
    plt.plot(ys, np.abs(deflecs), label="From Traced Rays", linewidth = 1.0)
    plt.plot(ys,theory,"--",label="Theory",linewidth = 1.0)
    plt.xlabel("Impact Parameter")
    plt.ylabel("Deflection Angle")
    plt.legend()
    plt.grid()
    plt.show()
