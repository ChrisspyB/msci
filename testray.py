from geodesic_solver.black_hole import BlackHole
from geodesic_solver.ray import Ray
import numpy as np
import matplotlib.pyplot as plt

b = BlackHole(a=0.995, M=1, R_0=1, v_r=0, spin_theta=0, spin_phi=0)

fig = plt.figure()
plt.scatter([0],[0])
for y in np.linspace(7,10,10):
    r = Ray(b,np.array([-10,y,0]),np.array([1,0,0]),np.linspace(0,40,100))
    plt.plot(r.x,r.y)
plt.show()