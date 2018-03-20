import numpy as np
import matplotlib.pyplot as plt

from geodesic_solver import BlackHole, Orbit

def precession_angle(theta, phi):
    bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2,
                   spin_theta=theta, spin_phi=phi)

    zeta = np.linspace(0, bh.from_years(18), 500000)
    
    s2 = Orbit(bh=bh,
               sma=0.1255,
               ecc=0.8839,
               incl=134.18,
               long_asc=226.94,
               arg_peri=65.51,
               period=16.0,
               zeta=zeta)
    
    imaxs = s2.i_apoapses
    
    apo_i = s2.xyz[0]
    apo_f = s2.xyz[imaxs[-1]]
    
    # u dot v = ||u|| ||v|| cos(theta)
    
    norms = (np.linalg.norm(apo_i) * np.linalg.norm(apo_f))
    angle = np.arccos((apo_i @ apo_f) / norms)
    
    return angle

ntheta = 16
nphi = ntheta

theta = np.linspace(0, 180, ntheta)
phi = np.linspace(0, 360, nphi)

angle = np.zeros((ntheta, nphi))

_ang = -1
_theta = -1
_phi = -1
for i in range(ntheta):
    for j in range(nphi):
        ang = precession_angle(theta[i], phi[j])
        angle[i, j] = ang
        if j == (nphi-1): print(i)
        if ang > _ang:
            _ang = ang
            _theta = theta[i]
            _phi = phi[j]


plt.close('all')

cs = plt.contourf(phi, theta, angle,
                  50, cmap='viridis')
plt.colorbar(cs, orientation='vertical')

# plot spin for max precession with orbit
bh = BlackHole(a=0.99, M=4.28e6, R_0=8.32, v_r=14.2,
               spin_theta=_theta, spin_phi=_phi)

zeta = np.linspace(0, bh.from_years(18), 500000)
s2 = Orbit(bh=bh,
           sma=0.1255,
           ecc=0.8839,
           incl=134.18,
           long_asc=226.94,
           arg_peri=65.51,
           period=16.0,
           zeta=zeta)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([0],[0])
ax.plot(s2.xyz[:, 0], s2.xyz[:, 1], zs=s2.xyz[:, 2])

# spin direction
_s = np.array([0,0,-10000])
s = bh.obs_from_bh(_s)

ax.plot([0,s[0]], [0,s[1]], zs=[0,s[2]])