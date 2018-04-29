import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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
phi = np.linspace(-180, 180, nphi)

#theta = np.linspace(130, 140, ntheta)
#phi = np.linspace(310, 320, nphi)

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


# for orbit theta,phi and plotting
bh = BlackHole(a=0.0, M=4.28e6, R_0=8.32, v_r=14.2,
                   spin_theta=orb_theta, spin_phi=orb_phi)
zeta = np.linspace(0, bh.from_years(18), 10000) 
s2 = Orbit(bh=bh,
           sma=0.1255,
           ecc=0.8839,
           incl=134.18,
           long_asc=226.94,
           arg_peri=65.51,
           period=16.0,
           zeta=zeta)
_n = np.array([0,0,1])
n = s2.obs_from_orb(_n)
x,y,z = n
orb_theta = np.arccos(z)*180/np.pi
orb_phi = np.arctan2(y,x)*180/np.pi

plt.close('all')

cs = plt.contourf(phi, theta, angle,
                  64, cmap='viridis')
plt.colorbar(cs, orientation='vertical')
plt.xlabel('ϕ / °')
plt.ylabel('θ / °')
plt.scatter(orb_phi, orb_theta, marker='o', color='w')
if orb_phi<0:
    plt.scatter(180 + orb_phi, 180-orb_theta, marker='x', color='w')
else:
    plt.scatter(orb_phi - 180, 180-orb_theta, marker='x', color='w')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter([0],[0])

# spin direction
_s = np.array([0,0,-10000])
s = bh.obs_from_bh(_s)

ax.plot(s2.xyz[:,0], s2.xyz[:,1], zs=s2.xyz[:,2])
ax.plot([0,s[0]], [0,s[1]], zs=[0,s[2]])
ax.plot([0,10000*n[0]], [0,10000*n[1]], zs=[0,10000*n[2]])
# scale axes 1:1:1 to check n is normal to orbit
ax.set_xlim3d(-5000,35000)
ax.set_ylim3d(-20000,20000)
ax.set_zlim3d(-5000,35000)