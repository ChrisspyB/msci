import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from deriv_funcs_massive import deriv, q, metric, time_contra

# BASICALLY INPUTS

# OUTPUT PARAMETERS
nt = 10000
T = 1000000000 # units of 1/2 r_s / c

# SYSTEM PARAMETERS
a = 0.0 # black hole angular momentum
M = 4.28e6 # bh mass (solar masses)
R_0 = 8.32 # bh distance from earth (kpc)

def calc_orbits(sma,ecc,incl,long_asc,arg_peri,period):

	# CURRENTLY A COPY-PASTE FROM S2.PY 14/02/18: HAVE THIS FUNCTION IN ONE PLACE.

	# convert to natural units
	# degrees -> radians
	# sma/arcseconds -> half Schwarzschild radii
	# years -> 1/2 r_s / c

	incl *= np.pi/180
	long_asc *= np.pi/180
	arg_peri *= np.pi/180

	YEAR = 86400 * 365.25 # seconds
	SGP_SUN = 1.32712440018e20 # solar standard gravitational parameter / m^3 s^-2
	SOL = 299792458 # speed of light / ms^-1
	AU = 149597870700 # astronomical unit / m
	half_rs = SGP_SUN * M / (SOL*SOL) # 1/2 schawrzschild radius

	sma *= 1000 * R_0 * AU / (half_rs)
	period *= YEAR * SOL / half_rs

	# cartesian coordinates of apoapsis in orbital plane
	ecc_anom = np.pi # 0 at periapsis, pi at apoapsis
	x_orb = sma * np.array([np.cos(ecc_anom)-ecc,
	                        np.sqrt(1-ecc*ecc)*np.sin(ecc_anom),
	                        0])

	# looking from earth, S2 orbit is clockwise
	# so sign of y here must agree (currently does)
	# anti-clockwise in orbital plane
	v_orb = 2*np.pi*sma*sma/(np.linalg.norm(x_orb)*period) * \
	        np.array([-np.sin(ecc_anom),
	                  np.cos(ecc_anom)*np.sqrt(1-ecc*ecc), 0])

	# rotation matrices
	R_arg = np.array([[np.cos(arg_peri), -np.sin(arg_peri), 0],
	                   [np.sin(arg_peri), np.cos(arg_peri), 0],
	                   [0, 0, 1]])
	R_incl = np.array([[1, 0, 0],
	                   [0, np.cos(incl), -np.sin(incl)],
	                   [0, np.sin(incl), np.cos(incl)]])
	R_long = np.array([[np.cos(long_asc), -np.sin(long_asc), 0],
	                   [np.sin(long_asc), np.cos(long_asc), 0],
	                   [0, 0, 1]])

	# orbital plane to
	# cartesian coords in observer frame
	# z axis points toward earth
	# x axis is north (declination)
	# y axis is east (right ascension)
	obs_from_orb = R_long @ R_incl @ R_arg
	orb_from_obs = obs_from_orb.transpose() # property of orthogonal matrix

	# spin orientation in observer frame
	# (using same parameters as for orbit)
	spin_phi = 0 # anti clockwise from x in x-y plane
	spin_theta = np.pi # angle made to z axis

	R_spin_phi = np.array([[np.cos(spin_phi), np.sin(spin_phi), 0],
	                      [-np.sin(spin_phi), np.cos(spin_phi), 0],
	                      [0, 0, 1]])

	R_spin_theta = np.array([[-np.cos(spin_theta), 0, np.sin(spin_theta)],
	                         [0, 1, 0],
	                         [-np.sin(spin_theta), 0, -np.cos(spin_theta)]])

	# BH frame has spin in -z direction
	bh_from_obs = R_spin_theta @ R_spin_phi
	obs_from_bh = bh_from_obs.transpose()

	x_bh = bh_from_obs @ obs_from_orb @ x_orb
	v_bh = bh_from_obs @ obs_from_orb @ v_orb

	# find Boyer Lindquist position
	x0 = x_bh[0]
	y0 = x_bh[1]
	z0 = x_bh[2]

	phi0 = np.arctan2(y0,x0)

	a_xyz = a*a-x0*x0-y0*y0-z0*z0
	r0 = np.sqrt(-0.5*a_xyz + 0.5*np.sqrt(a_xyz*a_xyz + 4*a*a*z0*z0))

	theta0 = np.arccos(z0/r0)

	t0 = 0

	# we need contravariant 4-momentum (= 4-velocity, as m=1)
	# which is d/dtau of t,r,theta,phi

	# first find d/dt of t,r,theta,phi
	_u_dt = np.zeros(4)

	# v_obs is d/dt of x,y,z

	mat = np.array([
	        [r0*x0/(r0*r0 + a*a), -x0*np.tan(theta0 - np.pi/2), -y0],
	        [r0*y0/(r0*r0 + a*a), -y0*np.tan(theta0 - np.pi/2), x0],
	        [z0/r0, -r0*np.sin(theta0), 0]
	        ])

	_u_dt[0] = 1
	_u_dt[1:4] = np.linalg.solve(mat, v_bh)

	# change to proper time derivative
	metric0 = metric(np.array([0, r0, theta0, 0, 0, 0]), a) # only depends on r, theta
	dt_dtau = 1/np.sqrt(-(metric0 @ _u_dt) @ _u_dt)

	_p = dt_dtau * _u_dt

	# multiply by metric for covariant 4-momentum
	p_cov = metric0 @ _p

	E = -p_cov[0] # by definition, for any stationary observer
	p_r0 = p_cov[1]
	p_theta0 = p_cov[2]
	p_phi = p_cov[3]

	orbit0 = np.array([t0, r0, theta0, phi0, p_r0, p_theta0])

	# these are functions of orbit0:
	# angular momentum (= r * p^phi for large r)
	b = p_phi

	# Carter's constant
	_q = q(theta0, p_theta0, a, E, b)

	zeta = np.linspace(0, T, nt + 1)

	return spi.odeint(deriv, orbit0, zeta, (a,E,b,_q), atol = 1e-10)

def read_params(filename,names=[]): # if name == "", read all
	# assume first line is headings
	orbits = []
	with open(filename,"r") as f:
		f.readline() # skip first line
		for line in f:
			val = line.split()
			# for now, ignore errors and extra info
			# - errors probably useful for calculations
			# - spectral type useful for lensing later?
			if val[0] not in names: continue
			# print (val)
			val[1:14] = [float(n) for n in line.split()[1:14]]
			orbit = {
				"name":	val[0],
				"sma":	val[1],
				"ecc":	val[3],
				"incl":	val[5],
				"long_asc":	val[7],
				"arg_peri":	val[9],
				"period":	val[13],
			}
			orbits.append(orbit)
	return orbits

def draw_orbits(params, shownames=True):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter([0],[0])
	for o in params:
		orbit = calc_orbits(o["sma"],o["ecc"],o["incl"],o["long_asc"],o["arg_peri"],o["period"])
		orbit_x,orbit_y,orbit_z = get_orbitxyz(orbit,a)
		ax.plot(orbit_x, orbit_y, zs=orbit_z,label=o["name"])
	if shownames: plt.legend()
	plt.show()
	return

def get_orbitxyz(orbit,a):
	x = np.sqrt(orbit[:,1]**2+a*a)*np.sin(orbit[:,2])*np.cos(orbit[:,3])
	y = np.sqrt(orbit[:,1]**2+a*a)*np.sin(orbit[:,2])*np.sin(orbit[:,3])
	z = orbit[:,1]*np.cos(orbit[:,2])
	return x,y,z

def save_orbitsxyz(params,fname="orbitsxyz"):
	#Save x,y,z orbits to be drawn by render.py
	x,y,z = [],[],[]
	for o in params:
		orbit = calc_orbits(o["sma"],o["ecc"],o["incl"],o["long_asc"],o["arg_peri"],o["period"])
		orbit_x,orbit_y,orbit_z = get_orbitxyz(orbit,a)
		x.append(orbit_x)
		y.append(orbit_y)
		z.append(orbit_z)
	np.save(fname, np.dstack((x,y,z)))
	return
def save_orbitani(params,fname="orbitsani"):
	#Save t,x,y,z orbits to be animated by render.py
	t,x,y,z = [],[],[],[]
	for o in params:
		orbit = calc_orbits(o["sma"],o["ecc"],o["incl"],o["long_asc"],o["arg_peri"],o["period"])
		orbit_x,orbit_y,orbit_z = get_orbitxyz(orbit,a)
		orbit_t = orbit[:,0]
		x.append(orbit_x)
		y.append(orbit_y)
		z.append(orbit_z)
		t.append(orbit_t)
	np.save(fname, np.dstack((t,x,y,z)))
	return

# eg 

# convenient list of star names
starnames = ['S1','S2','S4','S6','S8','S9','S12','S13','S14','S17',
'S18','S19','S21','S22','S23','S24','S29','S31','S33','S38','S39',
'S42','S54','S55','S60','S66','S67','S71','S83','S85','S87','S89',
'S91','S96','S97','S145','S175','R34','R44']

#eg 1: render (plt) orbits for first 20 stars
# draw_orbits(read_params("gillessen_orbits.txt",starnames[:20]))
#eg 2: save (t,x,y,z) for 10 orbits, to be rendered in render.py
# save_orbitani(read_params("gillessen_orbits.txt",starnames[:10]),"orbitsani10")