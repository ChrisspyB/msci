import numpy as np
import scipy.integrate as spi

from .deriv_funcs_massive import deriv, metric, inv_metric, q
from .ray import Ray
from .utils import minima, maxima

class Orbit:
    def __init__(self, bh, sma, ecc, incl, long_asc, arg_peri, period, zeta):
        """
        bh -- BlackHole object, around which the star orbits
        sma -- semi-major axis (arcseconds)
        ecc -- eccentricity
        incl -- inclination (degrees)
        long_asc -- longitude of ascending node (degrees)
        arg_peri -- argument of periapsis (degrees)
        period -- period (years)
        zeta -- array of values of elapsed proper time to integrate to
        """

        self.__bh = bh
        self.__ecc = ecc

        # to natural units
        self.__sma = bh.from_arcsec(sma)
        self.__period = bh.from_years(period)

        # to radians
        _incl = incl * np.pi/180
        _long_asc = long_asc * np.pi/180
        _arg_peri = arg_peri * np.pi/180

        R_arg = np.array([[np.cos(_arg_peri), -np.sin(_arg_peri), 0],
                          [np.sin(_arg_peri), np.cos(_arg_peri), 0],
                          [0, 0, 1]])
        R_incl = np.array([[1, 0, 0],
                           [0, np.cos(_incl), -np.sin(_incl)],
                           [0, np.sin(_incl), np.cos(_incl)]])
        R_long = np.array([[np.cos(_long_asc), -np.sin(_long_asc), 0],
                           [np.sin(_long_asc), np.cos(_long_asc), 0],
                           [0, 0, 1]])

        # orbital plane (anticlockwise, periapsis at +x) to
        # cartesian coords in observer frame
        # z axis points away from earth
        # x axis is north (declination)
        # y axis is east (right ascension)
        self.__obs_from_orb = R_long @ R_incl @ R_arg
        self.__orb_from_obs = self.__obs_from_orb.transpose()
        self.__integrate(zeta)

    @property
    def orbit(self):
        return self.__orbit
    
    @property
    def xyz(self):
        """Cartesian observer coords."""
        return self.__xyz

    def obs_from_orb(self, vec):
        """Transform a vector from orbit frame to observer frame."""
        return self.__obs_from_orb @ vec

    def orb_from_obs(self, vec):
        """Transform a vector from observer frame to orbit frame."""
        return self.__orb_from_obs @ vec

    def __integrate(self, zeta, ecc_anom=np.pi):
        """
        Calculate trajectory, returning points (t, r, theta, phi, pr, ptheta)
        for each element of zeta (elapsed proper time).

        ecc_anom -- eccentric anomaly of start point (radians),
                    defaults to pi, i.e. the apoapsis
        """

        a = self.__bh.a
        sma = self.__sma
        period = self.__period
        ecc = self.__ecc

        # cartesian coordinates of apoapsis in orbital plane
        x_orb = sma * np.array([np.cos(ecc_anom)-ecc,
                                np.sqrt(1-ecc*ecc)*np.sin(ecc_anom),
                                0])

        v_orb = 2*np.pi*sma*sma/(np.linalg.norm(x_orb)*period) * \
                np.array([-np.sin(ecc_anom),
                          np.cos(ecc_anom)*np.sqrt(1-ecc*ecc), 0])

        x_bh = self.__bh.bh_from_obs(self.obs_from_orb(x_orb))
        v_bh = self.__bh.bh_from_obs(self.obs_from_orb(v_orb))

        pos_bl = self.__bh.xyz_to_rtp(x_bh)
        r0, theta0, phi0 = pos_bl

        mat = self.__bh.deriv_xyz_to_rtp(x_bh, pos_bl)

        # find d/dt of t,r,theta,phi
        dx_dt = np.zeros(4)

        dx_dt[0] = 1
        dx_dt[1:4] = mat @ v_bh

        # change to proper time derivative
        metric0 = metric(np.array([0, r0, theta0, 0, 0, 0]), a) # only depends on r, theta
        dt_dtau = 1/np.sqrt(-(metric0 @ dx_dt) @ dx_dt)

        p = dt_dtau * dx_dt

        # multiply by metric for covariant 4-momentum
        p_cov = metric0 @ p

        E = -p_cov[0] # by definition, for any stationary observer
        p_r0 = p_cov[1]
        p_theta0 = p_cov[2]
        p_phi = p_cov[3]

        orbit0 = np.array([0, r0, theta0, phi0, p_r0, p_theta0])

        # these are conserved:
        # angular momentum (= r * p^phi for large r)
        b = p_phi
        # Carter's constant
        _q = q(theta0, p_theta0, a, E, b)

        self.__orbit = spi.odeint(deriv, orbit0, zeta, (a,E,b,_q))
        self.__b = b
        self.__E = E
        
        xyz = self.__bh.rtp_to_xyz(self.__orbit[:, 1:4])
        self.__xyz = np.zeros_like(xyz)
        
        for i in range(len(xyz)):
            self.__xyz[i] = self.__bh.obs_from_bh(xyz[i])

    def earth_obs(self, n):
        """
        Picks n sub-points from orbit and
        returns time, lensing and freqshift information
        """
        # [t, r, theta, phi, pr, pth]
        subs = self.orbit[::len(self.orbit)//n]
        xyz = self.__bh.rtp_to_xyz(subs[:, 1:4])
        orb_t = subs[:, 0]

        obs_t = np.zeros(len(subs))
        deflec = np.zeros(len(subs))
        fshift = np.zeros(len(subs))
        dopp = np.zeros(len(subs))
        grav = np.zeros(len(subs))

        for i in range(len(subs)):
            p_cov = np.array([-self.__E, subs[i, 4], subs[i, 5], self.__b])
            inv_g = inv_metric(subs[i], self.__bh.a)
            p = inv_g @ p_cov

            # find star 3-vel w.r.t. coordinate time
            u = np.zeros(3)
            u = self.__bh.deriv_rtp_to_xyz(xyz[i], subs[i, 1:4]) @ p[1:4]
            # divide by dt/dtau
            u = u / p[0]

            t, l_xy, shft, _dopp, _grav = Ray.earth_obs(self.__bh, xyz[i], u)

            deflec[i] = np.linalg.norm(l_xy - xyz[i, :2])
            fshift[i] = shft
            dopp[i] = _dopp
            grav[i] = _grav
            obs_t[i] = orb_t[i] + t
            print(i)

        obs_t = obs_t - obs_t[0]

        return obs_t, deflec, fshift, dopp, grav

    @property
    def apoapses(self):
        imaxs = maxima(self.__orbit[:,4])
        return self.__orbit[imaxs]

    @property
    def periapses(self):
        imins = minima(self.__orbit[:,4])
        return self.__orbit[imins]

    @property
    def i_apoapses(self):
        imaxs = maxima(self.__orbit[:,4])
        return imaxs

    @property
    def i_periapses(self):
        imins = minima(self.__orbit[:,4])
        return imins