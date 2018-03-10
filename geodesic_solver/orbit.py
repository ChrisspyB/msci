import numpy as np

class Orbit:
    def __init__(self, bh, sma, ecc, incl, long_asc, arg_peri, period):
        """
        bh -- BlackHole object, around which the star orbits
        sma -- semi-major axis (arcseconds)
        ecc -- eccentricity
        incl -- inclination (degrees)
        long_asc -- longitude of ascending node (degrees)
        arg_peri -- argument of periapsis (degrees)
        period -- period (years)
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
        
        def obs_from_orb(self, vec):
            """Transform a vector from orbit frame to observer frame."""
            return self.__obs_from_orb @ vec
        
        def orb_from_obs(self, vec):
            """Transform a vector from observer frame to orbit frame."""
            return self.__orb_from_obs @ vec