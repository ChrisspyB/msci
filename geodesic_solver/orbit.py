import numpy as np

from .constants import *
from .black_hole import *

sma = 0.1255 # arcseconds in Earth's sky
ecc = 0.8839
incl = 134.18 # degrees
long_asc = 226.94 # degrees
arg_peri = 65.51 # degrees
period = 16.0 # years

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
        self.__incl = incl * np.pi/180
        self.__long_asc = long_asc * np.pi/180
        self.__arg_peri = arg_peri * np.pi/180