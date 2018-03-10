import numpy as np

from .constants import *

class BlackHole:
    def __init__(self, a, M, R_0, v_r, incl, ):
        """
        a -- angular momentum per unit mass in c=G=1 units - [0, 1]
        M -- mass (solar masses)
        R_0 -- distance to Earth (kPc)
        v_r -- radial velocity (km/s)
        """
        
        half_rs = SGP_SUN * M / (SOL*SOL)
        to_arcsec = half_rs / (1000 * R_0 * AU)
        from_arcsec = 1000 * R_0 * AU / half_rs
        
        self.__a = a
        self.__M = M
        self.__R_0 = R_0
        
        # convert to natural units
        self.__v_r = v_r * 1000 / SOL
        
        self.__half_rs = SGP_SUN * M / (SOL*SOL)
        
        self.__to_arc = to_arcsec
        self.__from_arc = from_arcsec
        
    def to_arcsec(self, dist):
        """Converts distance from natural units to arcseconds."""
        return dist * self.__half_rs / (1000 * R_0 * AU)
    
    def from_arcsec(self, dist):
        """Converts distance from arcseconds to natural units."""
        return dist * 1000 * R_0 * AU / self.__half_rs
    
    def to_years(self, t):
        """Converts time from natural units to years."""
        return t * self.__half_rs / (YEAR * SOL)
    
    def from_years(self, t):
        """Converts time from years to natural units."""
        return t * YEAR * SOL / self.__half_rs
    
    def radial_velocity(self):
        """Radial velocity of black hole (natural units: v/c)."""
        return self.__v_r