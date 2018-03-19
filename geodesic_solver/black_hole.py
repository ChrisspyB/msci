import numpy as np

from .constants import YEAR, SGP_SUN, SOL, AU

class BlackHole:
    def __init__(self, a, M, R_0, v_r, spin_theta, spin_phi):
        """
        a -- angular momentum per unit mass in c=G=1 units - [0, 1]
        M -- mass (solar masses)
        R_0 -- distance to Earth (kPc)
        v_r -- radial velocity (km/s)
        spin_theta -- polar spin angle (degrees)
        spin_phi -- azimuthal spin angle (degrees)
        """
        
        half_rs = SGP_SUN * M / (SOL*SOL)
        to_arcsec = half_rs / (1000 * R_0 * AU)
        from_arcsec = 1000 * R_0 * AU / half_rs
        
        phi = spin_phi * np.pi/180
        theta = spin_theta * np.pi/180
        
        R_spin_phi = np.array([[np.cos(phi), np.sin(phi), 0],
                              [-np.sin(phi), np.cos(phi), 0],
                              [0, 0, 1]])
        
        R_spin_theta = np.array([[np.cos(theta), 0, -np.sin(theta)],
                                 [0, 1, 0],
                                 [np.sin(theta), 0, np.cos(theta)]])
        
        # BH frame has spin in -z direction
        self.__bh_from_obs = R_spin_theta @ R_spin_phi
        self.__obs_from_bh = self.__bh_from_obs.transpose()
        # Obs frame has:
        #     z away from earth
        #     x north (decl)
        #     y east (RA)
        
        self.__a = a
        self.__M = M
        self.__R_0 = R_0
        
        # convert to natural units
        self.__v_r = v_r * 1000 / SOL
        
        self.__half_rs = SGP_SUN * M / (SOL*SOL)
        
        self.__to_arc = to_arcsec
        self.__from_arc = from_arcsec
        
    @property
    def a(self):
        return self.__a
        
    def to_arcsec(self, dist):
        """Converts distance from natural units to arcseconds."""
        return dist * self.__half_rs / (1000 * self.__R_0 * AU)
    
    def from_arcsec(self, dist):
        """Converts distance from arcseconds to natural units."""
        return dist * 1000 * self.__R_0 * AU / self.__half_rs
    
    def to_years(self, t):
        """Converts time from natural units to years."""
        return t * self.__half_rs / (YEAR * SOL)
    
    def from_years(self, t):
        """Converts time from years to natural units."""
        return t * YEAR * SOL / self.__half_rs
    
#    def radial_velocity(self):
#        """Radial velocity of black hole (natural units: v/c)."""
#        return self.__v_r
    
    def bh_from_obs(self, vec):
        """Transform a vector from observer frame to BH frame."""
        return self.__bh_from_obs @ vec
    
    def obs_from_bh(self, vec):
        """Transform a vector from BH frame to observer frame."""
        return self.__obs_from_bh @ vec
    
    def __xyz_to_rtp(self, pos):
        x, y, z = pos
        a = self.__a
    
        phi = np.arctan2(y,x)
        a_xyz = a*a-x*x-y*y-z*z
        r = np.sqrt(-0.5*a_xyz + 0.5*np.sqrt(a_xyz*a_xyz + 4*a*a*z*z))
        theta = np.arccos(z/r)
        
        return np.array([r,theta,phi])
    
    def doppler(self):
        """Returns doppler shift due to radial velocity."""
        v = self.__v_r
        return np.sqrt((1 - v)/(1 + v))

    # vectorized
    def xyz_to_rtp(self, xyzs):
        """
        Changes cartesian coords to BL.
        """
        if xyzs.ndim == 1:
            return self.__xyz_to_rtp(xyzs)
        else:
            assert(xyzs.shape == (len(xyzs), 3))
            bl_pos = np.zeros((len(xyzs), 3))
            for i in range(len(xyzs)):
                bl_pos[i, :] = self.__xyz_to_rtp(xyzs[i, :])
            return bl_pos
    
    def __rtp_to_xyz(self, pos):
        r, theta, phi = pos
        a = self.__a
        
        x = np.sqrt(r*r + a * a) * np.sin(theta) * np.cos(phi)
        y = np.sqrt(r*r + a * a) * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        return np.array([x,y,z])
    
    # vectorized
    def rtp_to_xyz(self, bl_pos):
        """
        Changes BL coords to cartesian.
        """
        if bl_pos.ndim == 1:
            return self.__rtp_to_xyz(bl_pos)
        else:
            assert(bl_pos.shape == (len(bl_pos), 3))
            
            a = self.__a
            
            xyzs = np.zeros((len(bl_pos), 3))
    
            xyzs[:, 0] = np.sqrt(bl_pos[:, 0]*bl_pos[:, 0] + a * a) * \
                np.sin(bl_pos[:, 1]) * np.cos(bl_pos[:, 2])
            xyzs[:, 1] = np.sqrt(bl_pos[:, 0]*bl_pos[:, 0] + a * a) * \
                np.sin(bl_pos[:, 1]) * np.sin(bl_pos[:, 2])
            xyzs[:, 2] = bl_pos[:, 0] * np.cos(bl_pos[:, 1])
    
            return xyzs
    
    def deriv_rtp_to_xyz(self, xyz, rtp):
        """
        Returns matrix that changes derivatives of BL coords to cartesian.
        """
        x0, y0, z0 = xyz
        r0, theta0, phi0 = rtp
        a = self.__a
        
        return np.array([
                        [r0*x0/(r0*r0 + a*a), -x0*np.tan(theta0 - np.pi/2), -y0],
                        [r0*y0/(r0*r0 + a*a), -y0*np.tan(theta0 - np.pi/2), x0],
                        [z0/r0, -r0*np.sin(theta0), 0]
                        ])

    def deriv_xyz_to_rtp(self, xyz, rtp):
        """
        Returns matrix that changes derivatives of cartesian coords to BL.
        """
        x, y, z = xyz
        r, t, p = rtp
        a = self.__a
        
        D = -np.tan(t - np.pi/2)/(r*r + a*a)
        
        return np.array([
                        [x/r,y/r,z/r],
                        np.array([x*D,y*D,(z*np.cos(t) - r)/(r*r*np.sin(t))]),
                        [np.cos(p)*np.cos(p)*(-y)/(x*x),np.cos(p)*np.cos(p)/x,0]
                        ])