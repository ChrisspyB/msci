import numpy as np
import scipy.integrate as spi

from .deriv_funcs_light import deriv, metric

class Ray:
    def __init__(self, bh, xyz0, n0, zeta):
        """
        bh -- System's BlackHole object
        xyz0 -- initial ray position [x,y,z]
        n0   -- initial ray direction
        zeta -- times at which to evaluate ray coords
        """
        self.__bh = bh
        self.__integrate(xyz0,n0,zeta) # autocast

    @property
    def ray(self):
        #[r,theta,phi,pr,pt]
        return self.__ray

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def z(self):
        return self.__z
    
    @property
    def t(self):
        return self.__t

    def __integrate(self, xyz0, n0, zeta):        
        a = self.__bh.a

        rtp0 = self.__bh.xyz_to_rtp(xyz0) 
        ray0 = np.concatenate((rtp0, np.zeros(2))) # [t,r,theta,phi,pr,pt]

        mat = self.__bh.deriv_xyz_to_rtp(xyz0, rtp0)
        metric0 = metric(ray0, a)

        _p = np.zeros(4) # 4 momentum 
        _p[1:4] = mat @ n0
        _p[0] = (-1 - _p[3] * metric0[0, 3])/metric0[0,0] # by defn: E = -p^a u_a

        _p_cov = metric0 @ _p

        ray0[4:6] = _p_cov[1:3]
        b = _p_cov[3]

        ray = spi.odeint(deriv, ray0, zeta, (a,b))
        
        self.__x = np.sqrt(ray[:, 1]**2 + a * a) * \
            np.sin(ray[:, 2]) * np.cos(ray[:, 3])
        self.__y = np.sqrt(ray[:, 0]**2 + a * a) * \
            np.sin(ray[:, 2]) * np.sin(ray[:, 3])
        self.__z = ray[:, 1] * np.cos(ray[:, 2])
        self.__t = ray[:, 0]
        self.__ray = ray
        return