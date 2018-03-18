import numpy as np
import scipy.integrate as spi
import scipy.optimize as spo

from .deriv_funcs_light import deriv, metric, inv_metric
class Ray:
    def __init__(self, bh, obs_xyz0, obs_n0, zeta, eps=1.49012e-8):
        """
        bh -- System's BlackHole object
        xyz0 -- initial ray position [x,y,z] in obs coords
        n0   -- initial ray direction in obs coords
        zeta -- times at which to evaluate ray coords
        eps -- Threshold parameter for integrator
        """
        self.__bh = bh
        self.__integrate(obs_xyz0, obs_n0, zeta, eps) # autocast

    @property
    def ray(self):
        #[t,r,theta,phi,pr,pt]
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

    def __integrate(self, xyz0, n0, zeta, eps):   
        """
        Calcualte ray trajectory
        Args defined in __init__
        """     
        a = self.__bh.a
        
        bh_xyz0 = self.__bh.bh_from_obs(xyz0)

        rtp0 = self.__bh.xyz_to_rtp(bh_xyz0)

        ray0 = np.concatenate(([0], rtp0, np.zeros(2))) # [t,r,theta,phi,pr,pt]

        mat = self.__bh.deriv_xyz_to_rtp(xyz0, rtp0)
        metric0 = metric(ray0, a)

        _p = np.zeros(4) # 4 momentum 
        _p[1:4] = mat @ self.__bh.bh_from_obs(n0)
        _p[0] = (-1 - _p[3] * metric0[0, 3])/metric0[0,0] # by defn: E = -p^a u_a

        _p_cov = metric0 @ _p

        ray0[4:6] = _p_cov[1:3]
        b = _p_cov[3]

        ray = spi.odeint(deriv, ray0, zeta, (a,b), rtol=eps, atol=eps)

        self.__t = ray[:, 0]
        self.__ray = ray
        self.__zeta = zeta
        self.__b = b
        
        bh_x = np.sqrt(ray[:, 1]**2 + a * a) * \
            np.sin(ray[:, 2]) * np.cos(ray[:, 3])
        bh_y = np.sqrt(ray[:, 1]**2 + a * a) * \
            np.sin(ray[:, 2]) * np.sin(ray[:, 3])
        bh_z = ray[:, 1] * np.cos(ray[:, 2])
        
        self.__x = np.zeros_like(bh_x)
        self.__y = np.zeros_like(bh_x)
        self.__z = np.zeros_like(bh_x)
        
        for i in range(len(ray)):
            bh_xyz = np.array([bh_x[i],bh_y[i],bh_z[i]])
            obs_xyz = self.__bh.obs_from_bh(bh_xyz)
            self.__x[i] = obs_xyz[0]
            self.__y[i] = obs_xyz[1]
            self.__z[i] = obs_xyz[2]
        
        return
    
    def min_sqr_dist(self, obs_xyz):
        """
        Returns minimum distance from ray to point obs_xyz,
        and affine parameter of the closest point on the ray
        """
        bh_xyz = self.__bh.bh_from_obs(obs_xyz)
    
        ray_xyz = self.__bh.rtp_to_xyz(self.__ray[:, 1:4])
    
        dist_sqr_min = 1e50 # further than possible start
        i_min = len(self.__ray) + 1
        for i in range(len(self.__ray)):
            disp = ray_xyz[i] - bh_xyz
            dist_sqr = disp @ disp
            if dist_sqr < dist_sqr_min:
                dist_sqr_min = dist_sqr
                i_min = i
    
        return dist_sqr_min, self.__zeta[i_min]
    
    def freqshift(self, bh_emit_vel):
        """
        Returns frequency ratio between end points of ray (end/start),
        and also pure doppler and gravitational shifts
        (which are actually not fully seperable, and so are approximations)
        
        bh_emit_vel -- 3-velocity of ray source
        """
        bh = self.__bh
        a = bh.a
        b = self.__b
        
        # end points of ray
        # [t,r,theta,phi,pr,pt]
        detec = self.__ray[0]
        emit = self.__ray[-1]
        # find redshift
    
        # fully GR
    
        # metric at emission
        metric_e = metric(emit, a)
        inv_metric_e = inv_metric(emit, a)
        # covariant four-momentum at emission
        p_cov_e = np.array([-1, emit[4], emit[5], b])
        p_e = inv_metric_e @ p_cov_e
        
        rtp_e = emit[1:4]
        xyz_e = bh.rtp_to_xyz(rtp_e)
        mat_e = bh.deriv_rtp_to_xyz(xyz_e, rtp_e)
        mat_e_inv = bh.deriv_xyz_to_rtp(xyz_e, rtp_e)
        # find ray direction at emission
        # cartesian dx/dt
        n_e = mat_e @ p_e[1:4]
        # (should be normalised anyway)do
        n_e = n_e / np.linalg.norm(n_e)
    
        # observer dx/dt
        # project star velocity onto ray direction
        # observer moves with star
        u_dt_xyz = np.concatenate(([1], (bh_emit_vel @ n_e) * n_e))
    
        u_dt = np.ones(4)
        u_dt[1:4] = mat_e_inv @ u_dt_xyz[1:4]
        # to change dx/dt to 4-velocity
        dt_dtau1 = 1/np.sqrt(-(metric_e @ u_dt) @ u_dt)
        u = dt_dtau1 * u_dt
    
        # energy at emission
        E1 = - p_cov_e @ u
    
        # same for observer
        metric_detec = metric(detec, a)
        # trivial when four-velocity of observer is time-only
        E2 = 1/np.sqrt(-metric_detec[0,0])
    
        freqshift = E2/E1
    
        # gravitational and SR doppler
        # (for verification - should be very close if not the same)
        _grav = np.sqrt(-metric_e[0,0])/np.sqrt(-metric_detec[0,0])
        beta = bh_emit_vel @ n_e # radial veloctiy
        _doppler = np.sqrt((1 + beta)/(1 - beta))
    
        return freqshift, _doppler, _grav
    
    @staticmethod
    def from_earth(bh, obs_xyz, eps=1e-8, nt=1024):
        """
        Minimises the distance between rays casted from
        Earth (backwards in time along the z axis) and obs_xyz,
        returning the x, y of the closest ray.
        
        eps -- tolerance for minimisation
        nt -- number of time steps near target
        """
        z_inf = -1e7
        
        def cast(x,y):
            # initial position
            xyz0 = np.array([x,y,z_inf])
            # rays coming to Earth from star
            n0 = np.array([0, 0, -1])
            
            # time taken by ray travelling to star along straight line
            # in units of 1/2 r_s / c
            str_dist = np.sqrt((obs_xyz - xyz0) @ (obs_xyz - xyz0))
        
            _zeta_0 = np.zeros(1)
            _zeta_1 = np.linspace(-str_dist+10, -str_dist-10, nt + 1)
            zeta = np.concatenate((_zeta_0, _zeta_1))
            
            r = Ray(bh, xyz0, n0, zeta)
            
            d2_min, z_min = r.min_sqr_dist(obs_xyz)
            
            return d2_min, z_min
        
        # minimise this
        obj = lambda xy: cast(xy[0], xy[1])[0]
        
        res = spo.minimize(obj,
                           obs_xyz[:2],
                           method='Nelder-Mead',
                           options={'xatol':eps})
        
        return res.x # x, y