import numpy as np

def maxima(deriv):
    inds = []
    for i in range(len(deriv) - 1):
        # turning point is where deriv crosses zero
        if deriv[i+1] < 0 and deriv[i] >= 0:
            di = abs(deriv[i])
            di1 = abs(deriv[i+1])
            # find closest point to zero
            if di > di1:
                inds.append(i+1)
            else:
                inds.append(i)
    return inds

def minima(deriv):
    inds = []
    for i in range(len(deriv) - 1):
        # turning point is where deriv crosses zero
        if deriv[i+1] > 0 and deriv[i] <= 0:
            di = abs(deriv[i])
            di1 = abs(deriv[i+1])
            # find closest point to zero
            if di > di1:
                inds.append(i+1)
            else:
                inds.append(i)
    return inds

def _xyz_to_bl(pos, a):
    x, y, z = pos

    phi = np.arctan2(y,x)
    a_xyz = a*a-x*x-y*y-z*z
    r = np.sqrt(-0.5*a_xyz + 0.5*np.sqrt(a_xyz*a_xyz + 4*a*a*z*z))
    theta = np.arccos(z/r)
    
    return np.array([r,theta,phi])

# vectorized
def xyz_to_bl(xyzs, a):
    if xyzs.ndim == 1:
        return _xyz_to_bl(xyzs, a)
    else:
        assert(xyzs.shape == (len(xyzs), 3))
        bl_pos = np.zeros((len(xyzs), 3))
        for i in range(len(xyzs)):
            bl_pos[i, :] = _xyz_to_bl(xyzs[i, :], a)
        return bl_pos

def _bl_to_xyz(pos, a):
    r, theta, phi = pos
    
    x = np.sqrt(r*r + a * a) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(r*r + a * a) * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x,y,z])

# vectorized
def bl_to_xyz(bl_pos, a):
    if bl_pos.ndim == 1:
        return _bl_to_xyz(bl_pos, a)
    else:
        assert(bl_pos.shape == (len(bl_pos), 3))
        
        xyzs = np.zeros((len(bl_pos), 3))

        xyzs[:, 0] = np.sqrt(bl_pos[:, 0]*bl_pos[:, 0] + a * a) * \
            np.sin(bl_pos[:, 1]) * np.cos(bl_pos[:, 2])
        xyzs[:, 1] = np.sqrt(bl_pos[:, 0]*bl_pos[:, 0] + a * a) * \
            np.sin(bl_pos[:, 1]) * np.sin(bl_pos[:, 2])
        xyzs[:, 2] = bl_pos[:, 0] * np.cos(bl_pos[:, 1])

        return xyzs
    
def deriv_rtp_to_xyz(xyz, rtp, a):
    x0, y0, z0 = xyz
    r0, theta0, phi0 = rtp
    
    return np.array([
                    [r0*x0/(r0*r0 + a*a), -x0*np.tan(theta0 - np.pi/2), -y0],
                    [r0*y0/(r0*r0 + a*a), -y0*np.tan(theta0 - np.pi/2), x0],
                    [z0/r0, -r0*np.sin(theta0), 0]
                    ])

def deriv_xyz_to_rtp(xyz, rtp, a):
    x, y, z = xyz
    r, t, p = rtp
    
#    return np.linalg.inv(deriv_rtp_to_xyz(xyz, rtp, a))
    
    D = -np.tan(t - np.pi/2)/(r*r + a*a)
    
    return np.array([
                    [x/r,y/r,z/r],
                    np.array([x*D,y*D,(z*np.cos(t) - r)/(r*r*np.sin(t))]),
                    [np.cos(p)*np.cos(p)*(-y)/(x*x),np.cos(p)*np.cos(p)/x,0]
                    ])