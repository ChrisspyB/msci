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

def xyz_to_bl(pos, a):
    x, y, z = pos

    phi = np.arctan2(y,x)
    a_xyz = a*a-x*x-y*y-z*z
    r = np.sqrt(-0.5*a_xyz + 0.5*np.sqrt(a_xyz*a_xyz + 4*a*a*z*z))
    theta = np.arccos(z/r)
    
    return np.array([r,theta,phi])

def bl_to_xyz(pos, a):
    r, theta, phi = pos
    
    x = np.sqrt(r*r + a * a) * np.sin(theta) * np.cos(phi)
    y = np.sqrt(r*r + a * a) * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return np.array([x,y,z])