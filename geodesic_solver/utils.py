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

def minimise(f, xy0, r=10, tol=0.001):
    """
    Recursive grid minimisation à la Grould
    
    r -- initial half grid width;
          default is about 50 micro arcseconds
    tol -- half grid width at which to terminate;
           default is about 0.005 micro arcseconds
    """
    _min = f(xy0)
    _xy = xy0.copy()
    
    x0, y0 = xy0
    
    for x in np.linspace(x0-r, x0+r, 5):
        for y in np.linspace(y0-r, y0+r, 5):
            xy = np.array([x, y])
            _f = f(xy)
            if _f < _min:
                _min = _f
                _xy = xy
    
    if r > tol:
        return minimise(f, _xy, 2*r/5, tol)
    else:
        return _xy