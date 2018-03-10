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