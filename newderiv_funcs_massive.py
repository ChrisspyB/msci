# NOTE: All equations here assume mu = 1

import math
import numpy as np

def rho(y, a):
    r, theta, phi, p_r, p_theta, t = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)

def rho_sqr(y, a):
    r, theta, phi, p_r, p_theta, t = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost

def Delta(y, a):
    r, theta, phi, p_r, p_theta, t = y
    return r * r - 2 * r + a * a

def q(y, p_theta, a, E, b):
    r, theta, phi, p_r, p_theta, t = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (b * b / (sint * sint) + a * a * (1-E*E))

def P(y, a, E, b):
    # == P
    r, theta, phi, p_r, p_theta, t = y
    return E*(r * r + a * a) - a * b

def R(y, a, E, b, q):
    r, theta, phi, p_r, p_theta, t = y
    p = P(y, a, E, b)
    b_minus_aE = (b - a*E)
    return p * p - Delta(y, a) * (r*r + b_minus_aE * b_minus_aE + q)

def Theta(y, a, E, b, q):
    r, theta, phi, p_r, p_theta, t = y
    # cost = math.cos(theta)
    # sint = math.sin(theta)
    # return q - cost * cost * (b * b / (sint * sint) + a * a * (1-E*E))
    return p_theta*p_theta

def deriv_massive(y, zeta, a, E, b, q):   
    # note different y
    
    r, theta, phi, p_r, p_theta, t = y

    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _P = P(y, a, E, b)
    _R = R(y, a, E, b, q)
    _Theta = Theta(y, a,  E, b, q)
    _Lz = b
    cost = math.cos(theta)
    sint = math.sin(theta)
    cott = sint / cost
    sinsq = sint * sint
    cossq = cost * cost
    cotsq = cott * cott

    dr = _delta * p_r / _rho_sqr
    dtheta = p_theta / _rho_sqr

    dt = (_P*(r*r+a*a)+a*_delta*(b-a*E*+a*E*cossq) )/(_delta*_rho_sqr)
    dphi = (a*_P+_delta*(b-a*E+b*cotsq))/(_rho_sqr*_delta)
    
    dqdtheta = -2*sint*cost*(a*a*(1-E*E)+b*b/sinsq)-2*b*b*cotsq*cott
    dp_theta = -a*a*cost*sint*(p_theta*p_theta+_delta*p_r*p_r)/_rho_sqr + (a*a*cost*sint*(_R+_delta*_Theta)/_rho_sqr-_delta*dqdtheta/2)/(_delta*_rho_sqr)
    
    dRdr = 4*E*r*_P-(2*r-2)*(r*r+(b-a*E)**2+q)-2*_delta*r
    dp_r = (p_theta*p_theta-p_r*p_r*(_rho_sqr*(r-1)/r -_delta))*r/(_rho_sqr*_rho_sqr) + (dRdr+2*(r-1)*_Theta - (_R+_delta*_Theta)*(2*(r-1)/_delta + 2*r/_rho_sqr))/(2*_delta*_rho_sqr)
    
    # reached event horizon
    if (r - 2) < 1e-6:
        return np.zeros(6)
        
    return np.array([dr,dtheta,dphi,dp_r,dp_theta,dt])