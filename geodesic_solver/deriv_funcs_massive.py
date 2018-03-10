# NOTE: All equations here assume mu = 1

import math
import numpy as np

def rho(y, a):
    t, r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)

def rho_sqr(y, a):
    t, r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost

def Delta(y, a):
    t, r, theta, phi, p_r, p_theta = y
    return r * r - 2 * r + a * a

def q(theta, p_theta, a, E, b):
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (b * b / (sint * sint) + a * a * (1-E*E))

def P(y, a, E, b):
    t, r, theta, phi, p_r, p_theta = y
    return E*(r * r + a * a) - a * b

def R(y, a, E, b, q):
    t, r, theta, phi, p_r, p_theta = y = y
    p = P(y, a, E, b)
    b_minus_aE = (b - a*E)
    return p * p - Delta(y, a) * (r*r + b_minus_aE * b_minus_aE + q)

def Theta(y, a, E, b, q):
    t, r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q - cost * cost * (b * b / (sint * sint) + a * a * (1-E*E))
#    return p_theta*p_theta

def deriv(y, zeta, a, E, b, q):   
    t, r, theta, phi, p_r, p_theta = y

    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _P = P(y, a, E, b)
    _R = R(y, a, E, b, q)
    _Theta = Theta(y, a,  E, b, q)
    
    cost = math.cos(theta)
    sint = math.sin(theta)
    cott = -math.tan(theta - math.pi / 2.0)
    
    sinsq = sint * sint
    cossq = cost * cost
    cotsq = cott * cott
    
    dt = (_P*(r*r+a*a)+a*_delta*(b-a*E*+a*E*cossq) )/(_delta*_rho_sqr)
    dr = _delta * p_r / _rho_sqr
    dtheta = p_theta / _rho_sqr
    dphi = (a*_P+_delta*(b-a*E+b*cotsq))/(_rho_sqr*_delta)
    
    dRdr = 4*E*r*_P-(2*r-2)*(r*r+(b-a*E)**2+q)-2*_delta*r
    dp_r = (p_theta*p_theta-p_r*p_r*(_rho_sqr*(r-1)/r -_delta))*r/(_rho_sqr*_rho_sqr) + (dRdr+2*(r-1)*_Theta - (_R+_delta*_Theta)*(2*(r-1)/_delta + 2*r/_rho_sqr))/(2*_delta*_rho_sqr)
    
    dqdtheta = -2*sint*cost*(a*a*(1-E*E)+b*b/sinsq)-2*b*b*cotsq*cott
    dp_theta = -a*a*cost*sint*(p_theta*p_theta+_delta*p_r*p_r)/(_rho_sqr*_rho_sqr) + (a*a*cost*sint*(_R+_delta*_Theta)/_rho_sqr-_delta*dqdtheta/2)/(_delta*_rho_sqr)
    
    # close (or past) to event horizon
    # TODO: do this better (dependent on a)
#    if (r - 1) < 1e-6:
#        print("TERMINATED")
#        return np.zeros(6)
        
    return np.array([dt, dr, dtheta, dphi, dp_r, dp_theta])

# covariant
def metric(y, a):
    _, r, theta, _, _, _ = y
    
    _rho_sqr = rho_sqr(y, a)
    sinsq = math.sin(theta)*math.sin(theta)
    
    gtt = -(1 - 2*r/_rho_sqr)
    gtphi = -2*a*r*sinsq/_rho_sqr
    grr = _rho_sqr / Delta(y, a)
    gthth = _rho_sqr
    gphiphi = sinsq * (r*r + a*a + 2*r*a*a*sinsq/_rho_sqr)
    
    return np.array([
                [gtt, 0, 0, gtphi],
                [0, grr, 0, 0],
                [0, 0, gthth, 0],
                [gtphi, 0, 0, gphiphi]
            ])
    
# contravariant
def inv_metric(y, a):
    _, r, theta, _, _, _ = y
    
    g = metric(y, a)
    
    gtt = g[0,0]
    gtphi = g[0, 3]
    grr = g[1,1]
    gthth = g[2,2]
    gphiphi = g[3,3]
    
    div = 1/(gphiphi*gtt - gtphi*gtphi)
    
    return np.array([
                [gphiphi*div, 0, 0, -gtphi*div],
                [0, 1/grr, 0, 0],
                [0, 0, 1/gthth, 0],
                [-gtphi*div, 0, 0, gtt*div]
            ])
    
def energy(y, a, b):
    _, _, _, _, p_r, p_theta = y
    
    invg = inv_metric(y,a)
    
    _a = invg[0,0]
    _b = -2*b*invg[0,3]
    _c = 1 + p_r*p_r*invg[1,1] + p_theta*p_theta*invg[2,2] + b*b*invg[3,3]
    
    right = math.sqrt(_b*_b - 4*_a*_c)/(2*_a)
    left = -_b/(2*_a)
    
    return max(left - right, left + right)

# returns time component of contravariant four momentum, given spatial part
def time_contra(r, theta, pr, pt, pp, a):
    
    g = metric(np.array([0, r, theta, 0, 0, 0]), a)
    
    gtt = g[0,0]
    gtphi = g[0, 3]
    grr = g[1,1]
    gthth = g[2,2]
    gphiphi = g[3,3]
    
    _a = gtt
    _b = 2*pp*gtphi
    _c = pr*pr*grr + pt*pt*gthth + pp*pp*gphiphi + 1
    
    right = math.sqrt(_b*_b - 4*_a*_c)/(2*_a)
    left = -_b/(2*_a)
    
    return max(left - right, left + right)