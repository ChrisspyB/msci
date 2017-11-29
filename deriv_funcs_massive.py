import math
import numpy as np

def rho(y, a):
    t, r, theta, phi = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)


def rho_sqr(y, a):
    t, r, theta, phi = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost


def Sigma(y, a):
    t, r, theta, phi = y
    sint = math.sin(theta)
    rr_plus_aa = r * r + a * a
    return math.sqrt(rr_plus_aa * rr_plus_aa - a * a * sint * sint)


def Delta(y, a):
    t, r, theta, phi = y
    return r * r - 2 * r + a * a


def omega(y, a):
    t, r, theta, phi = y
    sig = Sigma(y, a)
    return 2 * a * r / (sig * sig)


def pomega(y, a):
    t, r, theta, phi = y
    return Sigma(y, a) * math.sin(theta) / rho(y, a)


def alpha(y, a):
    return rho(y, a) * math.sqrt(Delta(y, a)) / Sigma(y, a)


def q(y, p_theta, a, musq, E, b):
    t, r, theta, phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (b * b / (sint * sint) + a * a * (musq-E*E))


def P(y, a, E, b):
    # == P
    t, r, theta, phi = y
    return E*(r * r + a * a) - a * b


def R(y, a, musq, E, b, q):
    t, r, theta, phi = y
    p = P(y, a, E, b)
    b_minus_aE = (b - a*E)
    return p * p - Delta(y, a) * (musq*r*r + b_minus_aE * b_minus_aE + q)


def Theta(y, a, musq, E, b, q):
    t, r, theta, phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q - cost * cost * (b * b / (sint * sint) + a * a * (musq-E*E))

def deriv_massive(y, zeta, a, mu, E, b, q):
    musq = mu*mu
    
    # note different y
    t, r, theta, phi = y

    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _P = P(y, a, E, b)
    _R = R(y, a, musq, E, b, q)
    _Theta = Theta(y, a, musq, E, b, q)
    _Lz = b

    sinsq = math.sin(theta) * math.sin(theta)

    dtheta = math.sqrt(_Theta)
    dr = math.sqrt(_R)
    dphi = -(a * E - _Lz / sinsq) + _P * a / _delta
    dt = -a * (a * E * sinsq - _Lz) + _P * (r * r + a * a)/_delta
    
    # reached event horizon
    if (r - 2) < 1e-6:
        return np.zeros(4)
        

    return np.array([dt,dr,dtheta,dphi])/_rho_sqr