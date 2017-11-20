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


def E_f(y, n_phi, a):
    t, r, theta, phi = y
    return 1 / (alpha(y, a) + omega(y, a) * pomega(y, a) * n_phi)


def q(y, a, musq, b, p_theta):
    t, r, theta, phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (b * b / (sint * sint) + a * a * (musq-1))


def P(y, a, b):
    # == P
    t, r, theta, phi = y
    return r * r + a * a - a * b


def R(y, a, musq, b, q):
    t, r, theta, phi = y
    p = P(y, a, b)
    p_minus_a = (p - a)
    return p * p - Delta(y, a) * (p_minus_a * p_minus_a + musq*r*r + q)


def Theta(y, a, musq, b, q):
    # TODO: Check if this is ptheta**2 only
    t, r, theta, phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q - cost * cost * (b * b / (sint * sint) + a * a * (musq-1))

def deriv_massive(y,zeta,a,b,q,mu):
    musq = mu*mu
    
    # note different y
    t, r, theta, phi = y

    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _P = P(y, a, musq, b)
    _R = R(y, a, musq, b, q)
    _Theta = Theta(y, a, musq,b,q)
    _Lz = b

    sinsq = math.sin(theta) * math.sin(theta)

    dtheta = math.sqrt(_Theta)
    dr = math.sqrt(_R)
    dphi = -(a - _Lz / sinsq) + a / _delta * _P
    dt = -a * (a * sinsq - _Lz) + (r * r + a * a)/_delta * _P

    return np.array([dt,dr,dtheta,dphi])/_rho_sqr