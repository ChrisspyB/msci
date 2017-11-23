import math
import numpy as np

def b_0(r, a):
    return -(r * r * r - 3 * r * r + a * a * r + a * a) / (a * (r - 1))


def b_0_inv(b, a):
    # return 1+(-3+a*(a+b))/(3**(1/3)*(-9+9*a*a+(1/6)*math.sqrt((54-54*a*a)**2+108*(-3+a*(a+b))**3))**(1/3))-(-9+9*a*a+(1/6)*math.sqrt((54-54*a*a)**2+108*(-3+a*(a+b))**3))**(1/3)/3**(2/3)
    # awful minimisation algorithm - proof of concept
    # should be analytic
    r_1 = 2 + 2 * math.cos(2 * math.acos(-a) / 3)
    r_2 = 2 + 2 * math.cos(2 * math.acos(a) / 3)
    
    step = 1e-5
    tol = 1e-4
    guess = r_1

    while b_0(guess) - b > tol:
        guess += step
        if guess > r_2:
            return -1

    return guess


# write this as a function of b in the range [r_1, r_2]
# we should be able to invert b_0(r) in this range, or write below in
# terms of b_0(r_0)
def q_0(r, a):
    return -r * r * r * (r * r * r - 6 * r * r + 9 * r - 4 * a * a) / (a * a * (r - 1) * (r - 1))


def q_0b(b):
    return q_0(b_0_inv(b))


def rho(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)


def rho_sqr(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost


def Sigma(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    sint = math.sin(theta)
    rr_plus_aa = r * r + a * a
    return math.sqrt(rr_plus_aa * rr_plus_aa - a * a * sint * sint)


def Delta(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    return r * r - 2 * r + a * a


def omega(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    sig = Sigma(y, a)
    return 2 * a * r / (sig * sig)


def pomega(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    return Sigma(y, a) * math.sin(theta) / rho(y, a)


def alpha(y, a):
    return rho(y, a) * math.sqrt(Delta(y, a)) / Sigma(y, a)


def E_f(y, n_phi, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    return 1 / (alpha(y, a) + omega(y, a) * pomega(y, a) * n_phi)


def q(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (p_phi * p_phi / (sint * sint) - a * a)


def P(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    return r * r + a * a - a * p_phi


def R(y, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    p = P(y, a)
    p_minus_a = (p_phi - a)
    return p * p - Delta(y, a) * (p_minus_a * p_minus_a + q(y, a))


def Theta(y, a):
    # TODO: Check if this is ptheta**2 only
    r, theta, phi, p_r, p_theta, p_phi = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q(y, a) - cost * cost * (p_phi * p_phi / (sint * sint) - a * a)


def deriv_light(y, zeta, a):
    r, theta, phi, p_r, p_theta, p_phi = y
    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _rho_qua = _rho_sqr * _rho_sqr
    _P = P(y, a)
    _R = R(y, a)
    _Theta = Theta(y, a)
    _b = p_phi
    _q = q(y, a)

    cot = -math.tan(theta - math.pi / 2.0)
    aasincos = a * a * math.sin(theta) * math.cos(theta)

    dr = p_r * _delta / _rho_sqr
    dtheta = p_theta * 1.0 / _rho_sqr
    dphi = (a * _P / _delta + _b - a + _b * cot * cot) / _rho_sqr
    dp_r = p_r * p_r * (_delta * r / _rho_qua - (r - 1) / _rho_sqr) + p_theta * p_theta * r / _rho_qua + ((2 * r * _P - (r - 1) * ((_b - a) * (_b - a) + _q)) * _delta * _rho_sqr -_R * ((r - 1) * _rho_sqr + _delta * r)) / (_delta * _delta * _rho_qua) - r * _Theta / _rho_qua
    dp_theta = -aasincos * _delta * p_r * p_r / _rho_qua - aasincos * p_theta * p_theta / _rho_qua + (_R * aasincos - _delta * _rho_sqr * (aasincos - _b * _b * (cot + cot * cot * cot))) / (_delta * _rho_qua) + _Theta * aasincos / _rho_qua
    dp_phi = 0
    
    # reached event horizon
    if (r - 2) < 1e-6:
        return np.zeros(6)

    return np.array([dr, dtheta, dphi, dp_r, dp_theta, dp_phi])