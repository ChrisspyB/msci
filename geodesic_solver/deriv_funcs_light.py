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
    r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    return math.sqrt(r * r + a * a * cost * cost)


def rho_sqr(y, a):
    r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    return r * r + a * a * cost * cost


def Sigma(y, a):
    r, theta, phi, p_r, p_theta = y
    sint = math.sin(theta)
    rr_plus_aa = r * r + a * a
    return math.sqrt(rr_plus_aa * rr_plus_aa - a * a * sint * sint)


def Delta(y, a):
    r, theta, phi, p_r, p_theta = y
    return r * r - 2 * r + a * a


def omega(y, a):
    r, theta, phi, p_r, p_theta = y
    sig = Sigma(y, a)
    return 2 * a * r / (sig * sig)


def pomega(y, a):
    r, theta, phi, p_r, p_theta = y
    return Sigma(y, a) * math.sin(theta) / rho(y, a)


def alpha(y, a):
    return rho(y, a) * math.sqrt(Delta(y, a)) / Sigma(y, a)


def E_f(y, n_phi, a):
    r, theta, phi, p_r, p_theta = y
    return 1 / (alpha(y, a) + omega(y, a) * pomega(y, a) * n_phi)


def q(y, a, b):
    r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return p_theta * p_theta + cost * cost * (b * b / (sint * sint) - a * a)


def P(y, a, b):
    r, theta, phi, p_r, p_theta = y
    return r * r + a * a - a * b


def R(y, a, b):
    r, theta, phi, p_r, p_theta = y
    p = P(y, a, b)
    p_minus_a = (b - a)
    return p * p - Delta(y, a) * (p_minus_a * p_minus_a + q(y, a, b))


def Theta(y, a, b):
    # TODO: Check if this is ptheta**2 only
    r, theta, phi, p_r, p_theta = y
    cost = math.cos(theta)
    sint = math.sin(theta)
    return q(y, a, b) - cost * cost * (b * b / (sint * sint) - a * a)


def deriv(y, zeta, a, b):
    r, theta, phi, p_r, p_theta = y
    _delta = Delta(y, a)
    _rho_sqr = rho_sqr(y, a)
    _rho_qua = _rho_sqr * _rho_sqr
    _P = P(y, a, b)
    _R = R(y, a, b)
    _Theta = Theta(y, a, b)
    _b = b
    _q = q(y, a, b)

    cot = -math.tan(theta - math.pi / 2.0)
    aasincos = a * a * math.sin(theta) * math.cos(theta)

    dr = p_r * _delta / _rho_sqr
    dtheta = p_theta * 1.0 / _rho_sqr
    dphi = (a * _P / _delta + _b - a + _b * cot * cot) / _rho_sqr
    dp_r = p_r * p_r * (_delta * r / _rho_qua - (r - 1) / _rho_sqr) + p_theta * p_theta * r / _rho_qua + ((2 * r * _P - (r - 1) * ((_b - a) * (_b - a) + _q)) * _delta * _rho_sqr -_R * ((r - 1) * _rho_sqr + _delta * r)) / (_delta * _delta * _rho_qua) - r * _Theta / _rho_qua
    dp_theta = -aasincos * _delta * p_r * p_r / _rho_qua - aasincos * p_theta * p_theta / _rho_qua + (_R * aasincos - _delta * _rho_sqr * (aasincos - _b * _b * (cot + cot * cot * cot))) / (_delta * _rho_qua) + _Theta * aasincos / _rho_qua
    
    return np.array([dr, dtheta, dphi, dp_r, dp_theta])

# use this carefully, coordinate system of n0 is not that of pos0
def ray0_b_from_pos0_n0(pos0, n0, a):
    ray0 = np.concatenate((pos0, np.zeros(2)))
    ray0[3] = n0[0] * E_f(ray0, n0[2], a) * rho(ray0, a) / math.sqrt(Delta(ray0, a))
    ray0[4] = n0[1] * E_f(ray0, n0[2], a) * rho(ray0, a)
    b = n0[2] * E_f(ray0, n0[2], a) * pomega(ray0, a)
    return ray0, b

# covariant
def metric(y, a):
    r, theta, _, _, _ = y
    
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
    r, theta, _, _, _ = y
    
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