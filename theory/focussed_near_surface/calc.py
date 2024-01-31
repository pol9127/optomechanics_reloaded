import optomechanics.theory.focussed_near_surface._custom_module as cm
from collections import Iterable
import numpy as np

def E_inc(kx, ky, f, w0, k1, E0):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.E_inc(kx, ky, f, w0, k1, E0)

def E_inf(kx, ky, f, w0, k1, E0):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.E_inf(kx, ky, f, w0, k1, E0)

def E_r_inf(kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d=0):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.E_r_inf(kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d)

def E_r_integrand(x, y, z, kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d=0):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.E_r_integrand(x, y, z, kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d)

def E_f_integrand(x, y, z, kx, ky, f, w0, k1, E0):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.E_f_integrand(x, y, z, kx, ky, f, w0, k1, E0)

def E_r(x, y, z, kx_min, kx_max, ky_min, ky_max, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d=0, integration_grid=501):
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    if not isinstance(z, Iterable):
        z = [z]
    if not isinstance(x, list):
        x = list(x)
    if not isinstance(y, list):
        y = list(y)
    if not isinstance(z, list):
        z = list(z)

    if integration_grid % 2 == 0:
        integration_grid += 1

    kx = np.linspace(kx_min, kx_max, integration_grid)
    ky = np.linspace(ky_min, ky_max, integration_grid)
    delta_kx = kx[1] - kx[0]
    delta_ky = ky[1] - ky[0]
    return cm.E_r(x, y, z, list(kx), list(ky), delta_kx, delta_ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d)

def E_f(x, y, z, kx_min, kx_max, ky_min, ky_max, f, w0, k1, E0, integration_grid=501):
    if not isinstance(x, Iterable):
        x = [x]
    if not isinstance(y, Iterable):
        y = [y]
    if not isinstance(z, Iterable):
        z = [z]
    if not isinstance(x, list):
        x = list(x)
    if not isinstance(y, list):
        y = list(y)
    if not isinstance(z, list):
        z = list(z)

    if integration_grid % 2 == 0:
        integration_grid += 1

    kx = np.linspace(kx_min, kx_max, integration_grid)
    ky = np.linspace(ky_min, ky_max, integration_grid)
    delta_kx = kx[1] - kx[0]
    delta_ky = ky[1] - ky[0]
    return cm.E_f(x, y, z, list(kx), list(ky), delta_kx, delta_ky, f, w0, k1, E0)


def r_s(kx, ky, k1, k2, mu1, mu2):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.r_s(kx, ky, k1, k2, mu1, mu2)

def r_p(kx, ky, k1, k2, eps1, eps2):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.r_p(kx, ky, k1, k2, eps1, eps2)

def t_s(kx, ky, k1, k2, mu1, mu2):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.t_s(kx, ky, k1, k2, mu1, mu2)

def t_p(kx, ky, k1, k2, mu1, mu2, eps1, eps2):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.t_p(kx, ky, k1, k2, mu1, mu2, eps1, eps2)

def r_s_membrane(kx, ky, k1, k2, mu1, mu2, d):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.r_s_membrane(kx, ky, k1, k2, mu1, mu2, d)

def r_p_membrane(kx, ky, k1, k2, eps1, eps2, d):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.r_p_membrane(kx, ky, k1, k2, eps1, eps2, d)

def t_s_membrane(kx, ky, k1, k2, mu1, mu2, d):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.t_s_membrane(kx, ky, k1, k2, mu1, mu2, d)

def t_p_membrane(kx, ky, k1, k2, mu1, mu2, eps1, eps2, d):
    if not isinstance(kx, Iterable):
        kx = [kx]
    if not isinstance(ky, Iterable):
        ky = [ky]
    if not isinstance(kx, list):
        kx = list(kx)
    if not isinstance(ky, list):
        ky = list(ky)

    return cm.t_p_membrane(kx, ky, k1, k2, mu1, mu2, eps1, eps2, d)


if __name__ == '__main__':
    kx = 1
    ky = 1
    f = 1
    w0 = 1
    k1 = 30
    k2 = 30
    E0 = 1
    z0 = 0
    mu1 = 1
    mu2 = 1
    eps1 = 1
    eps2 = 7.5
    d = 0.5e-6

    # print(E_inc(kx, ky, f, w0, k1, E0))

    # print(E_r_integrand(0,0,0, kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d))

    # print(E_r_inf(kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d))
    print(E_r([0], [0], [0], -5, 5, -5, 5, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d))

    # print(E_f_integrand(0,0,0, kx, ky, f, w0, k1, E0))
