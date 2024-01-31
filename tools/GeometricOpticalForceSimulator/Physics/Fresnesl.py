import numpy as np


def reflection_p(n_i, n_t, theta_i, theta_t):
    """
    Returns the reflection coefficient for p-polarized light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of reflected light
    """
    nominator = n_i * np.cos(theta_t) - n_t * np.cos(theta_i)
    denominator = n_i * np.cos(theta_t) + n_t * np.cos(theta_i)
    fraction = nominator / denominator
    return fraction * np.conjugate(fraction)


def transmission_p(n_i, n_t, theta_i, theta_t):
    """
    Returns the transmission coefficient for p-polarized light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of transmitted light
    """
    nominator = 4 * n_i * n_t * np.cos(theta_i) * np.cos(theta_t)
    denominator = n_i * np.cos(theta_t) + n_t * np.cos(theta_i)
    denominator *= np.conjugate(denominator)
    return nominator / denominator


def reflection_s(n_i, n_t, theta_i, theta_t):
    """
    Returns the reflection coefficient for s-polarized light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of reflected light
    """
    nominator = n_i * np.cos(theta_i) - n_t * np.cos(theta_t)
    denominator = n_i * np.cos(theta_i) + n_t * np.cos(theta_t)
    fraction = nominator / denominator
    return fraction * np.conjugate(fraction)


def transmission_s(n_i, n_t, theta_i, theta_t):
    """
    Returns the transmission coefficient for s-polarized light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of transmitted light
    """
    nominator = 4 * n_i * n_t * np.cos(theta_i) * np.cos(theta_t)
    denominator = n_i * np.cos(theta_i) + n_t * np.cos(theta_t)
    denominator *= np.conjugate(denominator)
    return nominator / denominator


def transmission(n_i, n_t, theta_i, theta_t):
    """
    Returns the transmission coefficient for un-polarized/circular light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of transmitted light
    """
    return (transmission_p(n_i, n_t, theta_i, theta_t) + transmission_s(n_i, n_t, theta_i, theta_t)) / 2


def reflection(n_i, n_t, theta_i, theta_t):
    """
    Returns the reflection coefficient for un-polarized/circular light
    :param n_i: refractive index on incident side of the interface
    :param n_t: refractive index on the other side of the interface
    :param theta_i: incident angle of the incoming light
    :param theta_t: light angle after transmission through the interface
    :return: fraction of reflected light
    """
    return (reflection_p(n_i, n_t, theta_i, theta_t) + reflection_s(n_i, n_t, theta_i, theta_t)) / 2
