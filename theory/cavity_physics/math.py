# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:08 2014

@author: Dominik Windey
"""

import numpy as np
import optomechanics.theory.basics.physical_constants as pc

def beam_waists(length, curvature, lamda):
    g = 1 - length / curvature
    w = np.sqrt((length * lamda) / np.pi * np.sqrt(g[::-1] / (g * (1 - np.prod(g)))))
    return w


def minimum_beam_waist(length, curvature, lamda):
    g = 1 - length / curvature
    w0 = np.sqrt((length * lamda) / np.pi * np.sqrt((np.prod(g) * (1 - np.prod(g))) / (np.sum(g) - 2 * np.prod(g)) ** 2))
    return w0


def miniumum_waist_distances(length, curvature):
    g = 1 - length / curvature
    z = (g[::-1] * (1 - g) / (np.sum(g) - 2 * np.prod(g))) * length
    return z


def focal_length(d, w0, lamda, rayleigh_length_):
    f = rayleigh_length_ / np.sqrt(((d / 2) / w0) ** 2 - 1)
    z = f / (1 + (f / rayleigh_length_) ** 2)
    return f, z


def rayleigh_length(w0, lamda, n=1):
    return (n*np.pi*w0**2)/lamda


def gaussian_beam(w0, lamda, n=1):
    rl = rayleigh_length(w0, lamda, n)
    def beam(z):
        return w0 * np.sqrt(1 + (z**2) / (rl**2))
    return beam


def circle_segment(curvature, s, offset, w):
    if s > 2 * curvature:
        l = 2 * curvature
    else:
        l = s
    phi = np.arcsin(l / (2 * curvature))
    if phi != 0:
        offset -= np.sqrt(curvature ** 2 - w ** 2)

        def segment(n):
            n_phi = np.linspace(-1 * phi, phi, n)
            if s > (2 * curvature):
                bordersegments = [np.array([[curvature * np.cos(phi) + offset], [-1 * s / 2]]),
                                  np.array([[np.cos(phi) + offset], [s / 2]])]
                return np.hstack((bordersegments[0],
                                  curvature * np.array([np.cos(n_phi), np.sin(n_phi)]) + np.array([[offset], [0]]),
                                  bordersegments[1]))
            else:
                return curvature * np.array([np.cos(n_phi), np.sin(n_phi)]) + np.array([[offset], [0]])
        return segment

    else:

        def line(n):
            return np.array([offset * np.ones(n), np.linspace(-1*s/2, s/2, n)])

        return line


def transfer_function_reflectivity(reflectivity, transmittance, cavity_length, omega=None, lamda=None):
    """Function returning the reflection transfer function of the cavity for a given frequency or wavelength"""
    if (omega is None and lamda is None):
        return
    elif omega is None:
        omega = 2 * np.pi * pc.speed_of_light / lamda

    exponent = 2 * 1j * omega * cavity_length / pc.speed_of_light
    return (transmittance * np.exp(exponent) / (reflectivity * np.exp(exponent) - 1) + 1) * np.sqrt(reflectivity)


def transfer_function_transmittance(reflectivity, transmittance, cavity_length, omega=None, lamda=None):
    """Function returning the transmittance transfer function of the cavity for a given frequency or wavelength"""
    if (omega is None and lamda is None):
        return
    elif omega is None:
        omega = 2 * np.pi * pc.speed_of_light / lamda

    exponent = 1j * omega * cavity_length / pc.speed_of_light
    return transmittance * np.exp(exponent) / (reflectivity * np.exp(2 * exponent) - 1)

def mechanical_trap_frequency(permittivity, intensity, density, waist):
    epsilon = 3 * (permittivity - 1) / (permittivity + 2)
    return np.sqrt(4 * epsilon * intensity / (density * pc.speed_of_light * waist**2))