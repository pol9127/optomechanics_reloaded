# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:08 2014

@author: Dominik Windey
"""
import cmath
import itertools
from enum import Enum
from math import inf
import matplotlib.pyplot as plt

import numpy as np
import optomechanics.theory.basics.physical_constants as pc
import optomechanics.theory.cavity_physics.math as math


class CavityType(Enum):
    """Class the different types the Cavity class can assume."""
    theoretical = 0
    experimental = 1


class Cavity:
    """Class representing a substrate based cavity.

    Note
    ----
    This class allows visualization of the cavity as well as calculation of relevant physical quantities.
    The class has to be initialized passing either reflectivity and cavity_length resulting in a theoretical cavity or
    by passing free_spectral_range and linewidth, resulting in an experimental cavity. The missing quantities are in
    both cases calculated from the given.

    Attributes
    ----------
    diameters : 2-float-array
        Diameter of left and right mirror
    curvatures: 2-float-array
        Curvature of left and right mirror
    reflectivity: float
        Reflectivity of mirrors in units of :1
    transmittance: float
        Transmittance of mirrors in units of :1
    cavity_length: float
        Length of cavity in units of m
    free_spectral_range: float
        Free spectral range of cavity in units of Hz
    linewidth: float
        Linewidth of cavity in units of Hz
    finesse: float
        Finesse of cavity
    absorbance: float
        Absorbance of mirrors in units of :1
    """
    def __init__(self, diameters_, curvatures_, reflectivity_=None, transmittance_=None, cavity_length_=None, free_spectral_range_=None, linewidth_=None):
        if(reflectivity_ is not None and cavity_length_ is not None):
            self.type = CavityType(0)
        elif(free_spectral_range_ is not None and linewidth_ is not None):
            self.type = CavityType(1)
        else:
            print('Either reflectivity and cavity_length or free_spectral_range and linewidth must be passed.')
            self.initialized = False
            return

        print('Cavity type is ' + str(self.type))
        self.initialized = True

        self.curvatures = curvatures_
        self.diameters = diameters_

        if self.type == CavityType(0):
            if not isinstance(cavity_length_, np.ndarray):
                if isinstance(cavity_length_, list):
                    self.cavity_length = np.array(cavity_length_)
                else:
                    self.cavity_length = np.array([cavity_length_])
            else:
                self.cavity_length = cavity_length_
            self.reflectivity = reflectivity_
            self.transmittance = transmittance_
            self.free_spectral_range = np.pi*pc.speed_of_light / self.cavity_length
            self.finesse = np.pi * np.sqrt(self.reflectivity) / (1 - self.reflectivity)
            self.linewidth = self.free_spectral_range / self.finesse
        elif self.type == CavityType(1):
            self.free_spectral_range = free_spectral_range_
            self.linewidth = linewidth_
            self.cavity_length = np.array([np.pi * pc.speed_of_light / self.free_spectral_range])
            self.finesse = self.free_spectral_range / self.linewidth
            self.reflectivity = (np.pi**2 * self.linewidth**2 - np.pi * self.linewidth
                                 * np.sqrt(np.pi**2 * self.linewidth**2 + 4 * self.free_spectral_range**2)
                                 + 2 * self.free_spectral_range**2) / (2 * self.free_spectral_range**2)

        if(self.reflectivity is not None and self.transmittance is not None):
            self.absorbance = 1 - reflectivity_ - transmittance_

    def plot_transfer_function_reflectivity(self, omega=None, lamda=None):
        if len(self.cavity.cavity_length) != 1:
            print('WARNING: Multiple cavity lengths passed. Using the first one for plotting')
        if omega is not None:
            y = math.transfer_function_reflectivity(self.reflectivity, self.transmittance, self.cavity_length[0],
                                                    omega=omega)
            x = omega
        elif lamda is not None:
            y = math.transfer_function_reflectivity(self.reflectivity, self.transmittance, self.cavity_length[0],
                                                    lamda=lamda)
            x = lamda
        else:
            return

        plt.figure()
        plt.plot(x, y)
        plt.show()

    def plot_transfer_function_transmittance(self, omega=None, lamda=None):
        if len(self.cavity.cavity_length) != 1:
            print('WARNING: Multiple cavity lengths passed. Using the first one for plotting')
        if omega is not None:
            y = math.transfer_function_transmittance(self.reflectivity, self.transmittance, self.cavity_length[0],
                                                     omega=omega)
            x = omega
        elif lamda is not None:
            y = math.transfer_function_transmittance(self.reflectivity, self.transmittance, self.cavity_length[0],
                                                     lamda=lamda)
            x = lamda
        else:
            return

        plt.figure()
        plt.plot(x, y)
        plt.show()

class LaserCavity:
    def __init__(self, cavity_, wavelength_, diameter_):
        self.wavelength = wavelength_
        self.diameter = diameter_
        self.rayleigh_length = math.rayleigh_length(self.diameter / 2, self.wavelength)
        self.cavity = cavity_

        self.w0 = np.array([math.minimum_beam_waist(l, self.cavity.curvatures, self.wavelength)
                            for l in self.cavity.cavity_length])
        self.w = np.array([math.beam_waists(l, self.cavity.curvatures, self.wavelength)
                           for l in self.cavity.cavity_length]).T
        self.focal_length, self.focal_point = map(list, zip(*[math.focal_length(self.diameter, w0, self.wavelength,
                                                                                self.rayleigh_length)
                                                              for w0 in self.w0]))
        self.minimum_waist_position = math.miniumum_waist_distances(self.cavity.cavity_length[0],
                                                                    self.cavity.curvatures)

    def plot_beam_waists(self, savefig=False):
        plt.figure()
        plt.xlabel(r'cavity_physics length [m]')
        plt.ylabel(r'waist size [$\mu$m]')
        plt.plot(self.cavity.cavity_length, self.w0 * 10**6, label=r'$\omega_0$')
        plt.plot(self.cavity.cavity_length, self.w[0] * 10**6, label=r'$\omega_1$')
        plt.plot(self.cavity.cavity_length, self.w[1] * 10**6, label=r'$\omega_2$')
        plt.grid()
        plt.legend(loc='best')
        if savefig:
            plt.savefig('beam_waists.png', dpi=300)
        plt.show()

    def plot_mode_matching_lense(self, savefig=False):
        plt.figure()
        plt.xlabel(r'cavity length [m]')
        plt.ylabel(r'distance [m]')
        plt.plot(self.cavity.cavity_length, self.focal_length, label=r'f')
        plt.plot(self.cavity.cavity_length, self.focal_point, label=r'z')
        plt.grid()
        plt.legend(loc='best')
        if savefig:
            plt.savefig('mode_matching_lense.png', dpi=300)
        plt.show()

    def plot_waists_and_mode_matching(self, savefig=False):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ax1.set_xlabel(r'cavity length [m]')
        ax1.set_ylabel(r'waist size [$\mu$m]')
        ax1.plot(self.cavity.cavity_length, self.w0 * 10**6, label=r'$\omega_0$')
        ax1.plot(self.cavity.cavity_length, self.w[0] * 10**6, label=r'$\omega_1$')
        ax1.plot(self.cavity.cavity_length, self.w[1] * 10**6, label=r'$\omega_2$')

        ax2.set_ylabel(r'distance [m]')
        ax2.plot(self.cavity.cavity_length, self.focal_length, label=r'f')
        ax2.plot(self.cavity.cavity_length, self.focal_point, label=r'z')

        ax1.grid()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.tight_layout()
        if savefig:
            plt.savefig('waists_and_lense.png', dpi=300)
        plt.show()

    def plot_cavity(self, savefig=False):
        if len(self.cavity.cavity_length) != 1:
            print('WARNING: Multiple cavity lengths passed. Using the first one for plotting')
        right_lense = math.circle_segment(self.cavity.curvatures[1], self.cavity.diameters[1],
                                          self.minimum_waist_position[1], self.w[0,0])(100)
        left_lense = math.circle_segment(self.cavity.curvatures[0], self.cavity.diameters[0],
                                         self.minimum_waist_position[0], self.w[1,0])(100)
        x = np.linspace(self.minimum_waist_position[0], self.minimum_waist_position[1], 100)
        beam = math.gaussian_beam(self.w0[0], self.wavelength)(x)
        plt.figure(figsize=(10, 10))
        plt.plot(right_lense[0], right_lense[1], color='k')
        plt.plot(left_lense[0], left_lense[1], color='k')
        plt.plot(x, beam, 'r--')
        plt.plot(x, -1*beam, 'r--')
        plt.xlim(np.min(left_lense[0]) - 0.1 * (np.max(right_lense[0]) - np.min(left_lense[0])),
                 np.max(right_lense[0]) + 0.1 * (np.max(right_lense[0]) - np.min(left_lense[0])))
        plt.xlabel('m')
        plt.ylabel('m')
        plt.grid()
        if savefig:
            plt.savefig('cavity.png',dpi=300)
        plt.show()


    def plot_transfer_function_reflectivity(self, savefig=False):
        y = math.transfer_function_reflectivity(self.cavity.reflectivity, self.cavity.transmittance,
                                                self.cavity.cavity_length, lamda=self.wavelength)

        fig, ax1 = plt.subplots()
        ax2 = plt.twinx(ax1)
        ax1.plot(self.cavity.cavity_length * 10**9, abs(y)**2, 'b-')
        ax2.plot(self.cavity.cavity_length * 10**9, [cmath.phase(y0) for y0 in y], 'r-')

        ax1.set_ylabel(r'$|F_{\mathrm{ref}}|^2$', color='b', fontsize=20)
        ax2.set_ylabel(r'arg($F_{\mathrm{ref}}$)', color='r', fontsize=20)
        ax1.set_xlabel(r'cavity length [nm]')
        ax1.set_ylim(0, 1.2)
        plt.grid()
        plt.tight_layout()
        if savefig:
            plt.savefig('reflectivity.png',dpi=300)
        plt.show()


    def plot_transfer_function_transmittance(self, savefig=False):
        y = math.transfer_function_transmittance(self.cavity.reflectivity, self.cavity.transmittance,
                                                 self.cavity.cavity_length, lamda=self.wavelength)

        fig, ax1 = plt.subplots()
        ax2 = plt.twinx(ax1)
        ax1.plot(self.cavity.cavity_length * 10**9, abs(y)**2, 'b-')
        ax2.plot(self.cavity.cavity_length * 10**9, [cmath.phase(y0) for y0 in y], 'r-')

        ax1.set_ylabel(r'$|F_{trans}|^2$', color='b')
        ax2.set_ylabel(r'arg($F_{trans}$)', color='r')
        ax1.set_xlabel(r'cavity length [nm]')
        ax1.set_ylim(0, 1.2)
        plt.grid()
        plt.tight_layout()
        if savefig:
            plt.savefig('transmittance.png',dpi=300)
        plt.show()

if __name__ == '__main__':

    cavity_length = np.linspace(4999, 5001, 10000)*10**-6
    # cavity_length = np.array([5])*10**-3
    mirror_diameters = np.array([12.7, 12.7])*10**-3
    mirror_curvatures = np.array([inf, 10])*10**-3
    mirror_reflectivity = 0.99986
    mirror_transmittance = 1 - mirror_reflectivity
    laser_wavelength = 1550*10**-9
    laser_diameter = 2.4*10**-3

    cavity = Cavity(mirror_diameters, mirror_curvatures, reflectivity_=mirror_reflectivity,
                    transmittance_=mirror_transmittance, cavity_length_=cavity_length)


    laser_macro_cavity = LaserCavity(cavity, laser_wavelength, laser_diameter)
    laser_macro_cavity.plot_cavity()
    laser_macro_cavity.plot_waists_and_mode_matching()
    laser_macro_cavity.plot_transfer_function_reflectivity()
    laser_macro_cavity.plot_transfer_function_transmittance()
    plt.figure()
    plt.plot(cavity.cavity_length, cavity.linewidth)
    plt.grid()
    plt.figure()
    plt.plot(cavity.cavity_length, cavity.free_spectral_range)
    plt.grid()
    plt.show()