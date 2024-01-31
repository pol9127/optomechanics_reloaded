# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:08 2014

@author: Dominik Windey
"""
import numpy as np

import theory.cavity_physics.math as math


class NanoParticle:
    def __init__(self, material_, radius_):
        self.material = material_
        self.radius = radius_
        self.volume = 4/3 * np.pi * self.radius**3


class DipoleTrap:
    def __init__(self, intensity_, waist_, particle_):
        self.intensity = intensity_
        self.w = waist_
        self.particle = particle_
        self.mechanical_trap_frequency = math.mechanical_trap_frequency(self.particle.material.permittivity,
                                                                        self.intensity,
                                                                        self.particle.material.density,
                                                                        self.w)