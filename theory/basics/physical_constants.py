# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 11:32:08 2014

@author: Dominik Windey

Note: Module containing phyical constants for use in calculations.
"""

from enum import Enum

speed_of_light = 3*10**8


class FusedSilica:
    permittivity = 2
    density = 2203      # kg/m³


class Materials(Enum):
    fused_silica = FusedSilica

    def permittivity(self):
        return self.value.permittivity
    def density(self):
        return self.value.density

