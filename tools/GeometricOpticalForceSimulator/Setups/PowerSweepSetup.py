from Geometry import Vector
from Physics import Particle, Simulator
from Physics.utils import find_force_equilibrium
from Setups.BaseSetup import BaseSetup, plot
import numpy as np


class PowerSweepSetup(BaseSetup):
    def __init__(self, simulator: Simulator, displacement,
                 particle_radius=10.01e-6,
                 particle_refractive_index=1.59,
                 beam_powers=[500e-3, 1],
                 waist_position=Vector(),
                 beam_direction=Vector(0, 0, 1),
                 numerical_aperture=0.1,
                 n_diagonal_simulation_beams=30,
                 far_field=True,
                 diverging_near_field=True):
        super().__init__(simulator)
        self.numerical_aperture = numerical_aperture
        self.particle = Particle(n=particle_refractive_index, radius=particle_radius, position=Vector())
        self.particles = [self.particle]
        self.beam_powers = beam_powers
        self.n_diagonal_simulation_beams = n_diagonal_simulation_beams
        self.beam_waist_position = waist_position
        self.beam_direction = beam_direction
        self.displacement = displacement
        self.far_field = far_field
        self.diverging_near_field = diverging_near_field

    def run(self, show_plots=True, show_grid=False, save_png=False, save_svg=False, save_path=None, save_csv=False):
        super().run()

        if self._simulator.beam.rayleigh_range < 400e-6:
            self.far_field = True
            print('switching to far field')
        else:
            self.far_field = False
            self.diverging_near_field = True
            print('switching to near field')

        equilibrium_positions = list()
        used_powers = list()
        g = 9.81

        for power in self.beam_powers:
            print('***********************************************************')
            print('Finding equilibrium for laser power: {} mW'.format(power * 1e3))
            self._simulator.beams.clear()
            self.beam_power = power
            self.prepare()

            print('Rayleigh-length: {} µm'.format(self._simulator.beam.rayleigh_range*1e6))

            displacements, _, _, axial_force = self.simulate_displacement(self.particle, self.beam_direction, self.displacement)

            gravity = self.particle.mass * g

            if np.max(axial_force) < gravity:
                print('Axial force too small!')
                break

            z_eq = find_force_equilibrium(displacements, axial_force, gravity)
            equilibrium_positions.append(z_eq)
            used_powers.append(power)

            print('Equilibrium found: {} µm'.format(z_eq * 1e6))

        equilibrium_positions = np.array(equilibrium_positions)
        used_powers = np.array(used_powers)
        if save_csv and save_path is not None:
            data = np.vstack([used_powers, equilibrium_positions]).T
            np.savetxt(save_path + '_data.csv', data)

        if show_plots:
            plot(used_powers*1e6, equilibrium_positions,
                 xlabel='laser power [mW]',
                 ylabel='particle position [µm]',
                 save_png=save_png,
                 save_svg=save_svg)

