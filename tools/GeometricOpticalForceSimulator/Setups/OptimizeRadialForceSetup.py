from Setups.BaseSetup import BaseSetup
from Physics import Simulator, Particle
from Physics.utils import find_force_equilibrium
from Geometry import Vector
import numpy as np


class OptimizeRadialForceSetup(BaseSetup):
    """
    The purpose of the simulation setup is to determine the power of the trapping laser in such a way that the
    radial force becomes as high as possible without the scattering force becoming too weak for trapping.
    """

    def __init__(self, simulator: Simulator, initial_power, min_power, power_decrement, displacement, gradient_displacement,
                 particle_radius=10.01e-6,
                 particle_refractive_index=1.59,
                 beam_power=1,
                 waist_position=Vector(),
                 beam_direction=Vector(1, 0, 0),
                 gradient_force_direction=Vector(0, 1, 0),
                 far_field=True,
                 diverging_near_field=False,
                 numerical_aperture=0.1,
                 data_path=None):
        super().__init__(simulator)
        self.particle = Particle(n=particle_refractive_index, radius=particle_radius, position=Vector())
        self.numerical_aperture = numerical_aperture
        self.particles = [self.particle]
        self.beam_power = beam_power
        self.beam_waist_position = waist_position
        self.beam_direction = beam_direction
        self.radial_force_direction = gradient_force_direction
        self.displacement = displacement
        self.radial_displacement = gradient_displacement
        self.initial_power = initial_power
        self.min_power = min_power
        self.power_decrement = power_decrement
        self.far_field = far_field
        self.diverging_near_field = diverging_near_field
        self.powers = list()
        self.trapping_positions = list()
        self.radial_force = list()
        self.data_path = data_path

    def run(self, show_plots=True, save_png=False, save_svg=False, save_path=None):
        self.powers = list()
        self.trapping_positions = list()
        self.radial_force = list()
        initial_position = self.particle.position.clone()   # store initial position of particle (also focus of trapping beam)
        powers = np.flip(np.arange(self.min_power, self.initial_power + self.power_decrement, self.power_decrement), axis=0)    # get set of powers to simulate
        optimization_finished = False
        i = 0

        # gravity force
        g = 9.81
        gravitational_force = self.particle.mass * g

        last_radial_force_maximum = 0     # store best result
        last_optimal_power = 0              # store optimized laser power

        print('starting optimization...')

        # sweep power settings
        while not optimization_finished and i < len(powers):
            print('***********************************************************')
            print('Finding equilibrium for laser power: {} mW'.format(powers[i]*1e3))

            self._simulator.beams.clear()   # remove all previous beams
            self.beam_power = powers[i]     # set laser power
            self.prepare()                  # create Gaussian beam in simulation space
            print('NA={}'.format(self._simulator.beam.numerical_aperture))

            # simulate the radial force along the beam
            displacements, _, _, axial_force = self.simulate_displacement(self.particle, self.beam_direction,
                                                                               self.displacement)
            # if the gravitational force exceeds the scattering force trapping cannot be realized -> stop optimization
            if np.max(axial_force) < gravitational_force:
                optimization_finished = True
                print('Scattering force too small!')
                break

            # get position where scattering force and gravity are equal (point where particle will sit)
            equilibrium = find_force_equilibrium(displacements, axial_force, gravitational_force)

            # position particle at equilibrium
            self.particle.position += self.beam_direction * equilibrium
            print('Equilibrium found: {} µm'.format(equilibrium*1e6))
            print('NA={}'.format(self._simulator.beam.numerical_aperture))

            # simulate radial force on particle at equilibrium
            print('Simulating radial force at equilibrium position...')
            _, radial_force, _, _ = self.simulate_displacement(self.particle, self.radial_force_direction,
                                                               self.radial_displacement)

            # get maximum radial force
            radial_force_maximum = np.max(radial_force)
            print('Radial force retrieved: {} pN'.format(radial_force_maximum*1e12))
            print('NA={}'.format(self._simulator.beam.numerical_aperture))

            # if the radial force exceeds the current best result store it as new candidate, otherwise the last result is the best
            if radial_force_maximum > last_radial_force_maximum:
                last_radial_force_maximum = radial_force_maximum
                last_optimal_power = powers[i]
                self.radial_force.append(radial_force_maximum)
                self.powers.append(powers[i])
                self.trapping_positions.append(equilibrium)
            """else:
                optimization_finished = True
                print('Radial force did not increase!')"""

            self.particle.position = initial_position.clone()   # reset particle position for next iteration
            i += 1
            print('***********************************************\n')

        print('\n*************************************')
        print('Optimization Finished')
        print('*************************************')
        print('gravitational force: {} pN'.format(gravitational_force*1e12))
        print('radial force: {} pN'.format(last_radial_force_maximum * 1e12))
        print('power optimum: {} mW'.format(last_optimal_power * 1e3))
        print('NA={}'.format(self._simulator.beam.numerical_aperture))

        if self.data_path is not None:
            data = np.vstack([self.powers, self.trapping_positions, self.radial_force]).T
            np.savetxt(self.data_path, data)
