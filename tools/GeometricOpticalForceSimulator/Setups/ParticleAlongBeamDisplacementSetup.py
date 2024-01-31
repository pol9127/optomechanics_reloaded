from Geometry import Vector
from Physics import Particle, Simulator
from Setups.BaseSetup import BaseSetup, plot
import numpy as np


class ParticleAlongBeamDisplacementSetup(BaseSetup):

    def __init__(self, simulator: Simulator, displacement,
                 particle_radius=10.01e-6,
                 particle_refractive_index=1.59,
                 beam_power=1,
                 numerical_aperture=0.1,
                 waist_position=Vector(),
                 beam_direction=Vector(1, 0, 0),
                 far_field=True,
                 diverging_near_field=False):
        super().__init__(simulator)
        self.particle = Particle(n=particle_refractive_index, radius=particle_radius, position=Vector())
        self.numerical_aperture = numerical_aperture
        self.particles = [self.particle]
        self.beam_power = beam_power
        self.beam_waist_position = waist_position
        self.beam_direction = beam_direction
        self.displacement = displacement
        self.far_field = far_field
        self.diverging_near_field = diverging_near_field

    def run(self, show_plots=True, show_grid=False, save_png=False, save_svg=False, save_path=None, save_csv=False):
        super().run()

        displacements, _, _, axial_force = self.simulate_displacement(self.particle, self.beam_direction, self.displacement)

        plot(displacements*1e6, axial_force*1e12, grid=show_grid, title='axial force given displacement', xlabel='displacement on x [µm]', ylabel='axial force [pN]', show_plot=show_plots, save_png=save_png, save_svg=save_svg, save_path=save_path)

        if save_csv and save_path is not None:
            data = np.vstack([displacements, axial_force])
            np.savetxt(save_path+'_data.csv', data)
