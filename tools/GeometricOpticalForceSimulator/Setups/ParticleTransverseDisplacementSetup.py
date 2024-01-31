from Geometry import Vector, get_containing_orthonormal_basis
from Physics import Particle, Simulator
from Setups.BaseSetup import BaseSetup, plot
import numpy as np


class ParticleTransverseDisplacementSetup(BaseSetup):

    def __init__(self, simulator: Simulator, displacement,
                 particle_radius=10.01e-6,
                 particle_refractive_index=1.59,
                 beam_power=1,
                 waist_position=Vector(),
                 beam_direction=Vector(1, 0, 0),
                 far_field=False,
                 diverging_near_field=False):
        super().__init__(simulator)
        self.particle = Particle(n=particle_refractive_index, radius=particle_radius, position=Vector())
        self.particles = [self.particle]
        self.beam_power = beam_power
        self.beam_waist_position = waist_position
        self.beam_direction = beam_direction
        self.displacement = displacement
        self.far_field = far_field
        self.diverging_near_field = diverging_near_field

    def run(self, show_plots=True,
            save_png=False,
            save_svg=False,
            save_path=None,
            do_two_directions=False,
            show_axial=True,
            show_gradient_force=False,
            show_scattering_force=False,
            show_grid=False,
            save_csv=False):
        super().run()

        _, direction1, direction2 = get_containing_orthonormal_basis(self.beam_direction)

        displacements, radial_force1, _, axial_force, gradient_force, scattering_force = self.simulate_displacement(self.particle, direction1, self.displacement)

        if do_two_directions:
            displacements, _, radial_force2, _ = self.simulate_displacement(self.particle, direction2, self.displacement)

        save_path1, save_path2, save_path3, save_path4, save_path5 = save_path, save_path, save_path, save_path, save_path
        if save_path is not None:
            save_path1 += '_radial_1'
            save_path2 += '_radial_2'
            save_path3 += '_axial'
            save_path4 += '_gradient'
            save_path5 += '_scattering'

        plot(displacements*1e6, radial_force1*1e12, grid=show_grid, title='radial force along x', xlabel='displacement on y [µm]', ylabel='gradient force [pN]', show_plot=show_plots, save_png=save_png, save_svg=save_svg, save_path=save_path1)

        if do_two_directions:
            plot(displacements*1e6, radial_force2*1e12, grid=show_grid, title='radial force along y', xlabel='displacement on z [µm]', ylabel='gradient force [pN]', show_plot=show_plots, save_png=save_png, save_svg=save_svg, save_path=save_path2)

        if show_axial:
            plot(displacements*1e6, axial_force*1e12, grid=show_grid, title='axial force along z', xlabel='displacement on y [µm]', ylabel='scattering force [pN]', show_plot=show_plots, save_png=save_png, save_svg=save_svg, save_path=save_path3)

        if save_csv and save_path is not None:
            data1 = np.vstack([displacements, radial_force1])
            data3 = np.vstack([displacements, axial_force])
            np.savetxt(save_path1+'_data.csv', data1)
            np.savetxt(save_path3+'_data.csv', data3)

            np.savetxt(save_path4+'_data.csv', np.vstack([displacements, gradient_force]))
            np.savetxt(save_path5+'_data.csv', np.vstack([displacements, scattering_force]))

            if do_two_directions:
                data2 = np.vstack([displacements, radial_force2])
                np.savetxt(save_path2+'_data.csv', data2)
