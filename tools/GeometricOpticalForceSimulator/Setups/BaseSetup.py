from Geometry import Vector
from Physics import Particle, Medium, Simulator
import matplotlib.pyplot as plt


class BaseSetup(object):
    """
    This base setup can be inherited by other setups to be able to design simulations more efficiently
    Setups always use 3D Gaussian beams.
    """

    def __init__(self, simulator: Simulator):
        self._simulator = simulator
        self._prepared = False
        self.particles = list()
        self.beam_waist_position = Vector()
        self.beam_direction = Vector()
        self.beam_wavelength = 1565e-9
        self.beam_power = 1
        self.numerical_aperture = 0.1
        self.reflection_limit = 6
        self.n_diagonal_simulation_beams = 30
        self.medium = Medium()
        self.far_field = False
        self.diverging_near_field = False

    def prepare(self):
        """
        Prepare simulation domain according to the settings of the setup
        """
        self._simulator.medium = self.medium
        self._simulator.particles = self.particles
        self._simulator.setup_beam_gaussian_3d(self.beam_waist_position,
                                               self.beam_direction,
                                               self.beam_wavelength,
                                               self.beam_power,
                                               self.numerical_aperture,
                                               self.n_diagonal_simulation_beams,
                                               self.reflection_limit,
                                               self.far_field,
                                               self.diverging_near_field)
        self._prepared = True

    def simulate(self):
        """
        Run the simulation of light matter interactions
        """
        self._simulator.simulate(verbose=False)

    def simulate_displacement(self, particle: Particle, axis: Vector, displacement):
        """
        Simulate the displacement of a particle
        :param particle: particle to displace
        :param axis: axis to displace the particle on
        :param displacement: list of displacement values
        :return: displacements, gradient forces, scattering force
        """
        return self._simulator.simulate_displacement(particle=particle, axis=axis, displacements=displacement)

    def run(self, show_plots=True, save_png=False, save_svg=False, save_path=None):
        """
        Method for running the setup (should be implemented in every specific setup)
        :param show_plots: if true show plots
        :param save_png: if true save png image
        :param save_svg: if true save svg image
        :param save_path: filename (without extension) where plots and other data should be saved to
        """
        if not self._prepared:
            self.prepare()


def plot(*args, xlabel=None, ylabel=None, title=None, save_png=False, save_svg=False, save_path=None, show_plot=True, **kwargs):
        _invoke_matplot_(plt.plot, *args, xlabel=xlabel, ylabel=ylabel, title=title, save_png=save_png, save_svg=save_svg, save_path=save_path, show_plot=show_plot, **kwargs)


def smilogx(*args, xlabel=None, ylabel=None, title=None, save_png=False, save_svg=False, save_path=None, show_plot=True, **kwargs):
        _invoke_matplot_(plt.semilogx, *args, xlabel=xlabel, ylabel=ylabel, title=title, save_png=save_png, save_svg=save_svg, save_path=save_path, show_plot=show_plot, **kwargs)


def smilogy(*args, xlabel=None, ylabel=None, title=None, save_png=False, save_svg=False, save_path=None, show_plot=True, **kwargs):
        _invoke_matplot_(plt.semilogx, *args, xlabel=xlabel, ylabel=ylabel, title=title, save_png=save_png, save_svg=save_svg, save_path=save_path, show_plot=show_plot, **kwargs)


def _invoke_matplot_(matplot_function, *args, grid=False, xlabel=None, ylabel=None, title=None, save_png=False, save_svg=False, save_path=None, show_plot=True, **kwargs):
    figure = plt.figure()
    matplot_function(*args, **kwargs)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    plt.grid(grid)

    if save_path is not None:
        if save_png:
            figure.savefig(save_path + '.png')
        if save_svg:
            figure.savefig(save_path + '.svg')

    if show_plot:
        figure.show()

