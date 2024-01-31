""" This module simulates the stochastic motion of levitated particles under the influence of translational and rotational
acceleration. This module is a joint collaboration of several members of the nanophotonics group.

To develop code I recommend the following software:
 - PyCharm is my favourite IDE and integrates nicely with git
 - Git is used to synchronize code from out gitlab repository (Please push your updates at the end of the day)
 - Jupyterlab is running on sisyphous and will be used in the end to run the code developed in PyCharm on 30 cores with
   250GB Ram

 - Python. Sisyphous and my private computer is currently running Python3.7. In principle I don't see any problems with
   using other Python versions (Python3.x), although I'd strongly recommend to also use Python3.7

 - For integration of stochastic differential equation we use the library sdeint. You will have to install it if you
   plan to run code locally.

Style Rules:
 - Class Names: First letter capital and then camelCase.
 - Functions and variables: Always lower case and underscores instead of camelCase.
 - Two rows separation between classes
 - One row separation between member functions
 - All calculations in SI-units (also pressure)
 - All input frequencies are WITHOUT a prefactor of 2pi
"""


# Imports from standard libraries
import numpy as np
import sympy as sy
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import constants as con
from collections.abc import Iterable
import os
import pandas as pd

# Imports for non standard libraries
import sdeint
from optomechanics.theory.particle_simulator.physics import Equations

class LevitatedParticle:
    """ This class represents a levitated particle. All necessary material properties should be stored here. """

    # Here we initialize all relevant member variables as None or empty lists/dicts
    # mass = None         # kg
    # density = None      # kg/m^3
    # radii = {}          # {x: m, y: m, z: m}
    # frequencies = {}    # {x: Hz, y: Hz, z: Hz}
    # dampings = {}       # {x: rad/s, y: rad/s, z: rad/s}
    # shape = None
    axes = ['x', 'y', 'z']
    angles = ['alpha', 'beta', 'gamma']
    physics = None
    numeric_values = None
    parameters = None

    def __init__(self, **kwargs):
        """ INPUT:
                Here we expect keyword arguments describing the system. To get a list of possible keywords. Initialize
                a LevitatedParticle without a keyword and print the member variable parameters or take a look at the
                file parameters.json.
            OUTPUT:
                Returns Parameters if no parameter is specified.
            DESCRP:
                The function stores the input variables as member variables.
        """
        if not kwargs:
            self.print_parameters()
        else:
            # Here we define default values that will be the same for most experiment on levitated Silica particles.
            numeric_values = {
                'c_0': con.c,
                'eps_0': con.epsilon_0,
                'N_A': con.N_A,
                'k_B': con.k,
                'eps_m': 1,         # in vacuum
                'eps_p': 1.44 ** 2, # at 1550e-9 nm wavelength
                'M': 28.97e-3,      # kg/mol (Hebestreit thesis, p14)
                'eta': 18.27e-6,    # Pa * s (Hebestreit thesis, p14)
                'rho': 2200         # density of Silica
                }

            for key, value in kwargs.items():
                numeric_values[key] = value

            self.physics = Equations(numeric_values)
            self.parameters = pd.Series(self.physics.parameter_descriptor)
            self.physics.find_numeric_values()
            self.numeric_values = pd.Series(self.physics.numeric_values)

    def print_parameters(self):
            print('You did not specify any parameters. I assume you don"t know which paramters exist. '
                  'Here is an exhaustive list:')
            self.physics = Equations()
            self.parameters = pd.Series(self.physics.parameter_descriptor)
            print(self.parameters)


    def check_parameters(self, parameters):
        values = []
        for param in parameters:
            if param not in self.physics.numeric_values:
                sys.exit('The following parameter was neither defined in LevitatedParticle nor could be calculated: ' + param)
            else:
                values.append(self.physics.numeric_values[param])
        return values

    def _fix_input(self, labels, data):
        if len(labels) != len(data):
            sys.exit('Number of labels does not match number of data to store. {0} != {1}'.format(len(labels), len(data)))

        data = np.array(data)
        labels = np.array(labels)
        is_None_msk = np.array([datum is None for datum in data])

        if all(is_None_msk):
            container = None
            return container
        elif np.sum(is_None_msk) == len(data) - 1:
            data[is_None_msk] = data[np.invert(is_None_msk)]
        else:
            new_val = data[np.invert(is_None_msk)][0]
            print('WARNING: No value for {0} was specified. I assume {1}'.format(labels[is_None_msk], new_val))
            data[is_None_msk] = new_val
        return data


class EquationOfMotion:
    """ This class provides a collection of equations of motions that involve a levitated particle. """
    levitated_particle = None
    f = None
    g = None
    axes = None
    angles = None
    underlying_problem = None
    drive = None
    q = None
    v = None
    a = None
    t = None

    def __init__(self, levitated_particle):
        """ INPUT:
                levitated_particle: Object of the LevitatedParticle class that provides material parameters of the
                studied particle.
        """
        self.levitated_particle = levitated_particle

    def init_harmonic_oscillator(self, axes=['x', 'y']):
        """ INPUT:
                axis: A string indicating along which particle axis the equation of motion should be calculated.
                      Possible values are: 'x', 'y', 'z'.
            OUTPUT: ndarray containing the fundamental matrix of the equation of motion
        """

        # If only one axis is desired and given as a string we put the string into a list of length one to make the
        # rest of the code compatible
        if isinstance(axes, str):
            axes = [axes]

        # Since other functions need to axes the desired axes we store it as a member variable
        self.axes = axes

        parameters = {'x': 'Omega_x0',
                      'y': 'Omega_y0',
                      'z': 'Omega_z0'}
        frequency = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])

        # We define sympy variables necessary to define the equation of motion
        # we make position, velocity, acceleration and time accessible as member variables so we can define functions
        # based on them outside of this function.
        # Particle oscillation frequencies
        om_0 = [sy.Symbol('\Omega_{0}'.format(axis), positive=True) for axis in axes]
        # spatial paramters
        self.q = [sy.Symbol('{0}'.format(axis), real=True) for axis in axes]
        # velocities
        self.v = [sy.Symbol('v_{0}'.format(axis), real=True) for axis in axes]
        # accelerations
        self.a = [sy.Symbol('\ddot{{{0}}}'.format(axis), real=True) for axis in axes]
        # time
        self.t = sy.Symbol('t', real=True)

        # We formulate the symbolic equation of motion
        self.underlying_problem = dict()
        self.underlying_problem['formula'] = [a_ + om_ ** 2 * q_ for a_, om_, q_ in zip(self.a, om_0, self.q)]
        self.underlying_problem['formula_numeric'] = []

        # We insert numerical parameters into the equation of motion
        for formula, om_, freq_ in zip(self.underlying_problem['formula'], om_0, frequency):
            self.underlying_problem['formula_numeric'].append(formula.subs(om_, freq_))

        self._extract_f()
        self._extract_g()

    def init_damped_harmonic_oscillator(self, axes=['x', 'y']):
        """ INPUT:
                axis: A string indicating along which particle axis the equation of motion should be calculated.
                      Possible values are: 'x', 'y', 'z'.
            OUTPUT: ndarray containing the fundamental matrix of the equation of motion
        """

        # If only one axis is desired and given as a string we put the string into a list of length one to make the
        # rest of the code compatible
        if isinstance(axes, str):
            axes = [axes]

        # Since other functions need to axes the desired axes we store it as a member variable
        self.axes = axes

        # We define sympy variables necessary to define the equation of motion
        # we make position, velocity, acceleration and time accessible as member variables so we can define functions
        # based on them outside of this function.
        # Particle oscillation frequencies
        parameters = {'x': 'Omega_x0',
                      'y': 'Omega_y0',
                      'z': 'Omega_z0'}
        frequency = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])
        om_0 = [sy.Symbol('\Omega_{0}'.format(axis), positive=True) for axis in axes]

        # Particle dampings
        parameters = {'x': 'gamma_x_gas',
                      'y': 'gamma_y_gas',
                      'z': 'gamma_z_gas'}
        damping = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])
        gamma_0 = [sy.Symbol('\Gamma_{0}'.format(axis), positive=True) for axis in axes]
        # spatial parameters
        self.q = [sy.Symbol('{0}'.format(axis), real=True) for axis in axes]
        # velocities
        self.v = [sy.Symbol('v_{0}'.format(axis), real=True) for axis in axes]
        # accelerations
        self.a = [sy.Symbol('\ddot{{{0}}}'.format(axis), real=True) for axis in axes]
        # time
        self.t = sy.Symbol('t', real=True)

        # We formulate the symbolic equation of motion
        self.underlying_problem = dict()
        zipped = zip(self.a, gamma_0, self.v, om_0, self.q)
        self.underlying_problem['formula'] = [a_ + gamma_ * v_ + om_ ** 2 * q_ for a_, gamma_, v_, om_, q_ in zipped]
        self.underlying_problem['formula_numeric'] = []

        # We insert numerical parameters into the equation of motion
        for formula, gamma_, gamma_N, om_, freq_ in zip(self.underlying_problem['formula'], gamma_0, damping, om_0,
                                                        frequency):
            self.underlying_problem['formula_numeric'].append(
                formula.subs([(om_, freq_), (gamma_, gamma_N)]))

        self._extract_f()
        self._extract_g()

    def init_damped_duffing_oscillator(self, axes=['x', 'y']):
        """ INPUT:
                axis: A string indicating along which particle axis the equation of motion should be calculated.
                      Possible values are: 'x', 'y', 'z'.
            OUTPUT: ndarray containing the fundamental matrix of the equation of motion
        """

        # If only one axis is desired and given as a string we put the string into a list of length one to make the
        # rest of the code compatible
        if isinstance(axes, str):
            axes = [axes]

        # Since other functions need to axes the desired axes we store it as a member variable
        self.axes = axes

        # We define sympy variables necessary to define the equation of motion
        # we make position, velocity, acceleration and time accessible as member variables so we can define functions
        # based on them outside of this function.
        # Particle oscillation frequencies
        parameters = {'x': 'Omega_x0',
                      'y': 'Omega_y0',
                      'z': 'Omega_z0'}
        frequency = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])

        om_0 = [sy.Symbol('\Omega_{0}'.format(axis), positive=True) for axis in axes]

        # Particle dampings
        parameters = {'x': 'gamma_x_gas',
                      'y': 'gamma_y_gas',
                      'z': 'gamma_z_gas'}
        damping = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])
        gamma_0 = [sy.Symbol('\Gamma_{0}'.format(axis), positive=True) for axis in axes]

        # Duffing coefficients
        parameters = {'x': 'xi_x',
                      'y': 'xi_y',
                      'z': 'xi_z'}
        duffing_parameters = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])
        csi = np.vstack([duffing_parameters for i in range(len(self.axes))])
        if len(self.axes) == 3:
            csi[2, 0] *= 2
            csi[2, 1] *= 2

        duffing_tensor = [[sy.Symbol('\\xi_{{{0}{1}}}'.format(axis_1, axis_2)) for axis_2 in self.axes] for axis_1 in self.axes]
        # spatial parameters
        self.q = [sy.Symbol('{0}'.format(axis), real=True) for axis in axes]
        # velocities
        self.v = [sy.Symbol('v_{0}'.format(axis), real=True) for axis in axes]
        # accelerations
        self.a = [sy.Symbol('\ddot{{{0}}}'.format(axis), real=True) for axis in axes]
        # time
        self.t = sy.Symbol('t', real=True)

        # We formulate the symbolic equation of motion
        self.underlying_problem = dict()
        distortion = []
        for current_mode in range(len(self.axes)):
            correction = 1
            for other_mode in range(len(self.axes)):
                correction += duffing_tensor[current_mode][other_mode] * self.q[current_mode] * self.q[other_mode]
            distortion.append(correction)
        zipped = zip(self.a, gamma_0, self.v, om_0, self.q, distortion)
        self.underlying_problem['formula'] = [a_ + gamma_ * v_ + om_ ** 2 * duff_ * q_ for a_, gamma_, v_, om_, q_, duff_ in zipped]

        self.underlying_problem['formula_numeric'] = []

        # We insert numerical parameters into the equation of motion
        enumerated = enumerate(zip(self.underlying_problem['formula'], gamma_0, damping, om_0, frequency))
        for current_element, (formula, gamma_, gamma_N, om_, freq_) in enumerated:
            substitutions = [(om_, freq_), (gamma_, gamma_N)]
            for other_element in range(len(self.axes)):
                substitutions.append((duffing_tensor[current_element][other_element], csi[current_element][other_element]))
            self.underlying_problem['formula_numeric'].append(formula.subs(substitutions))

        self._extract_f()
        self._extract_g()

    def init_from_potential(self, potential, variables, parameters):
        self.axes = [ax for ax in self.levitated_particle.axes if ax in variables]
        self.underlying_problem = dict()
        m = self.levitated_particle.check_parameters(['m'])[0]

        # variables
        self.q = [variables[key] for key in self.axes]
        # velocities
        v_labels = ['v_{0}'.format(variable) for variable in self.q]
        self.v = []
        for v_label in v_labels:
            if v_label in variables:
                self.v.append(variables[v_label])
            else:
                 self.v.append(sy.Symbol(v_label, real=True))

        # self.v = [sy.Symbol('v_{0}'.format(variable), real=True) for variable in self.q]
        # accelerations
        self.a = [sy.Symbol('\ddot{{{0}}}'.format(variable), real=True) for variable in self.q]
        # time
        self.t = sy.Symbol('t', real=True)

        self.underlying_problem['formula'] = [a + sy.diff(potential, q) / m for q, a in zip(self.q, self.a)]
        self.underlying_problem['formula_numeric'] = [f.subs(parameters) for f in self.underlying_problem['formula']]

        self._extract_f()
        self._extract_g()

    def init_drive(self, amplitude_x=0, amplitude_y=0, amplitude_z=0, frequency_x=0,
                   frequency_y=0, frequency_z=0, phase_x=0, phase_y=0, phase_z=0):
        """INPUT: amplitude_i: The Force amplitude of a periodic drive in direction i in N
                  frequency_i: The angular frequency of a periodic drive in direction i
                  phase_i: The phase of a periodic drive in direction i
        """
        # Before specifying the driving that is applied we need to define the bare underlying problem.
        if self.underlying_problem is None:
            sys.exit('Please initialize the underlying model before initializing the drive.')

        # Here we collect the parameters from the function input in lists
        amplitude = []
        frequency = []
        phase = []
        m = self.levitated_particle.check_parameters(['m'])[0]
        for axis, amp, freq, ph in zip(self.levitated_particle.axes,
                                       [amplitude_x / m,
                                        amplitude_y / m,
                                        amplitude_z / m],
                                       [frequency_x, frequency_y, frequency_z],
                                       [phase_x, phase_y, phase_z]):
            if axis in self.axes:
                amplitude.append(amp)
                frequency.append(freq)
                phase.append(ph)

        # Now we define symbolic expressions for the drive
        self.drive = {}
        amplitude_sy = [sy.Symbol('F_{0}'.format(axis), real=True) for axis in self.axes]
        frequency_sy = [sy.Symbol('\omega_{0}'.format(axis), positive=True) for axis in self.axes]
        phase_sy = [sy.Symbol('\phi_{0}'.format(axis), real=True) for axis in self.axes]
        self.drive['formula'] = [d_a * sy.sin(d_f * self.t + d_p) for d_a, d_f, d_p in zip(amplitude_sy, frequency_sy, phase_sy)]

        zipped = zip(self.drive['formula'], amplitude, frequency, phase, amplitude_sy, frequency_sy, phase_sy)

        # We replace symbols for which numeric expressions were given
        self.drive['formula_numeric'] = [formula.subs([(amp_sy, amp), (freq_sy, freq), (ph_sy, ph)])
                                         for formula, amp, freq, ph, amp_sy, freq_sy, ph_sy in zipped]

        # Now we reextract f since the equation of motion has changed. Since the stochastic part does not depend on the
        # drive we do not need to reextract g.
        self._extract_f()

    def _extract_f(self):
        """This functions extracts the f function which is necessary for solving the Ito equation."""
        # The first entries of the f function are simply all velocities given.
        f_ = self.v.copy()
        # If a drive is specified we modify the resulting equation by subtracting the drive since it is on the
        # opposite side of the equation

        if self.drive is None:
            formulae = self.underlying_problem['formula_numeric']
        else:
            formulae = [form_0 - form_1 for form_0, form_1 in zip(self.underlying_problem['formula_numeric'],
                                                                  self.drive['formula_numeric'])]

        # We define a parameter zeta which is equal to the acceleration. We then solve for zeta and obtain the rest of
        # the entries of f
        for formula, a_ in zip(formulae, self.a):
            zeta = sy.Symbol('\zeta', real=True)
            formula = formula.subs(a_, zeta)
            zeta_ = sy.solve(formula, zeta)[0]
            f_.append(zeta_)

        # For speed reasons we want f as a numpy function. In a last step we wrap the function to fit the requirements
        # of sdeint.
        f_np = sy.lambdify(self.q + self.v + [self.t], f_, 'numpy')
        self.f = lambda x, t: np.array(f_np(*x, t))

    def _extract_g(self):
        """This functions extracts the G function which is necessary for solving the Ito equation."""

        parameters = {'x': 'gamma_x_gas',
                      'y': 'gamma_y_gas',
                      'z': 'gamma_z_gas'}

        gammas = self.levitated_particle.check_parameters([parameters[axis] for axis in self.axes])

        m, k, T = self.levitated_particle.check_parameters(['m', 'k_B', 'T'])
        k_t = k * T
        diffusivities = [np.sqrt(2 * k_t / m * gamma) for gamma in gammas]
        position_sigma = np.diag(np.zeros(len(self.axes)))
        velocity_sigma = np.diag(diffusivities)
        self.g = lambda x, t: np.concatenate((position_sigma, velocity_sigma))


class InitialValueProblem:
    """ This class solves initial value problems given an equation of motion."""

    eq_of_motion = None
    initial_values = None
    timestep = None
    timespan = None

    def __init__(self, equation_of_motion, initial_values=[0, 0], timestep=1, timespan=100):
        """ At initialization we pass matrices that describe the deterministic or stochastic motion of the particle.
        I suppose depending on the problem it might be better to pass functions f and G directly. I am currently not sure
        how to construct these matrices and functions for a given equation of motion."""
        self.eq_of_motion = equation_of_motion
        self.initial_values = [float(i) for i in initial_values]
        self.timestep = timestep
        self.timespan = timespan

    @property
    def runtime(self):
        """Class that creates an array containing the runtime of the timetrace from timespan and timesteps."""
        return np.arange(0, self.timespan, self.timestep)

    def solve(self):
        """Solve Ito equation: dx = f(x,t) dt + G(x,t) dW """

        # Here we seed the random number generator which will be used during the calculation of ito integrals
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

        # Use the sdeint library to simulate the particle motion
        timetrace = sdeint.itoint(self.eq_of_motion.f, self.eq_of_motion.g, self.initial_values, self.runtime)
        return timetrace


if __name__ == '__main__':
    """Here parts of the implementation can be tested. This part is only executed if the this file is called directly.
    So if the module is imported elsewhere (e.g. in a Jupyter Notebook) this part is ignored."""
    levitated_particle_ = LevitatedParticle(rho=2200,
                                            r_x=70e-9,
                                            r_y=70e-9,
                                            r_z=70e-9,
                                            P=400e-3,
                                            p_gas=1e2,
                                            lamda=1550e-9,
                                            w_x0=1e-6,
                                            w_y0=1e-6,
                                            T=300)
    eqn = EquationOfMotion(levitated_particle_)
    eqn.init_damped_duffing_oscillator(['x', 'y'])
    #
    # x, y = sy.symbols('x y', real=True)
    # kx, ky = sy.symbols('k_x k_y', positive=True)
    #
    # pot = 0.5 * kx * x ** 2 + 0.5 * ky * y ** 2
    # eqn.init_from_potential(pot, {'x': x,'y': y}, {kx: 4e-5, ky: 3e-5})
    #
    # # eqn.init_damped_duffing_oscillator(axes=['x'])
    # # # eqn.init_drive(amplitude_x=1e-12, frequency_x=10e3, phase_x=0)
    # # eqn2 = EquationOfMotion(levitated_particle_)
    # # eqn2.init_damped_harmonic_oscillator(axes=['x'])
    # #
    initial_value_problem = InitialValueProblem(eqn, initial_values=[0e-7, 0, 0, 0], timestep=1e-8, timespan=1e-3)
    # # initial_value_problem2 = InitialValueProblem(eqn2, initial_values=[1e-7, 0], timestep=1e-8, timespan=1e-3)
    # #
    timetrace_ = initial_value_problem.solve()
    # # timetrace2_ = initial_value_problem2.solve()
    # #
    # # # Here we plot the position of the harmonic oscillator over 1ms.
    fig, ax = plt.subplots()
    ax.plot(initial_value_problem.runtime * 1e3, timetrace_[::, 1] * 1e9)
    # ax.plot(initial_value_problem2.runtime * 1e3, timetrace2_[::, 0])
    # # ax.plot(initial_value_problem.runtime * 1e3, timetrace_[::, 1])
    plt.show()
