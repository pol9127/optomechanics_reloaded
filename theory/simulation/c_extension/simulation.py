import numpy as np
from optomechanics.theory.simulation.c_extension._custom_module import *
import scipy as sp

class Engine:
    _parameters = {'width_x' : None,
                   'width_y' : None,
                   'wavelength' : None,
                   'radius' : None,
                   'power' : None,
                   'damping_rate' : None,
                   'rho' : None,
                   'mass' : None,
                   'temperature' : None,
                   'rayleigh_len' : None,
                   'volume' : None,
                   'radius' : None,
                   'permittivity_particle' : None,
                   'permittivity_medium' : None,
                   'e_field' : None,
                   'power' : None,
                   'focal_distance' : None,
                   'NA' : None,
                   'filling_factor' : None,
                   'width_inc' : None,
                   'aperture_radius' : None,
                   'jones_vector' : None,
                   'delta' : None}

    def __init__(self, parameters=None, **kwargs):
        if parameters is not None:
            self.parameters = parameters
        if kwargs:
            self.parameters = kwargs

    def start_condition(self, weight=None, x_sign=None, v_sign=None):
        if weight is None:
            weight = np.random.rand(3)
        if x_sign is None:
            x_sign = np.random.choice([1, -1])
        if v_sign is None:
            v_sign = np.random.choice([1, -1])

        if isinstance(weight, list):
            weight = np.array(weight)

        if not isinstance(weight, np.ndarray):
            weight = np.array(weight, weight, weight)

        energy_kin = sp.constants.k * self.parameters['temperature'] * weight
        energy_pot = sp.constants.k * self.parameters['temperature'] * (1 - weight)

        omega = np.sqrt(self.trap_stiffness(self.parameters['radius'] * 3) / self.parameters['mass'])
        v0 = v_sign * np.sqrt(2 * energy_kin / self.parameters['mass'])
        x0 = x_sign * np.sqrt(2 * energy_pot / (self.parameters['mass'] * omega**2))
        return [*x0, *v0]

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, parameters=None):
        if parameters is not None:
            self._parameters.update(parameters)

    @property
    def rayleigh_length(self, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate rayleigh length!')
            return
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        return rayleigh_length(self.parameters['width_x'], self.parameters['width_y'], self.parameters['wavelength'])

    def trap_stiffness(self, amplitude_x, amplitude_y=None, amplitude_z=None):
        if amplitude_y is None:
            amplitude_y = amplitude_x
        if amplitude_z is None:
            amplitude_z = amplitude_x
        x = np.linspace(-amplitude_x, amplitude_x, 100)
        y = np.linspace(-amplitude_y, amplitude_y, 100)
        z = np.linspace(-amplitude_z, amplitude_z, 100)

        stiffness = [-1 * np.polyfit(x, self.total_force(x, 0, 0, calc_magnetic=False, dougnut=True)[..., 0].flatten(), 1)[0],
                     -1 * np.polyfit(y, self.total_force(0, y, 0, calc_magnetic=False, dougnut=True)[..., 1].flatten(), 1)[0],
                     -1 * np.polyfit(z, self.total_force(0, 0, z, calc_magnetic=False, dougnut=True)[..., 2].flatten(), 1)[0]]
        return np.array(stiffness)

    def width(self, z=None, axis=b'x', **kwargs):
        if kwargs:
            self.parameters = kwargs
        if z is None:
            print('Position where to calculate the beam width must be specified!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the beam width!')
            return
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        if isinstance(axis, str):
            axis = bytes(axis, 'utf8')
        return width(z, self.parameters['width_x'], self.parameters['width_y'], axis, self.parameters['wavelength'])

    def wavefront_radius(self, z=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if z is None:
            print('Position where to calculate the beam curvature must be specified!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the beam curvature!')
            return
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        return wavefront_radius(z, 0, self.parameters['width_x'], self.parameters['width_y'], self.parameters['wavelength'])

    def polarizability(self, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['volume'] is None and self.parameters['radius'] is None:
            print('Particle volume or radius must be specified to calculate polarizability!')
            return
        elif self.parameters['volume'] is None:
            return polarizability(0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                  self.parameters['permittivity_medium'])
        else:
            return polarizability(self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                  self.parameters['permittivity_medium'])

    def effective_polarizability(self, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['volume'] is None and self.parameters['radius'] is None:
            print('Particle volume or radius must be specified to calculate effective polarizability!')
            return
        if self.parameters['wavelength'] is None:
            print('Excitation wavelength must be specified to calculate effective polarizability!')
            return

        elif self.parameters['volume'] is None:
            return effective_polarizability(0, self.parameters['radius'], self.parameters['wavelength'], self.parameters['permittivity_particle'],
                                            self.parameters['permittivity_medium'])
        else:
            return effective_polarizability(self.parameters['volume'], 0, self.parameters['wavelength'], self.parameters['permittivity_particle'],
                                            self.parameters['permittivity_medium'])

    def intensity_gauss(self, x=None, y=None, z=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the beam intensity must be specified!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the beam intensity!')
            return
        if self.parameters['power'] is None and self.parameters['e_field'] is None:
            print('Either electric field or power in focus must be specified.')
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        if self.parameters['power'] is None:
            return intensity_gauss(x, y, z, self.parameters['width_x'], self.parameters['width_y'],
                                   self.parameters['e_field'], -1, self.parameters['wavelength'])
        else:
            return intensity_gauss(x, y, z, self.parameters['width_x'], self.parameters['width_y'], -1,
                                   self.parameters['power'], self.parameters['wavelength'])

    def strongly_focussed(self, x=None, y=None, z=None, calc_magnetic=False, dougnut=False, surface=None, **kwargs):
        vectorize = False

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0
        if dougnut:
            fields = fields_doughnut_rad
            fields_vect = fields_doughnut_rad_vect
        else:
            fields = fields_00
            fields_vect = fields_00_vect

        if kwargs:
            self.parameters = kwargs
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the beam intensity must be specified!')
            return
        if self.parameters['power'] is None:
            self.parameters['power'] = -1
            if self.parameters['e_field'] is None:
                print('Either electric field or power in focus and incoming beam width must be specified.')
                return
        elif self.parameters['e_field'] is None:
            self.parameters['e_field'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate electric field from power.')
                return
        if self.parameters['aperture_radius'] is None:
            self.parameters['aperture_radius'] = -1
            if self.parameters['filling_factor'] is None:
                print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                return
        elif self.parameters['filling_factor'] is None:
            self.parameters['filling_factor'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                return
        if self.parameters['width_inc'] is None:
            self.parameters['width_inc'] = -1
        if self.parameters['focal_distance'] is None:
            print('Focal Distance must be specified to calculated fields.')
            return
        if self.parameters['NA'] is None:
            print('Focussing NA must be specified to calculate fields.')
            return
        if self.parameters['wavelength'] is None:
            print('Laser Wavelength must be specified to calculate fields.')
            return
        if self.parameters['jones_vector'] is None:
            print('No Jones Vector specified, using standard ([1, 0]).')

        if isinstance(x, np.ndarray):
            x = list(x)
        if isinstance(y, np.ndarray):
            y = list(y)
        if isinstance(z, np.ndarray):
            z = list(z)

        if isinstance(x, list):
            if not isinstance(y, list):
                y = [y]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(y, list):
            x = [x]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(z, list):
            x = [x]
            y = [y]
            vectorize = True

        if vectorize:
            if calc_magnetic:
                fields = fields_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['e_field'], self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self.parameters['aperture_radius'],
                                   self.parameters['width_inc'], 2, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
                return fields[...,:3], fields[...,3:]
            else:
                return fields_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                 self.parameters['e_field'], self.parameters['power'], self.parameters['jones_vector'],
                                 self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                 self.parameters['filling_factor'], self.parameters['aperture_radius'], self.parameters['width_inc'],
                                   0, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
        else:
            if calc_magnetic:
                fields = fields(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['e_field'], self.parameters['power'],
                                   self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self.parameters['aperture_radius'],
                                   self.parameters['width_inc'], 2, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
                return fields[:3], fields[3:]
            else:
                return fields(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                 self.parameters['e_field'], self.parameters['power'], self.parameters['jones_vector'],
                                 self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                 self.parameters['filling_factor'], self.parameters['aperture_radius'],
                                 self.parameters['width_inc'], 0, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

    def gradient_force_gaussian(self, x=None, y=None, z=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the gradient force!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the gradient force!')
            return
        if self.parameters['power'] is None:
            print('Power in focus must be specified to calculate the gradient force.')
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        if self.parameters['radius'] is None:
            print('Particle radius must be specified to calculate the gradient force!')
            return
        if self.parameters['wavelength'] is None:
            print('Excitation Wavelength must be specified to calculate the gradient force!')
            return
        return gradient_force_gaussian(x, y, z, self.parameters['power'], self.parameters['width_x'],
                                       self.parameters['width_y'], 0, self.parameters['radius'],
                                       self.parameters['wavelength'], 0, self.parameters['permittivity_particle'],
                                       self.parameters['permittivity_medium'])

    def scattering_force_gaussian(self, x=None, y=None, z=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the gradient force!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the gradient force!')
            return
        if self.parameters['power'] is None:
            print('Power in focus must be specified to calculate the gradient force.')
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        if self.parameters['radius'] is None:
            print('Particle radius must be specified to calculate the gradient force!')
            return
        return scattering_force_gaussian(x, y, z, self.parameters['power'], self.parameters['width_x'],
                                         self.parameters['width_y'], 0, self.parameters['radius'],
                                         self.parameters['wavelength'], 0, self.parameters['permittivity_particle'],
                                         self.parameters['permittivity_medium'])

    def total_force_gaussian(self, x=None, y=None, z=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the gradient force!')
            return
        if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters['wavelength'] is None:
            print('Focal width and wavelength must be specified to calculate the gradient force!')
            return
        if self.parameters['power'] is None:
            print('Power in focus must be specified to calculate the gradient force.')
        elif self.parameters['width_x'] is None:
            self.parameters['width_x'] = self.parameters['width_y']
        elif self.parameters['width_y'] is None:
            self.parameters['width_y'] = self.parameters['width_x']
        if self.parameters['radius'] is None:
            print('Particle radius must be specified to calculate the gradient force!')
            return
        return total_force_gaussian(x, y, z, self.parameters['power'], self.parameters['width_x'],
                                    self.parameters['width_y'], 0, self.parameters['radius'],
                                    self.parameters['wavelength'], 0, self.parameters['permittivity_particle'],
                                    self.parameters['permittivity_medium'])

    def fluctuating_force(self, dt=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['temperature'] is None:
            print('Temperature not specified, using standard value 300K.')
            self.parameters = {'temperature' : 300}
        if self.parameters['damping_rate'] is None or self.parameters['mass'] is None:
            print('Damping rate and particle mass must be specified to calculate the fluctuating force!')
            return
        if dt is None:
            print('Timestep dt in which the fluctuations are accounted for must be specified.')
            return
        return fluctuating_force(self.parameters['damping_rate'], self.parameters['mass'], self.parameters['temperature'], dt, 3)

    def ode_runge_kutta(self, problem=None, y0=None, t=None, field_kind=0, surface=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['temperature'] is None:
            print('Temperature not specified, using standard value 300K.')
            self.parameters = {'temperature' : 300}
        if self.parameters['radius'] is None:
            print('Particle radius must be specified to simulate particle motion!')
            return
        if self.parameters['damping_rate'] is None or self.parameters['mass'] is None:
            print('Damping rate and particle mass must be specified to simulate particle motion!')
            return
        if problem is None or y0 is None or t is None:
            print('The kind of problem, initial values y0 and desired time steps t must be specified to simulate particle motion.')
            return

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0

        if isinstance(y0, np.ndarray):
            y0 = list(y0)
        if isinstance(t, np.ndarray):
            t = list(t)

        if problem == 'initial_value_optical_gaussian':
            if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters[
                'wavelength'] is None:
                print('Focal width and wavelength must be specified to simulate particle motion!')
                return
            if self.parameters['power'] is None:
                print('Power in focus must be specified to simulate particle motion.')
            elif self.parameters['width_x'] is None:
                self.parameters['width_x'] = self.parameters['width_y']
            elif self.parameters['width_y'] is None:
                self.parameters['width_y'] = self.parameters['width_x']

            return ode_runge_kutta(problem, y0, t, self.parameters['power'], self.parameters['width_x'],
                                   self.parameters['damping_rate'], self.parameters['mass'], self.parameters['width_y'],
                                   0, self.parameters['radius'], self.parameters['wavelength'],
                                   self.parameters['temperature'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], 0, 0, 0, [0, 0], 0, 0, 0, 0, 0, 0, 0,
                                   surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

        elif problem == 'initial_value_optical':
            if self.parameters['power'] is None:
                self.parameters['power'] = -1
                if self.parameters['e_field'] is None:
                    print('Either electric field or power in focus and incoming beam width must be specified.')
                    return
            elif self.parameters['e_field'] is None:
                self.parameters['e_field'] = -1
                if self.parameters['width_inc'] is None:
                    print('Incoming beam width must be specified to caculate electric field from power.')
                    return
            if self.parameters['aperture_radius'] is None:
                self.parameters['aperture_radius'] = -1
                if self.parameters['filling_factor'] is None:
                    print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                    return
            elif self.parameters['filling_factor'] is None:
                self.parameters['filling_factor'] = -1
                if self.parameters['width_inc'] is None:
                    print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                    return
            if self.parameters['width_inc'] is None:
                self.parameters['width_inc'] = -1
            if self.parameters['focal_distance'] is None:
                print('Focal Distance must be specified to calculated fields.')
                return
            if self.parameters['NA'] is None:
                print('Focussing NA must be specified to calculate fields.')
                return
            if self.parameters['wavelength'] is None:
                print('Laser Wavelength must be specified to calculate fields.')
                return
            if self.parameters['jones_vector'] is None:
                print('No Jones Vector specified, using standard ([1, 0]).')
            if self.parameters['delta'] is None:
                self.parameters['delta'] = 1e-10

            return ode_runge_kutta(problem, y0, t, self.parameters['power'], 0,
                                   self.parameters['damping_rate'], self.parameters['mass'], 0,
                                   0, self.parameters['radius'], self.parameters['wavelength'],
                                   self.parameters['temperature'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['e_field'], self.parameters['jones_vector'], self.parameters['n_1'],
                                   self.parameters['n_2'], self.parameters['filling_factor'], self.parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, self.parameters['delta'],
                                   surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

    def ode_euler(self, problem=None, y0=None, t=None, field_kind=0, surface=None, **kwargs):
        if kwargs:
            self.parameters = kwargs
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['temperature'] is None:
            print('Temperature not specified, using standard value 300K.')
            self.parameters = {'temperature' : 300}
        if self.parameters['radius'] is None:
            print('Particle radius must be specified to simulate particle motion!')
            return
        if self.parameters['damping_rate'] is None or self.parameters['mass'] is None:
            print('Damping rate and particle mass must be specified to simulate particle motion!')
            return
        if problem is None or y0 is None or t is None:
            print('The kind of problem, initial values y0 and desired time steps t must be specified to simulate particle motion.')
            return

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0


        if isinstance(y0, np.ndarray):
            y0 = list(y0)
        if isinstance(t, np.ndarray):
            t = list(t)

        if problem == 'initial_value_optical_gaussian':
            if self.parameters['width_x'] is None and self.parameters['width_y'] is None or self.parameters[
                'wavelength'] is None:
                print('Focal width and wavelength must be specified to simulate particle motion!')
                return
            if self.parameters['power'] is None:
                print('Power in focus must be specified to simulate particle motion.')
            elif self.parameters['width_x'] is None:
                self.parameters['width_x'] = self.parameters['width_y']
            elif self.parameters['width_y'] is None:
                self.parameters['width_y'] = self.parameters['width_x']

            return ode_euler(problem, y0, t, self.parameters['power'], self.parameters['width_x'],
                                   self.parameters['damping_rate'], self.parameters['mass'], self.parameters['width_y'],
                                   0, self.parameters['radius'], self.parameters['wavelength'],
                                   self.parameters['temperature'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], 0, 0, 0, [0, 0], 0, 0, 0, 0, 0,0, 0,
                             surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

        elif problem == 'initial_value_optical':
            if self.parameters['power'] is None:
                self.parameters['power'] = -1
                if self.parameters['e_field'] is None:
                    print('Either electric field or power in focus and incoming beam width must be specified.')
                    return
            elif self.parameters['e_field'] is None:
                self.parameters['e_field'] = -1
                if self.parameters['width_inc'] is None:
                    print('Incoming beam width must be specified to caculate electric field from power.')
                    return
            if self.parameters['aperture_radius'] is None:
                self.parameters['aperture_radius'] = -1
                if self.parameters['filling_factor'] is None:
                    print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                    return
            elif self.parameters['filling_factor'] is None:
                self.parameters['filling_factor'] = -1
                if self.parameters['width_inc'] is None:
                    print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                    return
            if self.parameters['width_inc'] is None:
                self.parameters['width_inc'] = -1
            if self.parameters['focal_distance'] is None:
                print('Focal Distance must be specified to calculated fields.')
                return
            if self.parameters['NA'] is None:
                print('Focussing NA must be specified to calculate fields.')
                return
            if self.parameters['wavelength'] is None:
                print('Laser Wavelength must be specified to calculate fields.')
                return
            if self.parameters['jones_vector'] is None:
                print('No Jones Vector specified, using standard ([1, 0]).')
            if self.parameters['delta'] is None:
                self.parameters['delta'] = 1e-10

            return ode_euler(problem, y0, t, self.parameters['power'], 0,
                                   self.parameters['damping_rate'], self.parameters['mass'], 0,
                                   0, self.parameters['radius'], self.parameters['wavelength'],
                                   self.parameters['temperature'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['e_field'], self.parameters['jones_vector'], self.parameters['n_1'],
                                   self.parameters['n_2'], self.parameters['filling_factor'], self.parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, self.parameters['delta'],
                             surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])


    def total_force(self, x=None, y=None, z=None, delta=1e-10, field_kind=0, surface=None, **kwargs):
        vectorize = False

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0

        if kwargs:
            self.parameters = kwargs
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the beam intensity must be specified!')
            return
        if self.parameters['power'] is None:
            self.parameters['power'] = -1
            if self.parameters['e_field'] is None:
                print('Either electric field or power in focus and incoming beam width must be specified.')
                return
        elif self.parameters['e_field'] is None:
            self.parameters['e_field'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate electric field from power.')
                return
        if self.parameters['aperture_radius'] is None:
            self.parameters['aperture_radius'] = -1
            if self.parameters['filling_factor'] is None:
                print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                return
        elif self.parameters['filling_factor'] is None:
            self.parameters['filling_factor'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                return
        if self.parameters['width_inc'] is None:
            self.parameters['width_inc'] = -1
        if self.parameters['focal_distance'] is None:
            print('Focal Distance must be specified to calculated fields.')
            return
        if self.parameters['NA'] is None:
            print('Focussing NA must be specified to calculate fields.')
            return
        if self.parameters['wavelength'] is None:
            print('Laser Wavelength must be specified to calculate fields.')
            return
        if self.parameters['jones_vector'] is None:
            print('No Jones Vector specified, using standard ([1, 0]).')
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['volume'] is None and self.parameters['radius'] is None:
            print('Particle volume or radius must be specified to calculate polarizability!')
            return

        if isinstance(x, np.ndarray):
            x = list(x)
        if isinstance(y, np.ndarray):
            y = list(y)
        if isinstance(z, np.ndarray):
            z = list(z)

        if isinstance(x, list):
            if not isinstance(y, list):
                y = [y]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(y, list):
            x = [x]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(z, list):
            x = [x]
            y = [y]
            vectorize = True

        if vectorize:
            if self.parameters['volume'] is None:
                return total_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return total_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

        else:
            if self.parameters['volume'] is None:
                return total_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return total_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])


    def gradient_force(self, x=None, y=None, z=None, delta=1e-10, field_kind=0, surface=None, **kwargs):
        vectorize = False

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0

        if kwargs:
            self.parameters = kwargs
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the beam intensity must be specified!')
            return
        if self.parameters['power'] is None:
            self.parameters['power'] = -1
            if self.parameters['e_field'] is None:
                print('Either electric field or power in focus and incoming beam width must be specified.')
                return
        elif self.parameters['e_field'] is None:
            self.parameters['e_field'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate electric field from power.')
                return
        if self.parameters['aperture_radius'] is None:
            self.parameters['aperture_radius'] = -1
            if self.parameters['filling_factor'] is None:
                print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                return
        elif self.parameters['filling_factor'] is None:
            self.parameters['filling_factor'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                return
        if self.parameters['width_inc'] is None:
            self.parameters['width_inc'] = -1
        if self.parameters['focal_distance'] is None:
            print('Focal Distance must be specified to calculated fields.')
            return
        if self.parameters['NA'] is None:
            print('Focussing NA must be specified to calculate fields.')
            return
        if self.parameters['wavelength'] is None:
            print('Laser Wavelength must be specified to calculate fields.')
            return
        if self.parameters['jones_vector'] is None:
            print('No Jones Vector specified, using standard ([1, 0]).')
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['volume'] is None and self.parameters['radius'] is None:
            print('Particle volume or radius must be specified to calculate polarizability!')
            return

        if isinstance(x, np.ndarray):
            x = list(x)
        if isinstance(y, np.ndarray):
            y = list(y)
        if isinstance(z, np.ndarray):
            z = list(z)

        if isinstance(x, list):
            if not isinstance(y, list):
                y = [y]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(y, list):
            x = [x]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(z, list):
            x = [x]
            y = [y]
            vectorize = True

        if vectorize:
            if self.parameters['volume'] is None:
                return gradient_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return gradient_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

        else:
            if self.parameters['volume'] is None:
                return gradient_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return gradient_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

    def scattering_force(self, x=None, y=None, z=None, delta=1e-10, field_kind=0, surface=None, **kwargs):
        vectorize = False

        if surface is not None:
            self.parameters['surface_distance'] = surface[0]
            self.parameters['surface_reflectance'] = surface[1]
            surface_exists = 1
        else:
            self.parameters['surface_distance'] = 0
            self.parameters['surface_reflectance'] = 0
            surface_exists = 0

        if kwargs:
            self.parameters = kwargs
        if z is None or x is None or y is None:
            print('Position (x, y, z) where to calculate the beam intensity must be specified!')
            return
        if self.parameters['power'] is None:
            self.parameters['power'] = -1
            if self.parameters['e_field'] is None:
                print('Either electric field or power in focus and incoming beam width must be specified.')
                return
        elif self.parameters['e_field'] is None:
            self.parameters['e_field'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate electric field from power.')
                return
        if self.parameters['aperture_radius'] is None:
            self.parameters['aperture_radius'] = -1
            if self.parameters['filling_factor'] is None:
                print('Either filling_factor or aperture_radius and incoming beam width must be specified.')
                return
        elif self.parameters['filling_factor'] is None:
            self.parameters['filling_factor'] = -1
            if self.parameters['width_inc'] is None:
                print('Incoming beam width must be specified to caculate filling_factor from aperture_radius.')
                return
        if self.parameters['width_inc'] is None:
            self.parameters['width_inc'] = -1
        if self.parameters['focal_distance'] is None:
            print('Focal Distance must be specified to calculated fields.')
            return
        if self.parameters['NA'] is None:
            print('Focussing NA must be specified to calculate fields.')
            return
        if self.parameters['wavelength'] is None:
            print('Laser Wavelength must be specified to calculate fields.')
            return
        if self.parameters['jones_vector'] is None:
            print('No Jones Vector specified, using standard ([1, 0]).')
        if self.parameters['permittivity_particle'] is None:
            print('Particle permittivity not specified, using standard value 2.101.')
            self.parameters = {'permittivity_particle' : 2.101}
        if self.parameters['permittivity_medium'] is None:
            print('Medium permittivity not specified, using standard value 1.')
            self.parameters = {'permittivity_medium': 1.}
        if self.parameters['volume'] is None and self.parameters['radius'] is None:
            print('Particle volume or radius must be specified to calculate polarizability!')
            return

        if isinstance(x, np.ndarray):
            x = list(x)
        if isinstance(y, np.ndarray):
            y = list(y)
        if isinstance(z, np.ndarray):
            z = list(z)

        if isinstance(x, list):
            if not isinstance(y, list):
                y = [y]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(y, list):
            x = [x]
            if not isinstance(z, list):
                z = [z]
            vectorize = True
        elif isinstance(z, list):
            x = [x]
            y = [y]
            vectorize = True

        if vectorize:
            if self.parameters['volume'] is None:
                return scattering_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return scattering_force_vect(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                        self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                        self.parameters['permittivity_medium'], self.parameters['e_field'],
                                        self.parameters['power'], self.parameters['jones_vector'],
                                        self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                        self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                        self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])

        else:
            if self.parameters['volume'] is None:
                return scattering_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   0, self.parameters['radius'], self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
            else:
                return scattering_force(x, y, z, self.parameters['focal_distance'], self.parameters['NA'],
                                   self.parameters['volume'], 0, self.parameters['permittivity_particle'],
                                   self.parameters['permittivity_medium'], self.parameters['e_field'],
                                   self.parameters['power'], self.parameters['jones_vector'],
                                   self.parameters['wavelength'], self.parameters['n_1'], self.parameters['n_2'],
                                   self.parameters['filling_factor'], self._parameters['aperture_radius'],
                                   self.parameters['width_inc'], field_kind, delta, surface_exists, self.parameters['surface_distance'], self.parameters['surface_reflectance'])
