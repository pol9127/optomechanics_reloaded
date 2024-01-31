import numpy as np
from Geometry import Vector, cross, get_containing_orthonormal_basis
from Physics import LightRay


class GaussianBeam(object):
    """
    Pure quantitative representation of a Gaussian beam
    """

    @property
    def beam_waist(self):
        """
        Gets the waist radius of the beam
        """
        return self._beam_waist

    @beam_waist.setter
    def beam_waist(self, value):
        """
        Sets the waist radius of the beam
        """
        self._beam_waist = value
        self._update_calculated_quantities_()

    @property
    def wavelength(self):
        """
        Sets the wavelength of the beam
        """
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        """
        Sets the wavelength of the beam
        """
        self._wavelength = value
        self._update_calculated_quantities_()

    @property
    def power(self):
        """
        Gets the power of the beam
        """
        return self._power

    @power.setter
    def power(self, value):
        """
        Sets the power of the beam
        """
        self._power = value
        self._update_calculated_quantities_()

    @property
    def rayleigh_range(self):
        """
        Gets the rayleigh range of the beam (changes when other parameters are changed)
        """
        return self._rayleigh_range

    @property
    def intensity(self):
        """
        Gets intensity amplitude of the beam
        """
        return self._intensity

    @property
    def numerical_aperture(self):
        """
        Gets the numerical aperture of the beam
        """
        return self.beam_waist / self.rayleigh_range

    def __init__(self, wavelength=1565e-9, beam_waist=10e-6, power=1):
        self._beam_waist = beam_waist
        self._wavelength = wavelength
        self._rayleigh_range = 0
        self._power = power
        self._intensity = 0
        self._update_calculated_quantities_()

    def waist(self, z):
        """
        Returns the waist radius at a certain position (at zero this is exactly the beam waist)
        :param z: position at which the waist should be taken
        """
        return self._beam_waist * np.sqrt(1 + (z / self._rayleigh_range)**2)

    def intensity_at(self, z, r):
        """
        Returns the intensity of the beam at a certain position (cylindrical coordinates)
        :param z: position on propagation axis
        :param r: offset from propagation axis
        :return:
        """
        w = self.waist(z)
        return self._intensity * (self._beam_waist / w)**2 * np.exp(-2*r**2/w**2)

    def _update_calculated_quantities_(self):
        """
        Calculates the dependent quantities of the beam
        """
        self._rayleigh_range = np.pi * self._beam_waist**2 / self._wavelength
        self._intensity = self._power / (np.pi * self._beam_waist**2)


def generate_gaussian_beam(waist_position: Vector, direction: Vector, wavelength=1565e-9, beam_waist=10e-6, power=1, diagonal_beam_count=30, reflection_limit=5, far_field=False, diverging_near_field=False):
    """
    Generates a set of light beams to represent the rayleigh range of a Gaussian beam for the simulation.
    Since the beams are straight a distinction between "pre, at and beyond focal plane" cases has to be made:

        case before focal plane:
            use far_field=False -> beams converge to fit beam waist at focal plane
        case at focal plane:
            use far_field=False -> beams converge to fit beam waist at focal plane
        case beyond focal plane:
            use far_field=True -> beams diverge stating at focal plane forming a circle in plane of radius equal to the beam waist

    CAUTION: Using far_field=False for beyond focal plane simulations will result in a positional error in propagation direction given by:

            pos_err =~ 2*beam_waist / (sin(wavelength / (pi*beam_waist)))

    :param waist_position: position of the focus
    :param direction: direction of the beam
    :param wavelength: wavelength of the beam
    :param beam_waist: waist radius at the focus
    :param power: power of the beam
    :param diagonal_beam_count: number of light beams that make up the diameter of the beam total number of beams n = n_d^2 pi/2
    :param reflection_limit: maximum number of reflections (for performance)
    :param far_field: This will use the NA calculated from the wavelength and the beam waist to generate a cone shaped beam accurate to the far field of the Gaussian beam
    :param diverging_near_field: If set to true the generated beams will diverge starting at the focus. Otherwise they converge towards the focus.
    :return: list of beams, logical gaussian beam
    """
    if far_field:
        return generate_gaussian_beam_far_field(waist_position, direction, wavelength, beam_waist, power, diagonal_beam_count, reflection_limit)
    else:
        if diverging_near_field:
            return generate_gaussian_beam_near_field_after_focus(waist_position, direction, wavelength, beam_waist, power, diagonal_beam_count, reflection_limit)
        else:
            return generate_gaussian_beam_near_field_before_focus(waist_position, direction, wavelength, beam_waist, power, diagonal_beam_count, reflection_limit)


def generate_gaussian_beam_far_field(waist_position: Vector, direction: Vector, wavelength=1565e-9, beam_waist=10e-6, power=1, diagonal_ray_count=30, reflection_limit=5):
    """
    Generate a set of light beams to represent the far field of a gaussian beam starting at the focus.
    :param waist_position: position of the focus
    :param direction: direction of the beam
    :param wavelength: wavelength of the beam
    :param beam_waist: waist radius at the focus
    :param power: power of the beam
    :param diagonal_ray_count: number of light rays that make up the diameter of the beam total number of rays n = n_d^2 pi/2
    :param reflection_limit: maximum number of reflections (for performance)
    :return: list of rays, logical gaussian beam
    """
    rays = list()
    gaussian_beam = GaussianBeam(wavelength, beam_waist, power)  # create Gaussian beam representation
    radius = gaussian_beam.beam_waist
    dx = 2 * radius / diagonal_ray_count  # get the discretization step of the Gaussian beam
    dA = dx ** 2  # get the area per ray

    base1, base2, base3 = get_containing_orthonormal_basis(direction)  # get the orthonormal unit vectors of the beams local coordinate system

    for i in range(diagonal_ray_count):
        x_coord = dx * i - radius
        for j in range(diagonal_ray_count):
            y_coord = dx * j - radius

            # check if the generated potential origin lies within the radius
            center_offset = np.sqrt(x_coord ** 2 + y_coord ** 2)
            if center_offset <= radius:
                ray_origin = waist_position
                ray_offset = base2 * x_coord + base3 * y_coord
                ray_target = ray_origin + gaussian_beam.rayleigh_range * direction + ray_offset
                ray_direction = ray_target - ray_origin  # get the beam direction
                intensity = gaussian_beam.intensity_at(gaussian_beam.rayleigh_range, center_offset)  # calculate the beam intensity given its radial position
                rays.append(LightRay(wavelength, intensity * dA, reflection_limit, origin=ray_origin, direction=ray_direction))  # add the ray to the list

    return rays, gaussian_beam


def generate_gaussian_beam_near_field_after_focus(waist_position: Vector, direction: Vector, wavelength=1565e-9, beam_waist=10e-6, power=1, diagonal_ray_count=30, reflection_limit=5):
    """
    Generates a set of light beams to represent the rayleigh range of a Gaussian beam for the simulation.
    The beam is meant to be use beyond the focal plane. -> beam collection diverging starting at focus
    :param waist_position: position of the focus
    :param direction: direction of the beam
    :param wavelength: wavelength of the beam
    :param beam_waist: waist radius at the focus
    :param power: power of the beam
    :param diagonal_ray_count: number of light rays that make up the diameter of the beam total number of rays n = n_d^2 pi/2
    :param reflection_limit: maximum number of reflections (for performance)
    :return: list of rays, logical gaussian beam
    """
    rays = list()
    gaussian_beam = GaussianBeam(wavelength, beam_waist, power)  # create Gaussian beam representation
    radius = gaussian_beam.waist(gaussian_beam.rayleigh_range)  # get the radius at the origin from where the beam is launched
    radius_fraction = radius / beam_waist   # get the scaling factor of the radius compared to the focus
    rayleigh_offset = direction * gaussian_beam.rayleigh_range  # get the offset from the focus to the origin
    dx = 2 * beam_waist / diagonal_ray_count  # get the discretization step of the Gaussian beam
    dA = dx ** 2  # get the area per beam

    base1, base2, base3 = get_containing_orthonormal_basis(direction)  # get the orthonormal unit vectors of the beams local coordinate system

    for i in range(diagonal_ray_count):
        x_coord = dx * i - beam_waist
        for j in range(diagonal_ray_count):
            y_coord = dx * j - beam_waist

            # check if the generated potential origin lies within the radius
            center_offset = np.sqrt(x_coord ** 2 + y_coord ** 2)
            if center_offset < beam_waist:
                ray_offset = base2 * x_coord + base3 * y_coord
                ray_origin = ray_offset + waist_position  # set the ray origin
                radial_direction = -cross(direction, cross(direction, ray_offset))  # get the vector pointing radially outward from the propagation axis
                radial_direction.normalize()
                directional_target = radial_direction * (radius_fraction * center_offset) + rayleigh_offset  # set the target the beam is propagating towards in the focal plane
                ray_direction = directional_target - ray_origin  # get the ray
                # direction
                intensity = gaussian_beam.intensity_at(0, center_offset)  # calculate the beam intensity given its radial position
                rays.append(LightRay(wavelength, intensity * dA, reflection_limit, origin=ray_origin, direction=ray_direction))  # add the ray to the list

    return rays, gaussian_beam


def generate_gaussian_beam_near_field_before_focus(waist_position: Vector, direction: Vector, wavelength=1565e-9, beam_waist=10e-6, power=1, diagonal_ray_count=30, reflection_limit=5):
    """
    Generates a set of light beams to represent the rayleigh range of a Gaussian beam for the simulation.
    The beam is meant to be use either before or at the focal plane of the beam. -> beam collection converging towards focus
    :param waist_position: position of the focus
    :param direction: direction of the beam
    :param wavelength: wavelength of the beam
    :param beam_waist: waist radius at the focus
    :param power: power of the beam
    :param diagonal_ray_count: number of light rays that make up the diameter of the beam total number of rays n = n_d^2 pi/2
    :param reflection_limit: maximum number of reflections (for performance)
    :return: list of rays, logical gaussian beam
    """
    rays = list()
    gaussian_beam = GaussianBeam(wavelength, beam_waist, power)     # create Gaussian beam representation
    radius = gaussian_beam.waist(-gaussian_beam.rayleigh_range)     # get the radius at the origin from where the beam is launched
    radius_fraction = beam_waist / radius  # get the scaling factor of the radius compared to the focus
    rayleigh_offset = -direction * gaussian_beam.rayleigh_range     # get the offset from the focus to the origin
    dx = 2 * radius / diagonal_ray_count   # get the discretization step of the Gaussian beam
    dA = dx**2  # get the area per beam

    base1, base2, base3 = get_containing_orthonormal_basis(direction)   # get the orthonormal unit vectors of the beams local coordinate system

    for i in range(diagonal_ray_count):
        x_coord = dx * i - radius
        for j in range(diagonal_ray_count):
            y_coord = dx * j - radius

            # check if the generated potential origin lies within the radius
            center_offset = np.sqrt(x_coord**2 + y_coord**2)
            if center_offset < radius:
                ray_origin = base2 * x_coord + base3 * y_coord + waist_position + rayleigh_offset  # set the ray origin to be at -rayleigh_range
                ray_offset = waist_position - ray_origin  # get the offset vector to the center of the focus
                radial_direction = cross(direction, cross(direction, ray_offset))  # get the vector pointing radially outward from the propagation axis
                radial_direction.normalize()
                directional_target = radial_direction * (radius_fraction * center_offset) + waist_position  # set the target the ray is propagating towards in the focal plane
                ray_direction = directional_target - ray_origin   # get the rays direction
                intensity = gaussian_beam.intensity_at(-gaussian_beam.rayleigh_range, center_offset)    # calculate the beam intensity given its radial position
                rays.append(LightRay(wavelength, intensity * dA, reflection_limit, origin=ray_origin, direction=ray_direction))     # add the ray to the list

    return rays, gaussian_beam


if __name__ == '__main__':
    wavelength = 1565e-9
    numerical_aperture = 0.037
    beam_waist = wavelength / (numerical_aperture * np.pi)
    """beams, beam = generate_gaussian_beam(Vector(), Vector(0, 0, 1), wavelength, beam_waist, power=1, diagonal_beam_count=100, reflection_limit=6, far_field=False, diverging_near_field=False)
    print(len(beams))"""
    beam = GaussianBeam(wavelength, beam_waist, 1)
    print(beam.numerical_aperture)
