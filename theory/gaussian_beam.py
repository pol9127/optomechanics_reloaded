from numpy import sqrt, inf, arctan2, pi, exp, isnan, divide, \
    isscalar, conjugate, arcsin, sin, cos, array, real, imag, \
    vectorize, moveaxis

from scipy.constants import c, epsilon_0

from scipy.integrate import quad
from scipy.special import jn

import warnings


def rayleigh_length(width_x, width_y=None, wavelength=1064e-9):
    if width_y is None:
        width_y = width_x

    return pi * width_x * width_y / wavelength


def width(z, width_x, width_y=None, axis='x', wavelength=1064e-9,
          rayleigh_len=None):
    if width_y is None:
        width_y = width_x
        width_0 = width_x
    else:
        if axis in ('x', 'X'):
            width_0 = width_x
        elif axis in ('y', 'Y'):
            width_0 = width_y
        else:
            raise ValueError("Axis has to be either 'x' or 'y'.")

    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    return width_0 * sqrt(1 + (z / rayleigh_len) ** 2)


def wavefront_radius(z, rayleigh_len=None, width_x=None,
                     width_y=None, wavelength=1064e-9):
    if rayleigh_len is None:
        if width_x is None:
            raise ValueError("Either rayleigh_length or width_x have "
                             "to be specified.")
        if width_y is None:
            width_y = width_x

        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    # If we want to suppress the RuntimeWarning for 0 input:
    # while warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    radius = z * (1 + divide(rayleigh_len, z)**2)

    if isnan(radius).any():
        if isscalar(radius):
            radius = inf
        else:
            radius[isnan(radius)] = inf

    return radius


def gauss_phase(x, y, z, width_x, width_y=None, wavelength=1064e-9,
                rayleigh_len=None):
    if width_y is None:
        width_y = width_x

    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    # wave number
    k = 2 * pi / wavelength

    # wavefront radius
    rad = wavefront_radius(z, rayleigh_len)

    # Gouy phase correction
    eta = arctan2(z, rayleigh_len)

    return k * z - eta + k * (x ** 2 + y ** 2) / (2 * rad)


def electric_field_gauss(x, y, z, width_x, width_y=None, e_field=None,
                         power=None, wavelength=1064e-9,
                         rayleigh_len=None):
    if width_y is None:
        width_y = width_x

    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    if e_field is None:
        assert power is not None, 'Either E0 or power have to be ' \
                                  'specified.'
        e_field = sqrt(4 * power / (c * epsilon_0 * pi * width_x *
                                    width_y))

    exponent = - (x ** 2 / width(z, width_x, width_y, 'x',
                                 wavelength, rayleigh_len) ** 2 +
                  y ** 2 / width(z, width_x, width_y, 'y',
                                 wavelength, rayleigh_len) ** 2) + \
        1j * gauss_phase(x, y, z, width_x, width_y, wavelength,
                         rayleigh_len)

    return e_field * exp(exponent) / sqrt(1 + z ** 2 /
                                          (rayleigh_len ** 2))


def intensity_gauss(x, y, z, width_x, width_y=None, e_field=None,
                    power=None, wavelength=1064e-9, rayleigh_len=None):
    if width_y is None:
        width_y = width_x

    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    if power is None:
        assert e_field is not None, 'Either E0 or power have to be ' \
                               'specified.'
        power = c * epsilon_0 * pi * width_x * width_y * e_field * \
            conjugate(e_field) / 4

    width2 = width(z, width_x, width_y, 'x', wavelength,
                   rayleigh_len) * width(z, width_x, width_y, 'y',
                                         wavelength, rayleigh_len)

    exponent = -2 * (x ** 2 / width(z, width_x, width_y, 'x',
                                    wavelength, rayleigh_len) ** 2 +
                     y ** 2 / width(z, width_x, width_y, 'y',
                                    wavelength, rayleigh_len) ** 2)

    return power / (pi * width2 / 2) * exp(exponent)


def focus_beam(waist_0, waist_0_distance=0, waist_1=None, focal_length=None, wavelength=1064e-9):
    """Calculate parameters of a Gaussian beam when passing through a focusing lens. Either desired waist or
    focal length must be specified. If not a TypeError is raised.

    Input
    ----------
    waist_0 : minimal waist of Gaussian beam before traversing lens.
    waist_0_distance : distance to waist minimum. If the beam is collimated: 0
    waist_1 : desired waist after focusing. (Either waist_1 or focal_length must be specified)
    focal_length : focal length of lens used for focusing. (Either waist_1 or focal_length must be specified)
    wavelength : wavelength of input beam

    Output
    ----------
    Tuple of either (focal_length, waist_1_distance) or (waist_1, waist_1_distance).
    focal_length : required focal length for the desired focusing
    waist_1 : waist of Gaussian beam after focusing
    waist_1_distance : distance from focusing lens to focused beam.
    """
    if waist_1 is None and focal_length is None:
        raise TypeError('Either desired beam waist or lens\' focal length must be specified')
    elif waist_1 is not None and focal_length is not None:
        raise TypeError('Only desired beam waist or lens\' focal length must be specified, not both')
    elif waist_1 is not None:
        rayleigh_len = rayleigh_length(waist_0, wavelength=wavelength)
        focal_length = (rayleigh_len**2 + waist_0_distance**2) \
                       / (waist_0_distance + sqrt((waist_0 / waist_1)**2 * (rayleigh_len**2 + waist_0_distance**2)
                                                  - rayleigh_len**2))
        waist_distance = focal_length * (1 + (waist_0_distance / focal_length - 1)
                                         / ((waist_0_distance / focal_length - 1)**2 + (rayleigh_len / focal_length)**2))
        return focal_length, waist_distance
    elif focal_length is not None:
        rayleigh_len = rayleigh_length(waist_0, wavelength=wavelength)
        waist_1 = waist_0 / sqrt((1 - waist_0_distance / focal_length)**2 + (rayleigh_len / focal_length)**2)
        waist_distance = focal_length * (1 + (waist_0_distance / focal_length - 1)
                                         / ((waist_0_distance / focal_length - 1)**2 + (rayleigh_len / focal_length)**2))
        return waist_1, waist_distance

def magnify_beam(focal_length_1, focal_length_2, waist_0=None, wavelength=None):
    """Calculate the magnification factor of two lenses for a collimated input and output beam.

    Input
    ----------
    focal_length_1 : focal length of first lens (if negative biconcave if positive biconvex)
    focal_length_2 : focal length of second lens (should be positive)

    waist_0 : minimal waist of Gaussian beam before traversing lens (if not provided geometrical optics approximation is used)
    wavelength : wavelength of input beam (if not provided geometrical optics approximation is used)

    Output
    ----------
    Tuple of (magnification factor, distance between lenses).
    magnification : magnification obtained by setup
    distance : distance between telescope lenses
    """
    if waist_0 is None or wavelength is None:
        return focal_length_2 / abs(focal_length_1)
    z0 = rayleigh_length(waist_0, waist_0, wavelength)
    if focal_length_1 < 0:
        d = focal_length_2 + focal_length_1
    else:
        d = focal_length_2 + 2 * focal_length_1
    return focal_length_2 / z0 * sqrt(1 + z0**2 / focal_length_1**2), d


def strongly_focussed(x, y, z, focal_distance, NA, e_field=None,
                      power=None, jones_vector=(1, 0),
                      wavelength=1064e-9, n_1=1, n_2=1,
                      filling_factor=inf, aperture_radius=None,
                      width_inc=None, output_magnetic_field=False):
    """Electric field of a strongly focussed Gaussian beam.

    Notes
    -----
    This function is an implementation of chapter 3.6, equation 3.66
    in Principles of Nano-Optics (2nd ed.).

    Parameters
    ----------
    x, y, z : float
        Coordinates where the field will be evaluated in m.
    focal_distance : float
        Focal distance of the focussing lens/objective in m.
    NA : float
        Numerical aperture of the lens/objective.
    e_field : complex, optional
        Field amplitude $E_0$ of the incident Gaussian beam in V/m. If
        None (default), power and width_inc are used to derive the
        field amplitude.
    power : float, optional
        Power of the incident Gaussian beam in Watt (defaults to None).
    jones_vector : 2-tuple of complex, optional
        2D Jones vector that defines the polarization of the incident
        beam. Defaults to (1, 0) for linear x polarization.
    wavelength : float, optional
        Wavelength of the incident monochromatic wave in m.
        Defaults to 1064 nm.
    n_1 : float, optional
        Refractive index of the material before the lens (defaults
        to 1 for vacuum).
    n_2 : float, optional
        Refractive index of the material after the lens (defaults
        to 1 for vacuum).
    filling_factor : float, optional
        Filling factor of the incoming beam (beam width/aperture
        radius). Defaults to inf for perfect focus. If
        aperture_radius and width_inc are given, the filling factor
        is calculated from these values.
    aperture_radius : float, optional
        Back aperture radius of the lens/objective in m (defaults to
        None).
    width_inc : float, optional
        Beam width of the incident Gaussian beam in m (defaults to
        None).
    output_magnetic_field : boolean, optional
        If True, magnetic field is put out in addition to the
        electric field vector (defaults to False).

    Returns
    -------
    ndarray
        3 dimensional vector with the x, y, and z component of the
        electric field in V/m.
    """
    if e_field is None:
        e_field = sqrt(4 * power / (c * epsilon_0 * pi * width_inc**2))

    if aperture_radius is not None and width_inc is not None:
        filling_factor = width_inc / aperture_radius

    r = sqrt(x**2 + y**2)
    phi = arctan2(y, x)

    theta_max = arcsin(NA / n_2)

    k = 2 * pi / wavelength

    prefactor = - 1j * k * focal_distance / 2 * sqrt(
        n_1 / n_2) * e_field * exp(- 1j * k * focal_distance)

    I00 = integral_00(r, z, theta_max, k, filling_factor)
    I01 = integral_01(r, z, theta_max, k, filling_factor)
    I02 = integral_02(r, z, theta_max, k, filling_factor)

    vector = jones_vector[0] * array([I00 + I02 * cos(2 * phi),
                                      I02 * sin(2 * phi),
                                      -2j * I01 * cos(phi)]) + \
             jones_vector[1] * array([-I02 * sin(2 * phi + pi),
                                      I00 + I02 * cos(2 * phi + pi),
                                      -2j * I01 * cos(phi + pi/2)])
    # for the y component, we have to rotate the vector and phi by pi/2

    vector = moveaxis(vector, 0, -1)

    if output_magnetic_field:
        prefactor_mgn = prefactor * c * epsilon_0 * n_2
        vector_mgn = jones_vector[0] * array(
                        [I02 * sin(2 * phi),
                         I00 - I02 * cos(2 * phi),
                         -2j * I01 * cos(phi)]) + \
                     jones_vector[1] * array(
                         [-I00 + I02 * cos(2 * phi + pi),
                          I02 * sin(2 * phi + pi),
                          -2j * I01 * cos(phi + pi/2)])

        vector_mgn = moveaxis(vector_mgn, 0, -1)

        return prefactor * vector, prefactor_mgn * vector_mgn
    else:
        return prefactor * vector


@vectorize
def integral_00(rho, z, theta_max, k, filling_factor=None):
    fw = lambda theta: exp(-sin(theta)**2/(filling_factor**2*sin(
        theta_max)**2))

    func = lambda theta: fw(theta) * sqrt(cos(theta)) * sin(theta) * \
                         (1+cos(theta)) * jn(0, k*rho*sin(theta)) * \
                         exp(1j*k*z*cos(theta))

    real_integral = quad(lambda theta: real(func(theta)), 0, theta_max)
    imag_integral = quad(lambda theta: imag(func(theta)), 0, theta_max)
    return real_integral[0] + 1j * imag_integral[0]


@vectorize
def integral_01(rho, z, theta_max, k, filling_factor=None):
    fw = lambda theta: exp(-sin(theta)**2/(filling_factor**2*sin(
        theta_max)**2))

    func = lambda theta: fw(theta) * sqrt(cos(theta)) * sin(theta)**2 \
                         * jn(1, k*rho*sin(theta)) * \
                         exp(1j*k*z*cos(theta))

    real_integral = quad(lambda theta: real(func(theta)), 0, theta_max)
    imag_integral = quad(lambda theta: imag(func(theta)), 0, theta_max)
    return real_integral[0] + 1j * imag_integral[0]


@vectorize
def integral_02(rho, z, theta_max, k, filling_factor=None):
    fw = lambda theta: exp(-sin(theta)**2/(filling_factor**2*sin(
        theta_max)**2))

    func = lambda theta: fw(theta) * sqrt(cos(theta)) * sin(theta) * \
                         (1-cos(theta)) * jn(2, k*rho*sin(theta)) * \
                         exp(1j*k*z*cos(theta))

    real_integral = quad(lambda theta: real(func(theta)), 0, theta_max)
    imag_integral = quad(lambda theta: imag(func(theta)), 0, theta_max)
    return real_integral[0] + 1j * imag_integral[0]

def mode_matching(waist_distance, beam_waist_0_x, beam_waist_1_x, beam_waist_0_y=None, beam_waist_1_y=None, wavelength=1550e-9):
    if beam_waist_0_y is None:
        beam_waist_0_y = beam_waist_0_x
    if beam_waist_1_y is None:
        beam_waist_1_y = beam_waist_1_x

    rl_0 = rayleigh_length(beam_waist_0_x, beam_waist_0_y, wavelength)
    rl_1 = rayleigh_length(beam_waist_1_x, beam_waist_1_y, wavelength)

    tau_a_x = 2 / sqrt((beam_waist_1_x / beam_waist_0_x + beam_waist_0_x / beam_waist_1_x)**2 + (waist_distance / (rl_0 * rl_1))**2)
    tau_a_y = 2 / sqrt((beam_waist_1_y / beam_waist_0_y + beam_waist_0_y / beam_waist_1_y)**2 + (waist_distance / (rl_0 * rl_1))**2)

    return tau_a_x * tau_a_y