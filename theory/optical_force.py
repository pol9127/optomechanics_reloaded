from numpy import imag, real, conjugate, array, pi, angle, isnan, \
    isscalar, moveaxis
from scipy import constants
from scipy.misc import derivative
from .gaussian_beam import rayleigh_length, intensity_gauss, \
    wavefront_radius
from .particle import effective_polarizability


def harmonic_force(x, omega, mass):
    """ Restoring force of a harmonic oscillator with given
    frequency and mass.

    The force is given as

        F = - omega**2 * mass * x

    Notes
    -----
    The frequency omega is given as an angular frequency which is
    related to the ordinary frequency f by

        Omega = 2 * pi * f

    Parameters
    ----------
    x : float or array
        Position of the oscillator.
    omega : float
        Natural angular frequency of the oscillator in Hz.
    mass : float
        Mass of the oscillator in kg.

    Return
    ------
    force : float or array, shape (len(t), len(y0))
        Fluctuating force in N. If size is given, an array with
        specified sample size is returned.
    """
    return -omega**2 * mass * x


def gradient_force_gaussian(x, y, z, power, width_x, width_y=None,
                            volume=None, radius=None,
                            wavelength=1064e-9, rayleigh_len=None,
                            permittivity_particle=2.101+1j*0,
                            permittivity_medium=1):
    """Gradient force acting on a particle in a Gaussian beam focus.

    This function returns the analytical solution of the gradient
    force for a Gaussian beam as noted in Jan Gieselers thesis
    Eq. (2.14).

    The point x=y=z=0 marks the geometrical focus of the beam.

    Parameters
    ----------
    x : float or array
        X coordinate of the particle in m.
    y : float or array
        Y coordinate of the particle in m.
    z : float or array
        Z coordinate of the particle in m.
    power : float
        Power of the Gaussian beam in W.
    width_x : float
        Width of the Gaussian focus in x direction in m.
    width_y : float, optional
        Width of the Gaussian focus in y direction in m. If None
        (default), width_y = width_x.
    volume : float, optional
        Volume of the particle in m**3. If None (default),
        the radius is used to derive the volume. Either volume or
        radius have to be specified.
    radius : float, optional
        Radius of the particle in m. If None (default), the volume is
        used to derive the radius. Either volume or radius have to
        be specified.
    wavelength : float, optional
        Wavelength of the monochromatic Gaussian beam in m. Defaults
        to 1064e-9.
    rayleigh_len : float, optional
        Rayleigh length of the Gaussian beam in m. If None (default),
        this is derived from the beam width and wavelength.
    permittivity_particle : complex, optional
        Permittivity of the particle. Defaults to 2.101 (Silica).
    permittivity_medium : complex, optional
        Permittivity of the surrounding medium. Defaults to 1 (vacuum).

    Returns
    -------
    array
        3 dimensional array, containing the gradient force in x, y,
        and z direction in N.

    """
    if width_y is None:
        width_y = width_x
    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    prefactor = -real(
        effective_polarizability(volume, radius, wavelength,
                                 permittivity_particle,
                                 permittivity_medium))
    prefactor *= intensity_gauss(x, y, z, width_x, width_y,
                                 power=power, wavelength=wavelength,
                                 rayleigh_len=rayleigh_len) * 2
    prefactor /= constants.c * constants.epsilon_0

    x_factor = x * rayleigh_len ** 2 / (
        width_x ** 2 * (z ** 2 + rayleigh_len ** 2))
    y_factor = y * rayleigh_len ** 2 / (
        width_y ** 2 * (z ** 2 + rayleigh_len ** 2))
    z_factor = z * ((z / rayleigh_len) ** 2 + (
        1 - 2*x ** 2 / width_x ** 2 - 2 * y ** 2 / width_y ** 2)) * \
        rayleigh_len ** 2 / (2 * (z ** 2 + rayleigh_len ** 2) ** 2)

    return array([prefactor * x_factor, prefactor * y_factor,
                  prefactor * z_factor])


def scattering_force_gaussian(x, y, z, power, width_x, width_y=None,
                              volume=None, radius=None,
                              wavelength=1064e-9, rayleigh_len=None,
                              permittivity_particle=2.101 + 1j * 0,
                              permittivity_medium=1):
    """Scattering force acting on a particle in a Gaussian beam focus.

    This function returns the analytical solution of the scattering
    force for a Gaussian beam as noted in Jan Gieselers thesis
    Eq. (2.15).

    The point x=y=z=0 marks the geometrical focus of the beam.

    Parameters
    ----------
    x : float or array
        X coordinate of the particle in m.
    y : float or array
        Y coordinate of the particle in m.
    z : float or array
        Z coordinate of the particle in m.
    power : float
        Power of the Gaussian beam in W.
    width_x : float
        Width of the Gaussian focus in x direction in m.
    width_y : float, optional
        Width of the Gaussian focus in y direction in m. If None
        (default), width_y = width_x.
    volume : float, optional
        Volume of the particle in m**3. If None (default),
        the radius is used to derive the volume. Either volume or
        radius have to be specified.
    radius : float, optional
        Radius of the particle in m. If None (default), the volume is
        used to derive the radius. Either volume or radius have to
        be specified.
    wavelength : float, optional
        Wavelength of the monochromatic Gaussian beam in m. Defaults
        to 1064e-9.
    rayleigh_len : float, optional
        Rayleigh length of the Gaussian beam in m. If None (default),
        this is derived from the beam width and wavelength.
    permittivity_particle : complex, optional
        Permittivity of the particle. Defaults to 2.101 (Silica).
    permittivity_medium : complex, optional
        Permittivity of the surrounding medium. Defaults to 1 (vacuum).

    Returns
    -------
    array
        3 dimensional array, containing the scattering force in x, y,
        and z direction in N.
    """
    if width_y is None:
        width_y = width_x
    if rayleigh_len is None:
        rayleigh_len = rayleigh_length(width_x, width_y, wavelength)

    k = 2 * pi / wavelength

    prefactor = imag(
        effective_polarizability(volume, radius, wavelength,
                                 permittivity_particle,
                                 permittivity_medium)) / 2
    prefactor *= intensity_gauss(x, y, z, width_x, width_y,
                                 power=power, wavelength=wavelength,
                                 rayleigh_len=rayleigh_len) * 2
    prefactor /= constants.c * constants.epsilon_0
    prefactor *= k

    x_factor = x / wavefront_radius(z, rayleigh_len)
    y_factor = y / wavefront_radius(z, rayleigh_len)
    z_factor = array(1 + (x ** 2 + y ** 2) * rayleigh_len ** 2 / (
        z ** 2 * wavefront_radius(z, rayleigh_len) ** 2) - (
        x ** 2 + y ** 2 + 2 * z * rayleigh_len) / (
        2 * z * wavefront_radius(z, rayleigh_len)))

    if isnan(z_factor).any():
        if isscalar(z_factor):
            z_factor = 1
        else:
            z_factor[isnan(z_factor)] = 1

    return array([prefactor * x_factor, prefactor * y_factor,
                  prefactor * z_factor])


def total_force_gaussian(x, y, z, power, width_x, width_y=None,
                         volume=None, radius=None,
                         wavelength=1064e-9, rayleigh_len=None,
                         permittivity_particle=2.101 + 1j * 0,
                         permittivity_medium=1):
    """Total force acting on a particle in a Gaussian beam focus.

    This function returns the analytical solution of the total optical
    force for a Gaussian beam as noted in Jan Gieselers thesis
    Eqs. (2.14), (2.15).

    The point x=y=z=0 marks the geometrical focus of the beam.

    Parameters
    ----------
    x : float or array
        X coordinate of the particle in m.
    y : float or array
        Y coordinate of the particle in m.
    z : float or array
        Z coordinate of the particle in m.
    power : float
        Power of the Gaussian beam in W.
    width_x : float
        Width of the Gaussian focus in x direction in m.
    width_y : float, optional
        Width of the Gaussian focus in y direction in m. If None
        (default), width_y = width_x.
    volume : float, optional
        Volume of the particle in m**3. If None (default),
        the radius is used to derive the volume. Either volume or
        radius have to be specified.
    radius : float, optional
        Radius of the particle in m. If None (default), the volume is
        used to derive the radius. Either volume or radius have to
        be specified.
    wavelength : float, optional
        Wavelength of the monochromatic Gaussian beam in m. Defaults
        to 1064e-9.
    rayleigh_len : float, optional
        Rayleigh length of the Gaussian beam in m. If None (default),
        this is derived from the beam width and wavelength.
    permittivity_particle : complex, optional
        Permittivity of the particle. Defaults to 2.101 (Silica).
    permittivity_medium : complex, optional
        Permittivity of the surrounding medium. Defaults to 1 (vacuum).

    Returns
    -------
    array
        3 dimensional array, containing the total optical force in x, y,
        and z direction in N.

    """
    grad_force = gradient_force_gaussian(x, y, z, power, width_x,
                                         width_y, volume, radius,
                                         wavelength, rayleigh_len,
                                         permittivity_particle,
                                         permittivity_medium)
    scat_force = scattering_force_gaussian(x, y, z, power, width_x,
                                           width_y, volume, radius,
                                           wavelength, rayleigh_len,
                                           permittivity_particle,
                                           permittivity_medium)

    return grad_force + scat_force


def gradient_force(x0, y0, z0, electric_field, polarizability,
                   dx=1e-10, dy=1e-10, dz=1e-10):
    """Gradient force acting on a particle given an electric field.

    This function returns the numerical solution of the gradient
    force for an arbitrary electric field as noted in Jan Gieselers
    thesis Eq. (2.10).

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the gradient force in x, y,
        and z direction in N.
    """
    grad_e_x = derivative(lambda x: electric_field(x, y0, z0), x0,
                          dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0), y0,
                          dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z), z0,
                          dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = e_conj * array([grad_e_x, grad_e_y, grad_e_z])

    return real(real(polarizability) / 2 * real(vector))


def scattering_force(x0, y0, z0, electric_field, polarizability,
                     dx=1e-10, dy=1e-10, dz=1e-10):
    """Scattering force acting on a particle given an electric field.

    This function returns the numerical solution of the scattering
    force for an arbitrary electric field as noted in Jan Gieselers
    thesis Eq. (2.11).

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the scattering force in x, y,
        and z direction in N.
    """
    grad_e_x = derivative(lambda x: electric_field(x, y0, z0), x0,
                          dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0), y0,
                          dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z), z0,
                          dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = e_conj * array([grad_e_x, grad_e_y, grad_e_z])

    return real(imag(polarizability) / 2 * imag(vector))


def total_force(x0, y0, z0, electric_field, polarizability,
                dx=1e-10, dy=1e-10, dz=1e-10):
    """Total force acting on a particle given an electric field.

    This function returns the numerical solution of the total optical
    force for an arbitrary electric field as noted in Jan Gieselers
    thesis Eqs. (2.10), (2.11).

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the total optical force in x, y,
        and z direction in N.
    """
    grad_e_x = derivative(lambda x: electric_field(x, y0, z0), x0,
                          dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0), y0,
                          dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z), z0,
                          dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = e_conj * array([grad_e_x, grad_e_y, grad_e_z])

    return real(real(polarizability) / 2 * real(vector) +
                imag(polarizability) / 2 * imag(vector))


def gradient_force_vector(x0, y0, z0, electric_field, polarizability,
                          dx=1e-10, dy=1e-10, dz=1e-10):
    """Gradient force acting on a particle given a vector field.

    This function returns the numerical solution of the gradient
    force for an electric vector field as in Principles of
    Nano-Optics equation 14.40.

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the gradient force in x, y,
        and z direction in N.
    """

    grad_e_x = derivative(lambda x: electric_field(x, y0, z0),
                           x0, dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0),
                           y0, dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z),
                           z0, dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = array([e_conj[..., 0] * grad_e_x[..., 0] +
                    e_conj[..., 1] * grad_e_x[..., 1] +
                    e_conj[..., 2] * grad_e_x[..., 2],
                    e_conj[..., 0] * grad_e_y[..., 0] +
                    e_conj[..., 1] * grad_e_y[..., 1] +
                    e_conj[..., 2] * grad_e_y[..., 2],
                    e_conj[..., 0] * grad_e_z[..., 0] +
                    e_conj[..., 1] * grad_e_z[..., 1] +
                    e_conj[..., 2] * grad_e_z[..., 2]])

    return real(real(polarizability) / 2 *
                moveaxis(real(vector), 0, -1))


def scattering_force_vector(x0, y0, z0, electric_field, polarizability,
                            dx=1e-10, dy=1e-10, dz=1e-10):
    """Scattering force acting on a particle given a vector field.

    This function returns the numerical solution of the scattering
    force for an electric vector field as in Principles of
    Nano-Optics equation 14.40.

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the scattering force in x, y,
        and z direction in N.
    """

    grad_e_x = derivative(lambda x: electric_field(x, y0, z0),
                          x0, dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0),
                          y0, dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z),
                          z0, dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = array([e_conj[..., 0] * grad_e_x[..., 0] +
                    e_conj[..., 1] * grad_e_x[..., 1] +
                    e_conj[..., 2] * grad_e_x[..., 2],
                    e_conj[..., 0] * grad_e_y[..., 0] +
                    e_conj[..., 1] * grad_e_y[..., 1] +
                    e_conj[..., 2] * grad_e_y[..., 2],
                    e_conj[..., 0] * grad_e_z[..., 0] +
                    e_conj[..., 1] * grad_e_z[..., 1] +
                    e_conj[..., 2] * grad_e_z[..., 2]])
    return real(imag(polarizability) / 2 *
                moveaxis(imag(vector), 0, -1))


def total_force_vector(x0, y0, z0, electric_field, polarizability,
                       dx=1e-10, dy=1e-10, dz=1e-10):
    """Total force acting on a particle given a vector field.

    This function returns the numerical solution of the total optical
    force for an electric vector field as in Principles of
    Nano-Optics equation 14.40.

    Parameters
    ----------
    x0 : float or array
        X coordinate of the particle in m.
    y0 : float or array
        Y coordinate of the particle in m.
    z0 : float or array
        Z coordinate of the particle in m.
    electric_field : callable(x, y, z)
        Function that returns the complex valued electric field in V/m.
    polarizability : complex
        Complex valued polarizability of the particle.
    dx : float, optional
        Step to use to numerically evaluate the gradient in x
        direction in m. Defaults to 1e-10.
    dy : float, optional
        Step to use to numerically evaluate the gradient in y
        direction in m. Defaults to 1e-10.
    dz : float, optional
        Step to use to numerically evaluate the gradient in z
        direction in m. Defaults to 1e-10.

    Returns
    -------
    array
        3 dimensional array, containing the total optical force in x, y,
        and z direction in N.
    """

    grad_e_x = derivative(lambda x: electric_field(x, y0, z0),
                          x0, dx=dx)
    grad_e_y = derivative(lambda y: electric_field(x0, y, z0),
                          y0, dx=dy)
    grad_e_z = derivative(lambda z: electric_field(x0, y0, z),
                          z0, dx=dz)

    e_conj = conjugate(electric_field(x0, y0, z0))

    vector = array([e_conj[..., 0] * grad_e_x[..., 0] +
                    e_conj[..., 1] * grad_e_x[..., 1] +
                    e_conj[..., 2] * grad_e_x[..., 2],
                    e_conj[..., 0] * grad_e_y[..., 0] +
                    e_conj[..., 1] * grad_e_y[..., 1] +
                    e_conj[..., 2] * grad_e_y[..., 2],
                    e_conj[..., 0] * grad_e_z[..., 0] +
                    e_conj[..., 1] * grad_e_z[..., 1] +
                    e_conj[..., 2] * grad_e_z[..., 2]])

    return real(real(polarizability) / 2 *
                moveaxis(real(vector), 0, -1) +
                imag(polarizability) / 2 *
                moveaxis(imag(vector), 0, -1))
