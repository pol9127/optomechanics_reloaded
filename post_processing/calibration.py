from __future__ import division, print_function, unicode_literals

import numpy
from scipy import constants


def calibrate(fit_parameters, temperature, gas_pressure, density=2200,
              particle_size=None):
    # type: (numpy.ndarray or (numpy.ndarray, numpy.ndarray),
    #        float or (float, float), float or (float, float))
    """Derive calibration parameters and their errors from fit
    parameters.

    Given the parameters of an Lorentzian fit (amplitude, FWHM), the
    temperature and the gas pressure, the following values are derived:

    - particle radius
    - particle mass (from the radius, using the density
      $\rho_\mathrm{SiO_2}=2200 g cm^-3$)
    - calibration factor for relating the measured values to
      displacement in nm
    - R factor, necessary for deriving the effective temperature

    Notes
    -----
    More detailed information can be found in the appendix of Jan
    Gieseler's PhD thesis.

    The function expects fit parameters that belong to a PSD in
    f-space. See calibration document for details.

    The errors are not necessary as input parameters. 0 will be assumed.

    Parameters
    ----------
    fit_parameters
        Tuple of:
        -   ndarray with parameters of the fitted Lorentzian in the form
            [amplitude, FWHM, center frequency, offset]. At least the
            first two items of the list are necessary
        -   Covariance matrix for fit_parameters as returned by
            scipy.optimize.curve_fit.
    temperature
        Tuple of:
        -   Temperature at which the data was taken, in Kelvin (K).
        -   Standard deviation of the temperature value T0.
    gas_pressure
        Tuple of:
        -   Pressure at which the data was taken, in mBar.
        -   Standard deviation of the pressure value gas_pressure.
    density : float, optional
        Density of the particle in kg/m^3, Default is 2200.
    particle_size : optional
        Tuple of:
        -   Particle radius, in m.
        -   Standard deviation of the particle size in m.

    Returns
    -------
    tuple of float:
        calibration factor and its error
    tuple of float:
        R factor and its error
    tuple of float:
        particle radius and its error
    tuple of float:
        particle mass and its error
    """

    if type(fit_parameters) is not tuple:
        fit_parameters = (
        fit_parameters, numpy.zeros((len(fit_parameters),
                                     len(fit_parameters))))
    if type(temperature) is not tuple:
        temperature = (temperature, 0)
    if type(gas_pressure) is not tuple:
        gas_pressure = (gas_pressure, 0)

    # mBar to Pa
    gas_pressure = (gas_pressure[0] * 100, gas_pressure[1] * 100)

    # Constants
    dm = 0.372e-9
    eta = 18.27e-6
    rho_SiO2 = density

    if particle_size is None:
        particle_size = \
            0.619 * (9 * numpy.pi / numpy.sqrt(2)) * (eta * dm ** 2 / (
                rho_SiO2 * constants.Boltzmann * temperature[0])) * \
            (gas_pressure[0] / (2 * numpy.pi * fit_parameters[0][1]))

        particle_size_error = \
            numpy.abs(particle_size) * numpy.sqrt(
                (temperature[1] / temperature[0]) ** 2 +
                (gas_pressure[1] / gas_pressure[0]) ** 2 +
                (numpy.sqrt(fit_parameters[1][1, 1]) /
                 fit_parameters[0][1]) ** 2)

    elif type(particle_size) is tuple:
        particle_size_error = particle_size[1]
        particle_size = particle_size[0]

    particle_mass = 4 * numpy.pi * rho_SiO2 * particle_size ** 3 / 3

    particle_mass_error = numpy.abs(
        particle_mass) * 3 * particle_size_error / particle_size

    R_calibrated = 2 * numpy.pi ** 2 * fit_parameters[0][0] / \
        fit_parameters[0][1]

    R_calibrated_error = \
        numpy.abs(R_calibrated) * numpy.sqrt(
            (numpy.sqrt(fit_parameters[1][0, 0]) /
             fit_parameters[0][0]) ** 2 +
            (numpy.sqrt(fit_parameters[1][1, 1]) /
             fit_parameters[0][1]) ** 2 -
            (2 * fit_parameters[1][0, 1] /
             (fit_parameters[0][0] * fit_parameters[0][1])))

    c_calibrated = 2 * numpy.pi * numpy.sqrt(
        fit_parameters[0][0] / fit_parameters[0][1] * numpy.pi *
        particle_mass / (2 * constants.Boltzmann * temperature[0]))

    c_calibrated_error = numpy.abs(c_calibrated) * numpy.sqrt(
        (1.5 * gas_pressure[1] / gas_pressure[0]) ** 2 +
        (2 * temperature[1] / temperature[0]) ** 2 +
        (numpy.sqrt(fit_parameters[1][0, 0]) /
         (2 * fit_parameters[0][0])) ** 2 +
        (2 * numpy.sqrt(fit_parameters[1][1, 1]) /
         fit_parameters[0][1]) ** 2 -
        (2 * fit_parameters[1][0, 1] /
         (fit_parameters[0][0] * fit_parameters[0][1])))

    return (c_calibrated, c_calibrated_error), \
           (R_calibrated, R_calibrated_error), \
           (particle_size, particle_size_error), \
           (particle_mass, particle_mass_error)

def calibrate_overdamped(fit_parameters,
                         temperature,
                         gas_pressure, density = 2200,
                         particle_size = (68e-9,10e-9)):
    """Estimation of the calibration factor from fit_parameters of
    damped Lorentzian Distribution.
    
    Notes
    ----------------------------------------------
    A different error propagation analysis must be done at high pressure
    due to the fact that damping coefficient and mass have a non-zero covariance.
    The origin of the non-zero covariance lies in the fact that the fit does not allow
    for an independent estimation of the damping coefficient. Thus, the radius (and hence 
    its uncertainty) is used for calculating both mass and damping.
    Specifically, a function of the form A/(f**2 + fc**2) is used. According to the PhD Thesis
    of Erik Hebestreit, Eq.(4.11) on page 76, the calibration factor is given by
    c**2 = 2*pi**3*A*m*g/(kb*T), with g = Gamma/(2*pi). Eq. (38) on page 132 of Jan Gieseler'r 
    PhD's thesis can eventually be substituted, leading to an expression for c**2 made of independent
    parameters on which the error propagation used in this method is calculated.
    
    c**2 = 12*pi**4*0.619/sqrt(2) * eta*dm**2/kb**2 * pressure*A*radius**2/temperature**2
    ------------------------------------------------
    
    Parameters
    ----------
    fit_parameters
        Tuple of:
        -   ndarray with parameters of the fitted Lorentzian in the form
            [amplitude, cut_off_frequency]. At least the
            first item on the list is necessary
        -   Covariance matrix for fit_parameters as returned by
            scipy.optimize.curve_fit.
    temperature
        Tuple of:
        -   Temperature at which the data was taken, in Kelvin (K).
        -   Standard deviation of the temperature value T0.
    gas_pressure
        Tuple of:
        -   Pressure at which the data was taken, in mBar.
        -   Standard deviation of the pressure value gas_pressure.
    density : float, optional
        Density of the particle in kg/m^3, Default is 2200.
    particle_size : optional
        Tuple of:
        -   Particle radius, in m.
        -   Standard deviation of the particle size in m.

    Returns
    -------
    tuple of float:
        calibration factor and its error
    tuple of float:
        particle mass and its error
    tuple of float:
        damping coefficient and its error
    """
    
    if type(fit_parameters) is not tuple:
        fit_parameters = (
        fit_parameters, numpy.zeros((len(fit_parameters),
                                     len(fit_parameters))))
    if type(temperature) is not tuple:
        temperature = (temperature, 0)
    if type(gas_pressure) is not tuple:
        gas_pressure = (gas_pressure, 0)
    if type(particle_size) is not tuple:
        particle_size = (particle_size,0)
        
    # mBar to Pa
    gas_pressure = (gas_pressure[0] * 100, gas_pressure[1] * 100)

    # Constants
    dm = 0.372e-9           # air molecule diameter
    eta = 18.27e-6          # viscosity coefficient of air [Pa s]
    rho_SiO2 = density      # density of silica
    
    Gamma = \
            0.619 * (9 * numpy.pi / numpy.sqrt(2)) * (eta * dm ** 2 / (
                rho_SiO2 * constants.Boltzmann * temperature[0])) * \
            (gas_pressure[0] / (particle_size[0]))

    Gamma_error = \
        numpy.abs(particle_size[0]) * numpy.sqrt(
            (temperature[1] / temperature[0]) ** 2 +
            (gas_pressure[1] / gas_pressure[0]) ** 2 +
            (numpy.sqrt(particle_size[1]) /
             particle_size[0]) ** 2)
             
    particle_mass = 4/3 * numpy.pi * particle_size[0]**3*density
    particle_mass_error = particle_mass * numpy.sqrt(3 * particle_size[1]/particle_size[0])
    
    g = Gamma/(2*numpy.pi)
    g_error = Gamma_error/(2*numpy.pi)
    
    num = 12*numpy.pi**4*0.619/numpy.sqrt(2)
    const = eta * dm**2 /constants.Boltzmann**2
    
    c_calibrated = numpy.sqrt(num * const * (
            gas_pressure[0] * fit_parameters[0][0] * particle_size[0]**2 / temperature[0]**2))
    c_calibrated_error = c_calibrated * numpy.sqrt((
                    particle_size[1]/particle_size[0])**2 + (
                    temperature[1]/temperature[0])**2 + (
                    1/2*gas_pressure[1]/gas_pressure[0])**2 + (
                    1/2*numpy.sqrt(fit_parameters[1][0,0])/fit_parameters[0][0])**2)
                    
    g_tuple = (g,g_error)
    m_tuple = (particle_mass,particle_mass_error)
    c_tuple = (c_calibrated,c_calibrated_error)
    
    return c_tuple, m_tuple, g_tuple
    
    
def derive_effective_temperature(fit_parameters, R_calibration,
                                 temperature_calibration=(300, 0)):
    # type: (numpy.ndarray or (numpy.ndarray, numpy.ndarray),
    #        float or (float, float), float or (float, float))
    """Derive the effective temperature from fit parameters of a data
    set and the calibration data.

    Note
    ----
    The errors are not necessary as input parameters. 0 will be assumed.

    Parameters
    ----------
    fit_parameters
        Tuple of:
        -   ndarray with parameters of the fitted Lorentzian in the form
            [amplitude, FWHM, center frequency, offset]. At least the
            first two items of the list are necessary
        -   Covariance matrix for fit_parameters as returned by
            scipy.optimize.curve_fit.
    R_calibration
        Tuple of:
        -   R factor for calibration.
        -   Standard deviation of the R factor.
    temperature_calibration
        Tuple of:
        -   Temperature at which the calibration data was taken, in K.
        -   Standard deviation of the temperature.

    Returns
    -------
    tuple of float:
        effective temperature and its error
    """

    if type(fit_parameters) is not tuple:
        fit_parameters = (
        fit_parameters, numpy.zeros((len(fit_parameters),
                                     len(fit_parameters))))
    if type(R_calibration) is not tuple:
        R_calibration = (R_calibration, 0)
    if type(temperature_calibration) is not tuple:
        temperature_calibration = (temperature_calibration, 0)

    R_feedback = \
        (2 * numpy.pi) ** 2 / 2 * (fit_parameters[0][0] /
                                   fit_parameters[0][1])

    R_feedback_error = numpy.abs(R_feedback) * numpy.sqrt(
        (numpy.sqrt(fit_parameters[1][0, 0]) /
         fit_parameters[0][0]) ** 2 +
        (numpy.sqrt(fit_parameters[1][1, 1]) /
         fit_parameters[0][1]) ** 2 -
        (2 * fit_parameters[1][0, 1] /
         (fit_parameters[0][0] * fit_parameters[0][1])))

    T_eff = temperature_calibration[0] * R_feedback / R_calibration[0]

    T_eff_error = numpy.abs(T_eff) * numpy.sqrt(
        (R_feedback_error / R_feedback) ** 2 +
        (R_calibration[1] / R_calibration[0]) ** 2 +
        (temperature_calibration[1] / temperature_calibration[0]) ** 2)

    return T_eff, T_eff_error


def derive_natural_damping_rate(fit_parameters, R_calibration):
    # type: (numpy.ndarray or (numpy.ndarray, numpy.ndarray),
    #        float or (float, float))
    """Derive the natural damping rate from fit parameters of a data
    set and the calibration data.

    Note
    ----
    The errors are not necessary as input parameters. 0 will be assumed.

    Parameters
    ----------
    fit_parameters
        Tuple of:
        -   ndarray with parameters of the fitted Lorentzian in the form
            [amplitude, FWHM, center frequency, offset]. At least the
            first two items of the list are necessary
        -   Covariance matrix for fit_parameters as returned by
            scipy.optimize.curve_fit.
    R_calibration
        Tuple of:
        -   R factor for calibration.
        -   Standard deviation of the R factor.

    Returns
    -------
    tuple of float:
        natural damping rate and its error
    """

    if type(fit_parameters) is not tuple:
        fit_parameters = (fit_parameters,
                          numpy.zeros((len(fit_parameters),
                                       len(fit_parameters))))
    if type(R_calibration) is not tuple:
        R_calibration = (R_calibration, 0)

    g_0 = 4 * numpy.pi ** 3 * fit_parameters[0][0] / R_calibration[0]

    g_0_error = numpy.abs(g_0) * \
                numpy.sqrt((numpy.sqrt(fit_parameters[1][0, 0]) /
                            fit_parameters[0][0]) ** 2 +
                           (R_calibration[1] / R_calibration[0]) ** 2)

    return g_0, g_0_error
