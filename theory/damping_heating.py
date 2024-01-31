from numpy import sqrt, pi
from numpy.random import normal
from scipy.constants import Boltzmann, gas_constant, speed_of_light, \
    hbar
from optomechanics.theory.particle import particle_mass


def fluctuating_force(damping_rate, mass, temperature=300, dt=1,
                      size=None):
    """Fluctuating force corresponding to a damping rate.

    The fluctuating force that corresponds to the damping rate,
    mass and temperature, as given by the fluctuation dissipation
    theorem in a time step dt.

    Parameters
    ----------
    damping_rate : float
        Damping rate that the particle experiences in Hz.
    mass : float
        Mass of the particle in kg.
    temperature : float, optional
        Temperature of the system in equilibrium in Kelvin. Defaults to
        300.
    dt : float, optional
        Time step in which the fluctuating force is acting in s.
        Defaults to 1.
    size : int or tuple of ints, optional
        Output shape. Default is None, in which case a single value
        is returned.

    Return
    ------
    force : float or array, shape (len(t), len(y0))
        Fluctuating force. If size is given, an array with specified
        sample size is returned.
    """
    return sqrt(2 * Boltzmann * temperature * damping_rate * mass /
                dt) * normal(0., 1., size)


def heating_rate(damping_rate, Omega0, temperature=300):
    """Heating rate corresponding to a given damping rate, assuming a
    harmonic oscillator with oscillation frequency Omega0.

    Parameters
    ----------
    damping_rate : float
        Damping rate in Hz (angular frequency).
    Omega0 : float
        Oscillation frequency of the harmonic oscillator in Hz (
        angular frequency).
    temperature : float, optional
        Equilibrium temperature of the system, defaults to 300 (room
        temperature).

    Returns
    -------
    float
        Heating rate in Hz (angular frequency).
    """
    return damping_rate * Boltzmann * temperature / (hbar * Omega0)


def damping_rate(heating_rate, Omega0, temperature=300):
    """Heating rate corresponding to a given damping rate, assuming a
    harmonic oscillator with oscillation frequency Omega0.

    Parameters
    ----------
    heating_rate : float
        Heating rate in Hz (angular frequency).
    Omega0 : float
        Oscillation frequency of the harmonic oscillator in Hz (
        angular frequency).
    temperature : float, optional
        Equilibrium temperature of the system, defaults to 300 (room
        temperature).

    Returns
    -------
    float
        Damping rate in Hz (angular frequency).
    """
    return heating_rate * hbar * Omega0 / (Boltzmann * temperature)


def gas_damping(pressure, radius, viscosity=17.81e-6,
                mass=None, density=None, temperature=300,
                molar_mass=28):
    """Derive the gas damping for a particle in an ideal gas. This
    assumes that the particle is in thermal equilibrium with the
    environment.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    radius : float
        Radius of the spherical particle in m.
    viscosity : float, optional
        Viscosity of the gas in Pa*s, defaults to 17.81e-6 (N2 at 300K).
    mass : float, optional
        Mass of the particle in kg. Defaults to None, in which case
        the mass is derived from radius and density.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to None,
        in which case the default value in
        optomechanics.theory.particle.particle_mass is used.
    temperature : float, optional
        Temperature of the gas in Kelvin, defaults to 300 (room
        temperature).
    molar_mass : float, optional
        Molar mass of the gas in g/mol, defaults to 28 (Nitrogen gas
        N2).

    Returns
    -------
    float
        Damping coefficient in Hz (as angular frequency).
    """
    knudsen_number = mean_free_path(
        pressure, temperature=temperature, viscosity=viscosity,
        molar_mass=molar_mass, method='viscosity') / radius
    c_k = 0.31 * knudsen_number / (0.785 + 1.152 * knudsen_number +
                                   knudsen_number**2)

    if mass is None:
        if density is None:
            mass = particle_mass(radius)
        else:
            mass = particle_mass(radius, density)

    return 6 * pi * viscosity * radius * 0.619 / (
        mass * (0.619 + knudsen_number)) * (1 + c_k)


def gas_damping_impinging(pressure, radius, mass=None, density=None,
                          temperature_impinging=300):
    """Gas damping caused by gas molecules impinging on the particle.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    radius : float
        Radius of the spherical particle in m.
    mass : float, optional
        Mass of the particle in kg. Defaults to None, in which case
        the mass is derived from radius and density.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to None,
        in which case the default value in
        optomechanics.theory.particle.particle_mass is used.
    temperature_impinging : float, optional
        Temperature of the impinging gas molecules in K, defaults to
        300 (room temperature).

    Returns
    -------
    Damping coefficient in Hz (angular frequency).
    """
    if mass is None:
        if density is None:
            mass = particle_mass(radius)
        else:
            mass = particle_mass(radius, density)

    velocity = mean_velocity(temperature_impinging)

    return 32/3 * radius ** 2 * pressure / (mass * velocity)


def gas_damping_emerging(pressure, radius, mass=None, density=None,
                         temperature_impinging=300,
                         temperature_emerging=300):
    """Gas damping caused by gas molecules emerging from the particle.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    radius : float
        Radius of the spherical particle in m.
    mass : float, optional
        Mass of the particle in kg. Defaults to None, in which case
        the mass is derived from radius and density.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to None,
        in which case the default value in
        optomechanics.theory.particle.particle_mass is used.
    temperature_impinging : float, optional
        Temperature of the impinging gas molecules in K, defaults to
        300 (room temperature).
    temperature_emerging : float, optional
        Temperature of the emerging gas molecules in K, defaults to
        300 (room temperature).

    Returns
    -------
    Damping coefficient in Hz (angular frequency).
    """
    if mass is None:
        if density is None:
            mass = particle_mass(radius)
        else:
            mass = particle_mass(radius, density)

    return pi / 8 * sqrt(temperature_emerging/temperature_impinging) \
           * gas_damping_impinging(pressure, radius, mass, density,
                                   temperature_impinging)


def gas_damping_2_bath(pressure, radius, mass=None, density=None,
                       environment_temperature=300,
                       particle_temperature=1600,
                       accommodation_coefficient=0.745):
    """Gas damping caused by gas molecules according to the two bath
    model.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    radius : float
        Radius of the spherical particle in m.
    mass : float, optional
        Mass of the particle in kg. Defaults to None, in which case
        the mass is derived from radius and density.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to None,
        in which case the default value in
        optomechanics.theory.particle.particle_mass is used.
    environment_temperature : float, optional
        Temperature of the environment in K, defaults to 300 (room
        temperature).
    particle_temperature : float, optional
        Temperature of the particle surface in K, defaults to 1600.
    accommodation_coefficient : float, optional
        Accommodation coefficient that characterizes the
        thermalization of the gas molecules on the particle surface,
        defaults to 0.745.

    Returns
    -------
    Damping coefficient in Hz (angular frequency).
    """
    temperature_impinging = environment_temperature
    temperature_emerging = accommodation_coefficient * (
        particle_temperature - temperature_impinging) + \
        temperature_impinging

    impinging = gas_damping_impinging(pressure, radius, mass, density,
                                      temperature_impinging)
    emerging = gas_damping_emerging(pressure, radius, mass, density,
                                    temperature_impinging,
                                    temperature_emerging)

    return impinging + emerging


def gas_heating_2_bath(pressure, radius, Omega0, mass=None,
                       density=None, environment_temperature=300,
                       particle_temperature=1600,
                       accommodation_coefficient=0.745):
    """Gas heating caused by gas molecules according to the two bath
    model.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    radius : float
        Radius of the spherical particle in m.
    Omega0 : float
        Oscillation frequency of the harmonic oscillator in Hz (
        angular frequency).
    mass : float, optional
        Mass of the particle in kg. Defaults to None, in which case
        the mass is derived from radius and density.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to None,
        in which case the default value in
        optomechanics.theory.particle.particle_mass is used.
    environment_temperature : float, optional
        Temperature of the environment in K, defaults to 300 (room
        temperature).
    particle_temperature : float, optional
        Temperature of the particle surface in K, defaults to 1600.
    accommodation_coefficient : float, optional
        Accommodation coefficient that characterizes the
        thermalization of the gas molecules on the particle surface,
        defaults to 0.745.

    Returns
    -------
    Heating coefficient in Hz (angular frequency).
    """
    temperature_impinging = environment_temperature
    temperature_emerging = accommodation_coefficient * (
        particle_temperature - temperature_impinging) + \
        temperature_impinging

    impinging = heating_rate(
        gas_damping_impinging(pressure, radius, mass, density,
                              temperature_impinging),
        Omega0, temperature_impinging)
    emerging = heating_rate(
        gas_damping_emerging(pressure, radius, mass, density,
                             temperature_impinging,
                             temperature_emerging),
        Omega0, temperature_emerging)

    return impinging + emerging


def radiation_damping(scattered_power, mass, temperature=300,
                      wavelength=1064e-6):
    """Radiation damping experienced by a particle in an optical trap.

    Parameters
    ----------
    scattered_power : float
        Power scattered from the particle in W.
    mass : float
        Mass of the particle in kg.
    temperature : float, optional
        Temperature in Kelvin, defaults to 300 (room temperature).
    wavelength : float, optional
        Wavelength of the light in m, defaults to 1064e-9.

    Returns
    -------
    float
        Radiation damping in Hz (angular frequency).
    """
    omega0 = 2 * pi * speed_of_light / wavelength
    return scattered_power * hbar * omega0 / (
        5 * mass * speed_of_light**2 * Boltzmann * temperature)


def recoil_heating(scattered_power, mass, Omega0, wavelength=1064e-6):
    """Recoil heating experienced by a particle in an optical trap.

    Parameters
    ----------
    scattered_power : float
        Power scattered from the particle in W.
    mass : float
        Mass of the particle in kg.
    Omega0 : float
        Oscillation frequency of the harmonic oscillator in Hz (
        angular frequency).
    wavelength : float, optional
        Wavelength of the light in m, defaults to 1064e-9.

    Returns
    -------
    float
        Recoil heating in Hz (angular frequency).
    """
    omega0 = 2 * pi * speed_of_light / wavelength
    return scattered_power * omega0 / (
        5 * mass * speed_of_light ** 2 * Omega0)


def mean_velocity(temperature=300, molar_mass=28):
    """Mean velocity of molecules in an ideal gas.

    Parameters
    ----------
    temperature : float, optional
        Temperature of the gas in Kelvin, defaults to 300 (room
        temperature).
    molar_mass : float, optional
        Molar mass of the gas in g/mol, defaults to 28 (Nitrogen gas
        N2).

    Returns
    -------
    float
        Mean velocity of the gas molecules in m/s.
    """
    return sqrt(8*gas_constant*temperature/(pi*(molar_mass/1000)))


def rms_velocity(temperature=300, molar_mass=28):
    """RMS velocity of molecules in an ideal gas.

    Parameters
    ----------
    temperature : float, optional
        Temperature of the gas in Kelvin, defaults to 300 (room
        temperature).
    molar_mass : float, optional
        Molar mass of the gas in g/mol, defaults to 28 (Nitrogen gas
        N2).

    Returns
    -------
    float
        RMS velocity of the gas molecules in m/s.
    """
    return sqrt(3*gas_constant*temperature/(molar_mass/1000))


def mean_free_path(pressure, temperature=300,
                   molecule_diameter=364e-12, viscosity=17.81e-6,
                   molar_mass=28, method='diameter'):
    """Calculate the mean free path of a gas.

    Parameters
    ----------
    pressure : float
        Pressure in Pa (1 mbar = 100 Pa).
    temperature : float, optional
        Temperature of the gas in Kelvin, defaults to 300 (room
        temperature).
    molecule_diameter : float, optional
        Kinetic diameter of the gas molecules in m, defaults to
        364e-12 (Nitrogen molecules).
    viscosity : float, optional
        Viscosity of the gas in Pa*s, defaults to 17.81e-6 (N2 at 300K).
    molar_mass : float, optional
        Molar mass of the gas in g/mol, defaults to 28 (Nitrogen gas
        N2).
    method : string, optional
        'diameter' (default) for equation using the kinetic molecule
        diameter, or 'viscosity' for equation using the viscosity and
        molar mass.

    Returns
    -------
    float
        Mean free path in m.
    """
    if method is 'diameter':
        return Boltzmann * temperature / (
            sqrt(2) * pi * molecule_diameter ** 2 * pressure)
    elif method is 'viscosity':
        return viscosity / pressure * sqrt(
            pi * gas_constant * temperature / (2 * molar_mass / 1000))
