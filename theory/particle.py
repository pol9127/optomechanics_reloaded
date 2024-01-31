from numpy import pi
from scipy.constants import epsilon_0


def polarizability(volume=None, radius=None,
                   permittivity_particle=2.101+1j*0,
                   permittivity_medium=1):
    """Polarizability of a spherical particle.

    Parameters
    ----------
    volume : float, optional
        Volume of the particle in m^3. If None (default) the radius
        is used to derive the volume.
    radius : float, optional
        Radius of the particle. If volume is not given, the radius is
        used to derive the volume.
    permittivity_particle : complex, optional
        Relative permittivity of the particle, defaults to 2.101+1j*0.
    permittivity_medium : complex, optional
        Relative permittivity of the surrounding medium, defaults to 1.

    Returns
    -------
    complex
        Polarizability in A*s*m^2/V.
    """
    if not volume and radius:
        volume = 4 / 3 * pi * radius ** 3

    return 3 * volume * epsilon_0 * (
        permittivity_particle - permittivity_medium) / (
        permittivity_particle + 2 * permittivity_medium)


def effective_polarizability(volume=None, radius=None,
                             wavelength=1064e-9,
                             permittivity_particle=2.101+1j*0,
                             permittivity_medium=1):
    """Effective polarizability of a spherical particle, considering
    backaction or radiation reaction.

    Parameters
    ----------
    volume : float, optional
        Volume of the particle in m^3. If None (default) the radius
        is used to derive the volume.
    radius : float, optional
        Radius of the particle. If volume is not given, the radius is
        used to derive the volume.
    permittivity_particle : complex, optional
        Relative permittivity of the particle, defaults to 2.101+1j*0.
    permittivity_medium : complex, optional
        Relative permittivity of the surrounding medium, defaults to 1.

    Returns
    -------
    complex
        Effective polarizability in A*s*m^2/V.
    """
    k = 2 * pi / wavelength
    alpha = polarizability(volume, radius, permittivity_particle,
                           permittivity_medium)

    return alpha / (1 - 1j * k**3 * alpha / (6 * pi * epsilon_0))


def particle_mass(radius, density=None):
    """Mass of a spherical particle with given radius and density.

    Parameters
    ----------
    radius : float
        Radius of the spherical particle in m.
    density : float, optional
        Mass density of the particle in kg/m^3. Defaults to 2200,
        which is the mass density of bulk fused silica.

    Returns
    -------
    float
        Mass in kg.
    """
    if density is None:
        density = 2200

    return 4 * pi * density * radius ** 3 / 3
