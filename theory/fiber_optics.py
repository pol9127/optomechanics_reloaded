import numpy as np

def numerical_aperture(ref_idx_core, ref_idx_cladding):
    return np.sqrt(ref_idx_core**2 - ref_idx_cladding**2)


def numerical_aperture_eff(mode_field_diameter, wavelength=1550e-9):
    return 2 * wavelength / (np.pi * mode_field_diameter)


def collimated_waist(mode_field_diameter, focal_length, wavelength=1550e-9):
    return focal_length * numerical_aperture_eff(mode_field_diameter, wavelength)


def v_number(core_radius, numerical_aperture, wavelength=1550e-9):
    return (2 * np.pi * core_radius * numerical_aperture) / wavelength


def fiber_waist(core_radius, numerical_aperture, wavelength=1550e-9, profile='step'):
    if profile == 'step':
        return core_radius / np.sqrt(2 * np.log((2 * np.pi * core_radius * numerical_aperture) / wavelength ))
    elif profile == 'gauss':
        return core_radius / np.sqrt((2 * np.pi * core_radius * numerical_aperture) / wavelength - 1)
    else:
        return None


def profile_height_parameter(ref_idx_core, ref_idx_cladding):
    return (ref_idx_core**2 - ref_idx_cladding**2) / (2 * ref_idx_core**2)
