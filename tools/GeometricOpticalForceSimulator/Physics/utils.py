import numpy as np


def find_force_equilibrium(displacement, scattering_force, gravitational_force):
    """
    Find the position at which the scattering force and gravity are equal. (Uses interpolation for accurate results)
    :param displacement: array of displacements
    :param scattering_force: array of scattering force values along displacement axis
    :param gravitational_force: gravity
    :return: equilibrium position
    """
    idx = np.argmin(np.abs(scattering_force - gravitational_force))     # get index of point closest to gravity
    y1 = scattering_force[idx]      # get the scattering force value at this point
    y0 = gravitational_force        # use gravity as starting point for linear interpolation
    x1 = displacement[idx]          # get first position point for interpolation

    if y1 > y0:     # if the scattering force point is bigger than gravity take next force (usually smaller)
        y2 = scattering_force[idx + 1]
        x2 = displacement[idx + 1]
    else:    # if the scattering force point is smaller, take one point earlier (usually bigger)
        y2 = y1
        y1 = scattering_force[idx - 1]
        x2 = x1
        x1 = displacement[idx - 1]

    # it can be assumed that y2 < y0 < y1, therefore gravity (y0) lies between y1, y2
    m = (y2 - y1) / (x2 - x1)       # calculate slope between y1, y2
    b = (y1 * x2 - y2 * x1) / (x2 - x1)     # calculate linear offset
    equilibrium_distance = (y0 - b) / m     # calculate exact point on which y0 lies
    return equilibrium_distance
