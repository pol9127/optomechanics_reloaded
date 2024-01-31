import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.optimize import curve_fit


def find_x(x, y, y0):
    idx = np.argmin(np.abs(y - y0))
    y1 = y[idx]
    x1 = x[idx]

    if y1 > y0:
        y2 = y[idx + 1]
        x2 = x[idx + 1]
    else:
        y2 = y1
        y1 = y[idx - 1]
        x2 = x1
        x1 = x[idx - 1]

    m = (y2 - y1) / (x2 - x1)
    b = (y1*x2 - y2*x1) / (x2 - x1)
    x_res = (y0 - b) / m
    return x_res


# load data
path = '../../Documentation/report/source/simulations/'


def plot_file(path, title, plot_position, subsampling=3):
    # particle properties
    particle_diameter = 20.02e-6
    particle_density = 1850
    mass = particle_density * 1/6 * np.pi * particle_diameter**3

    # forces
    g = 9.81
    gravity = mass * g

    displacement, scattering_force = np.loadtxt(path)

    cut_start = True
    if cut_start:
        start_cut = 270e-6
        scattering_force = scattering_force[displacement > start_cut]
        displacement = displacement[displacement > start_cut]

    smooth_scattering_force = signal.savgol_filter(scattering_force, 41, 2)
    equilibrium_displacement = find_x(displacement, scattering_force, gravity)
    print('Equilibrium at: {} µm'.format(equilibrium_displacement*1e6))
    print('gravity: {} pN'.format(gravity*1e12))

    # fit
    func = lambda x, I, x0, gamma: I * np.divide(gamma**2, np.power(x-x0, 2) + gamma**2)
    popt, pvar = curve_fit(func, displacement, scattering_force, p0=(300e-12, 0, 300e-6))
    print(popt)
    fitted = func(displacement, *popt)

    # plot
    plt.subplot(1, 2, plot_position)
    plt.axhline(gravity*1e12, color='b')
    plt.plot(displacement[::subsampling]*1e6, scattering_force[::subsampling]*1e12, 'r+', linewidth=1.5)
    plt.plot(displacement*1e6, fitted*1e12, 'k--')
    plt.plot(equilibrium_displacement*1e6, gravity*1e12, 'g.', markersize=10)
    plt.title(title)
    plt.xlabel('particle displacement [µm]')
    plt.ylabel('force [pN]')
    plt.grid(True)
    plt.legend(['gravity', 'axial force', 'Lorentzian fit', 'equilibrium'])


fig = plt.figure(figsize=(10, 4))
plot_file(path+'scattering_force_500.0mW_NA0.055_data.csv', 'Axial force in beam direction (NA=0.055)', 1, subsampling=1)
plot_file(path+'scattering_force_500.0mW_NA0.036_data.csv', 'Axial force in beam direction (NA=0.036)', 2, subsampling=2)

save = False
file = 'force_equilibrium.svg'

if save:
    fig.savefig(path + file)

plt.show()
