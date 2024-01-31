import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


path = '../../Documentation/report/source/simulations/'
gradient_force_mie_file = 'ottb_500mW_eq_gradient.txt'
scattering_force_mie_file = 'ottb_500mW_eq_scattering.txt'
#gradient_force_geom_file = 'gradient_force_500mW_grad1_data.csv'
#scattering_force_geom_file = 'gradient_force_500mW_scattering_data.csv'
gradient_force_geom_file = 'radial_force_500.0mW_NA0.1_gradient_data.csv'
scattering_force_geom_file = 'radial_force_500.0mW_NA0.1_scattering_data.csv'

mie_grad_force = np.loadtxt(path + gradient_force_mie_file, skiprows=5)
mie_scat_force = np.loadtxt(path + scattering_force_mie_file, skiprows=5)
mie_displacement_grad = np.linspace(-6, 6, 80)
mie_displacement_scat = np.linspace(-4, 4, 80)
geom_displacement, geom_grad_force = np.loadtxt(path + gradient_force_geom_file)
geom_scat_displacement, geom_scat_force = np.loadtxt(path + scattering_force_geom_file)

mie_displacement_grad = mie_displacement_grad[(mie_grad_force < 10) & (mie_grad_force > -10)]
mie_grad_force = mie_grad_force[(mie_grad_force < 10) & (mie_grad_force > -10)]

geom_displacement *= 1e6
geom_scat_displacement *= 1e6
geom_grad_force *= 1e12
geom_scat_force *= 1e12

plt.figure()
plt.plot(mie_displacement_grad, mie_grad_force)
plt.show()
