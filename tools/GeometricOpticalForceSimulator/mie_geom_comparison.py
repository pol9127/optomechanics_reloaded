import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


path = '../../Documentation/report/source/simulations/'
gradient_force_mie_file = 'octave_d20_gradient_force.txt'
scattering_force_mie_file = 'octave_d20_scattering_force.txt'
displacement_mie_file = 'octave_d20_displacement.txt'
gradient_force_geom_file = 'gradient_forces_D20um_grad1_data.csv'
#scattering_force_geom_file = 'gradient_forces_D20um_scattering_data.csv'
scattering_force_geom_file = 'along_beam_displacement_D20um_data.csv'
#gradient_force_geom_file = 'radial_force_500.0mW_NA0.1_gradient_data.csv'
#scattering_force_geom_file = 'radial_force_500.0mW_NA0.1_scattering_data.csv'

mie_grad_force = np.loadtxt(path + gradient_force_mie_file, skiprows=5)
mie_scat_force = np.loadtxt(path + scattering_force_mie_file, skiprows=5)
mie_displacement = np.loadtxt(path + displacement_mie_file, skiprows=5)
geom_displacement, geom_grad_force = np.loadtxt(path + gradient_force_geom_file)
geom_scat_displacement, geom_scat_force = np.loadtxt(path + scattering_force_geom_file)

geom_displacement *= 1e6
geom_scat_displacement *= 1e6
geom_grad_force *= 1e12
geom_scat_force *= 1e12

func = lambda x, a, b, c, d, e, f, g, h: a*x**7 + b*x**6 + c*x**5 + d*x**4 + e*x**3 + f*x**2 + g*x + h
#func = lambda x, a, b, c, d, e, f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f
#func = lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d
popt_grad_mie, _ = curve_fit(func, mie_displacement, mie_grad_force)
popt_grad_geom, _ = curve_fit(func, geom_displacement, geom_grad_force)

x = np.linspace(-10, 10)
error_grad = np.abs(func(x, *popt_grad_mie) - func(x, *popt_grad_geom))

fig = plt.figure(figsize=(3.5, 3))
plt.plot(mie_displacement, mie_grad_force, geom_displacement, geom_grad_force, x, error_grad, 'r--')#,
         #mie_displacement, func(mie_displacement, *popt_mie), '--',
         #geom_displacement, func(geom_displacement, *popt_geom), '--')
plt.xlabel('displacement [µm]')
plt.ylabel('radial force [pN]')
plt.title('radial force')
plt.legend(['mie scattering', 'geometric', 'error'])
plt.grid(True)
plt.tight_layout()
fig.savefig(path + 'mie_geom_rad_comparison.png')
fig.savefig(path + 'mie_geom_rad_comparison.svg')
#plt.show()

popt_scat_mie, _ = curve_fit(func, mie_displacement, mie_scat_force)
popt_scat_geom, _ = curve_fit(func, geom_scat_displacement, geom_scat_force)

error_scat = np.abs(func(x, *popt_scat_mie) - func(x, *popt_scat_geom))

fig = plt.figure(figsize=(3.5, 3))
plt.plot(mie_displacement, mie_scat_force, geom_scat_displacement, geom_scat_force, x, error_scat, 'r--')#,
         #mie_displacement, func(mie_displacement, *popt_scat_mie), '--',
         #geom_scat_displacement, func(geom_scat_displacement, *popt_scat_geom), '--')
plt.xlabel('displacement [µm]')
plt.ylabel('axial force [pN]')
plt.title('axial force')
plt.legend(['mie scattering', 'geometric', 'error'])
plt.grid(True)
plt.tight_layout()
fig.savefig(path + 'mie_geom_ax_comparison.png')
fig.savefig(path + 'mie_geom_ax_comparison.svg')
plt.show()
