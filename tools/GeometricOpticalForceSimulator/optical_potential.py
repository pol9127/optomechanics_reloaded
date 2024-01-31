import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


path = '../../Documentation/report/source/simulations/'
scattering_file = 'scattering_force_500mW_data.csv'
gradient_file = 'gradient_force_500mW_grad1_data.csv'

ds, fs = np.loadtxt(path+scattering_file)
dg, fg = np.loadtxt(path+gradient_file)

# particle properties
particle_diameter = 20.02e-6
particle_density = 1850
mass = particle_density * 1/6 * np.pi * particle_diameter**3

# forces
g = 9.81
gravity = mass * g

f_scat = lambda x, x0, gamma: np.divide(1, np.pi * (1 + np.power(x - x0, 2)*1/gamma**2))
popt_scat, pvar_scat = curve_fit(f_scat, ds, fs, p0=(0, 1e-8))
print(popt_scat)
scat_fitted = f_scat(ds, *popt_scat)

#x0 = 0
#y0 = 0
f_grad = lambda x, sigma, x0, y0, scale: -scale*(x-x0)/(np.sqrt(2*np.pi)*sigma**3)*np.exp(-(x-x0)**2/(2*sigma**2))+y0
popt_grad, pvar_grad = curve_fit(f_grad, dg, fg, p0=(20e-6, 0, 0, 1e-21))
print(popt_grad)
grad_fitted = f_grad(dg, *popt_grad)

plt.figure()
plt.plot(ds, fs, ds, scat_fitted, 'r--')

plt.figure()
plt.plot(dg, fg, dg, grad_fitted, 'r--')


# potential
U_scat = lambda x, x0, gamma: -1/np.pi*np.arctan((x - x0)/gamma) + gravity * x
U_grad = lambda x, sigma, x0, y0, scale: -scale/(np.sqrt(2*np.pi)*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2)) + y0*x

ds_ = np.linspace(0, 1e-1, 1000)
U_scat_calc = U_scat(ds, *popt_scat)
U_grad_calc = U_grad(dg, *popt_grad)
U_sact0 = np.min(U_scat_calc)
U_grad0 = np.min(U_grad_calc)

plt.figure()
plt.plot(ds*1e6, (U_scat_calc - U_sact0)*1e6)
plt.grid(True)
plt.xlabel('z [µm]')
plt.ylabel('potential [µJ]')

plt.figure()
plt.plot(dg*1e6, (U_grad_calc - U_grad0)*1e12)
plt.grid(True)
plt.xlabel('x [µm]')
plt.ylabel('potential [pJ]')

plt.show()