import matplotlib.pyplot as plt
import numpy as np

path = '../../Documentation/report/source/simulations/focus_power_vs_force_NA0.022.csv'

powers, scattering_force = np.loadtxt(path, delimiter=';')

proportionality_factor = (scattering_force[-1] - scattering_force[0]) / (powers[-1] - powers[0])
print('proportionality factor: {} pN/W'.format(proportionality_factor*1e12))
print(proportionality_factor)

plt.figure()
plt.plot(powers*1e3, scattering_force*1e12)
plt.title('Scattering force at focus')
plt.xlabel('beam power [mW]')
plt.ylabel('scattering force [pN]')
plt.show()
