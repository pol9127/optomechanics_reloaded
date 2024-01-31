import numpy as np
import matplotlib.pyplot as plt


data_beam_count = np.loadtxt('force_vs_beam_count.csv', delimiter=';', skiprows=1)

n_diag = data_beam_count[:, 0]
n = data_beam_count[:, 1]
grad_y = data_beam_count[:, 2]
grad_z = data_beam_count[:, 3]
scat_x = data_beam_count[:, 4]

plt.figure()
plt.semilogx(n, grad_y, n, grad_z)
plt.title('simulated gradient forces on displaced particle')
plt.xlabel('number of simulated beams')
plt.ylabel('force [pN]')
plt.legend(['gradient force in y', 'gradient force in z'])
plt.show()

plt.figure()
plt.semilogx(n, scat_x)
plt.title('simulated scattering force on displaced particle')
plt.xlabel('number of simulated beams')
plt.ylabel('force [pN]')
plt.show()

data_reflection_limit = np.loadtxt('force_vs_reflection_count.csv', delimiter=';', skiprows=1)
n = data_reflection_limit[:, 0]
grad_y = data_reflection_limit[:, 1]
grad_z = data_reflection_limit[:, 2]
scat_x = data_reflection_limit[:, 3]

plt.figure()
plt.plot(n, grad_y, n, grad_z)
plt.title('simulated gradient forces on displaced particle')
plt.xlabel('number of allowed reflections')
plt.ylabel('force [pN]')
plt.legend(['gradient force in y', 'gradient force in z'])
plt.show()

plt.figure()
plt.plot(n, scat_x)
plt.title('simulated scattering force on displaced particle')
plt.xlabel('number of allowed reflections')
plt.ylabel('force [pN]')
plt.show()

