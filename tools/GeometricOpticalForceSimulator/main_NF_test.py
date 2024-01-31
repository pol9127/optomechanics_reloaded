import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../.')
from Physics import Simulator, Particle
from Geometry import Vector


simulator = Simulator()
simulator.setup_beam_gaussian_3d(waist_position=Vector(), direction=Vector(0, 0, 1), wavelength=1565e-9, power=1.5,
                                 numerical_aperture=0.1,
                                 n_diagonal_beams=100,
                                 reflection_limit=6,
                                 far_field=False)
print('beam count: {}'.format(len(simulator.rays)))
particle = Particle(radius=10.01e-6, position=Vector())
simulator.particles.append(particle)
displacement = np.linspace(-10e-6, 10e-6, 150)
displacement_vector = Vector(1, 2, 0)
displacements, gradient_force1, gradient_force2, scattering_force = simulator.simulate_displacement(particle, displacement_vector, displacement)

um = 1e6
pN = 1e12

plt.figure(figsize=(4.5, 3))
plt.plot(displacements*um, gradient_force1*pN, displacements*um, gradient_force2*pN, displacements*um, scattering_force*pN)
plt.title('Simulation: Optical Forces near focus')
plt.xlabel('displacement [µm]')
plt.ylabel('force [pN]')
plt.legend(['gradient force x', 'gradient force y', 'scattering force'])
plt.tight_layout()
plt.show()