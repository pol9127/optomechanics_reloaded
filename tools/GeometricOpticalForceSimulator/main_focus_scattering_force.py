from Geometry import Vector
from Physics import Simulator, Particle
import numpy as np
import matplotlib.pyplot as plt

path = '../../Documentation/report/source/simulations/focus_power_vs_force_NA0.022.csv'

simulator = Simulator()

particle = Particle(radius=10.01e-6, position=Vector())
simulator.particles.append(particle)

beam_direction = Vector(0, 0, 1)
n_diagonal_beams = 60
power_step = 5e-3
start_power = 250e-3
end_power = 1
powers = np.arange(start_power, end_power+power_step, power_step)
scattering_force = list()
count = len(powers)
i = 0

for power in powers:
    simulator.rays.clear()
    simulator.setup_beam_gaussian_3d(waist_position=Vector(),
                                     direction=beam_direction,
                                     wavelength=1565e-9,
                                     power=power,
                                     numerical_aperture=0.022,
                                     n_diagonal_beams=n_diagonal_beams,
                                     reflection_limit=6,
                                     far_field=False,
                                     diverging_near_field=False)
    simulator.simulate(verbose=False)
    res = particle.get_force_along(beam_direction)
    scattering_force.append(res)
    i += 1
    print('progress: {} %'.format(round(i / count * 100, 2)))

scattering_force = np.array(scattering_force)

np.savetxt(path, np.vstack([powers, scattering_force]), delimiter=';')

plt.figure()
plt.plot(powers*1e3, scattering_force*1e12)
plt.title('Scattering force at focus')
plt.xlabel('beam power [mW]')
plt.ylabel('scattering force [pN]')
plt.show()
