from Geometry import Vector
from Physics import Simulator
from Setups import ParticleTransverseDisplacementSetup, ParticleAlongBeamDisplacementSetup
from time import time
import numpy as np


calculate_forces = True

simulator = Simulator()
#beam_power = 1.40871   # watts for 6 A
#beam_power = 1.14775    # for 5 A
beam_power = 0.5  # watts

simulate_gradient_force = True

if simulate_gradient_force:
    chosen_setup = ParticleTransverseDisplacementSetup
    displacement = np.linspace(-25e-6, 25e-6, 100)
else:
    chosen_setup = ParticleAlongBeamDisplacementSetup
    displacement = np.linspace(50e-6, 1800e-6, 100)

setup = chosen_setup(simulator=simulator,
                     displacement=displacement,
                     particle_radius=10.01e-6,
                     particle_refractive_index=1.42,
                     beam_power=beam_power,
                     waist_position=Vector(0, 0, 0),
                     beam_direction=Vector(0, 0, 1),
                     far_field=False,
                     diverging_near_field=False)
#setup.numerical_aperture = 0.055       # old NA=0.06
setup.numerical_aperture = 0.1
setup.n_diagonal_simulation_beams = 60
setup.reflection_limit = 6
setup.prepare()
print('Number of rays: {}'.format(len(simulator.rays)))
#simulator.particles.remove(setup.particle)

#simulator.beams.append(LightBeam(wavelength=1565e-9, power=1, reflection_limit=100, origin=Vector(-20e-6, 0, 0), direction=Vector(1, 0.4, 0)))

particle = setup.particle

plot_name = 'axial_force_{}mW_NA{}'.format(beam_power*1e3, setup.numerical_aperture)

if simulate_gradient_force:
    ## OLD REFRACTIVE INDEX n=1.59
    #particle.position = Vector(772.3434491974568*1e-6, 0, 0)
    #particle.position = Vector(0, 0, 534.43156801621178 * 1e-6)
    #particle.position = Vector(0, 0, 319.6170662176055 * 1e-6)
    #particle.position = Vector(0, 0, 183.00446670071568 * 1e-6) # -> 230mW NA=0.06
    #particle.position = Vector(0, 0, 253.8160480510772 * 1e-6)  # -> 350mW NA=0.06
    #particle.position = Vector(0, 0, 509.47100970501407 * 1e-6)  # -> 500mW NA=0.02
    #particle.position = Vector(0, 0, 207.34023022883068 * 1e-6)  # -> 415mW NA=0.02
    #particle.position = Vector(0, 0, 480.1496662574735 * 1e-6)  # -> 500mW NA=0.04
    #particle.position = Vector(0, 0, 192.53190379730415 * 1e-6)  # -> 225mW NA=0.04

    ## NEW REFRACTIVE INDEX n=1.42
    #particle.position = Vector(0, 0, 284.3042731793606 * 1e-6)  # -> 500mW NA=0.055
    #particle.position = Vector(0, 0, 71.4661811588 * 1e-6)  # -> 250mW NA=0.055
    #particle.position = Vector(0, 0, 436.841474806608 * 1e-6)  # -> 500mW NA=0.036

    plot_name = 'radial_force_{}mW_NA{}'.format(beam_power*1e3, setup.numerical_aperture)

#plot_name = 'Gradient_force_5A_equilibrium'
#plot_name = 'gradient_forces_D20um_at_scattering_equilibrium'
save = False

t_start = time()
if calculate_forces:
    if simulate_gradient_force:
        setup.run(show_plots=True, show_grid=True,
                  save_path='../../Documentation/report/source/simulations/' + plot_name, save_png=save, save_svg=save,
                  save_csv=save, show_gradient_force=True, show_scattering_force=True)
    else:
        setup.run(show_plots=True, show_grid=True,
                  save_path='../../Documentation/report/source/simulations/' + plot_name, save_png=save, save_svg=save,
                  save_csv=save)

t_duration = time() - t_start
print('Time elapsed during simulation: {} s'.format(t_duration))

#particle.position = Vector(0, 1, 0) * 6e-6
#simulator.simulate(verbose=True)
"""simulator.visualize_2d(show_reflected_beams=True,
                       x_axis=Vector(1, 0, 0),
                       y_axis=Vector(0, 0, 1),
                       title='particle displaced in x direction',
                       use_intensity_for_visibility=True,
                       xlanel='x',
                       ylabel='z',
                       center_particle=particle)"""

#particle.position = Vector(0, 0, displacement[0])
#particle.position -= Vector(20e-6, 0, 0)
#particle.position = Vector(0, 0, 80e-6)
"""simulator.simulate(verbose=True)
simulator.visualize_2d(show_reflected_beams=True,
                       x_axis=Vector(1, 0, 0),
                       y_axis=Vector(0, 0, 1),
                       title='Simulation: Particle partially in trap',
                       xlanel='x',
                       ylabel='z',
                       center_particle=particle,
                       figzise=(1, 1))
                       #view_box=[-50e-6, 40e-6, 50e-6, 120e-6])"""

#input('press any key to exit...')
import matplotlib.pyplot as plt
plt.show()
