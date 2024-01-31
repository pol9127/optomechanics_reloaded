from Physics import Simulator
from Geometry import Vector
from Setups import ParticleTransverseDisplacementSetup
import numpy as np


calculate_forces = True

simulator = Simulator()
beam_power = 0.230  # watts

chosen_setup = ParticleTransverseDisplacementSetup
displacement = np.linspace(-15e-6, 15e-6, 120)

setup = ParticleTransverseDisplacementSetup(simulator=simulator,
                                            displacement=displacement,
                                            particle_radius=10.01e-6,
                                            beam_power=beam_power,
                                            waist_position=Vector(0, 0, 0),
                                            beam_direction=Vector(0, 0, 1),
                                            far_field=False,
                                            diverging_near_field=False)
setup.numerical_aperture = 0.025
setup.n_diagonal_simulation_beams = 60
setup.reflection_limit = 6
setup.prepare()
print('Number of beams: {}'.format(len(simulator.rays)))
#simulator.particles.remove(setup.particle)

#simulator.beams.append(LightBeam(wavelength=1565e-9, power=1, reflection_limit=100, origin=Vector(-20e-6, 0, 0), direction=Vector(1, 0.4, 0)))

particle = setup.particle

#particle.position = Vector(0, 0, 33.6 * 1e-6)
plot_name = 'scattering_anomaly_at_focus_230mW'

save = True
setup.run(show_plots=True, show_grid=True, save_path='../../Documentation/report/source/simulations/'+plot_name, save_png=save, save_svg=save, save_csv=save)
