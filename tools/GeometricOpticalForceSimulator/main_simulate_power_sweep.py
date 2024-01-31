from Geometry import Vector
from Physics import Simulator
from Setups import PowerSweepSetup
import numpy as np

path = '../VideoAnalyzerTestProjects/simulation_280mW-1W_n1.42_high_res'

simulator = Simulator()
displacement = np.linspace(100e-6, 1800e-6)
beam_powers = np.flip(np.linspace(280e-3, 1000e-3, 100))

setup = PowerSweepSetup(simulator=simulator,
                        displacement=displacement,
                        particle_radius=10.01e-6,
                        particle_refractive_index=1.42,  # based on data-sheet
                        beam_powers=beam_powers,
                        numerical_aperture=0.036,
                        waist_position=Vector(0, 0, 0),
                        beam_direction=Vector(0, 0, 1),
                        n_diagonal_simulation_beams=80,
                        far_field=True,
                        diverging_near_field=True)

setup.prepare()

print('beams per simulation: {}'.format(len(simulator.rays)))

setup.run(save_csv=True, save_path=path)
