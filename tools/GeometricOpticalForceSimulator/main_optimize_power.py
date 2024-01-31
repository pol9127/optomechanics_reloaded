from Geometry import Vector
from Physics import Simulator
from Setups import OptimizeRadialForceSetup
import numpy as np


simulator = Simulator()

scattering_force_displacement = np.linspace(50e-6, 1000e-6, 100)
gradient_force_displacement = np.linspace(-25e-6, 25e-6, 100)

save = True
data_path = None
if save:
    data_path = '../../experiments/Simulations/power_optimization_NA0.036_01.08.2019.csv'

setup = OptimizeRadialForceSetup(simulator, 0.400, 0.01, 0.005, scattering_force_displacement, gradient_force_displacement,
                                 numerical_aperture=0.036,
                                 particle_radius=10.01e-6,
                                 particle_refractive_index=1.42,
                                 waist_position=Vector(),
                                 beam_direction=Vector(1, 0, 0),
                                 gradient_force_direction=Vector(0, 1, 0),
                                 far_field=False,
                                 diverging_near_field=True,
                                 data_path=data_path)

setup.n_diagonal_simulation_beams = 80

setup.run()
