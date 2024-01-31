from Geometry import Vector
from Physics import Simulator, Particle
import numpy as np

simulator = Simulator()
particle = Particle(position=Vector(), radius=10.01e-6)
simulator.particles.append(particle)


path = '../VideoAnalyzerTestProjects/'
simulation_file = 'simulation_280mW-1W_high_res_data.csv'
powers, z_eq = np.loadtxt(path+simulation_file).T

# sim file evaluation
do_sim_file_evaluation = True
n_diagonal_beams = 5
numerical_aperture = 0.04
wavelength = 1565e-9
min_power = 330*1e-3
max_power = 440*1e-3

z_eq = z_eq[(powers > min_power) & (powers < max_power)]
powers = powers[(powers > min_power) & (powers < max_power)]

# general simulation
fixed_power = 400e-3
position_sweep = np.flip(np.linspace(50e-6, 400e-6, 10))
store_frames = False
frame_directory = './animation/'

view_box_width = 50e-6


def handle_sim(position, power, frame_name=None):
    print('power: {} mW, position: {} µm'.format(power * 1e3, position * 1e6))
    simulator.rays.clear()

    particle.position = Vector(0, 0, position)
    simulator.setup_beam_gaussian_3d(waist_position=Vector(),
                                     direction=Vector(0, 0, 1),
                                     wavelength=wavelength,
                                     power=power,
                                     numerical_aperture=numerical_aperture,
                                     n_diagonal_beams=n_diagonal_beams,
                                     reflection_limit=6,
                                     far_field=True,
                                     diverging_near_field=True)

    simulator.simulate(verbose=False)
    print('generating visuals...')

    roi_x = -view_box_width / 2
    roi_y = position - view_box_width / 2

    frame_store_path = None
    if store_frames:
        frame_store_path = frame_directory + frame_name

    simulator.visualize_2d(title='Particle at {} µm (Power: {} mW)'.format(round(position * 1e6, 2), round(power * 1e3), 2),
                           xlabel='radial direction',
                           ylabel='axial direction',
                           view_box=[roi_x, roi_y, view_box_width, view_box_width],
                           show=not store_frames,
                           save_path=frame_store_path,
                           show_force_field=False,
                           field_grid_size=2e-6,
                           particle_color='paleturquoise',
                           show_beams=True,
                           figzise=(3.5, 3))


if do_sim_file_evaluation:
    for i in range(len(powers)):
        handle_sim(z_eq[i], powers[i])
else:
    i = 0
    for pos in position_sweep:
        handle_sim(pos, fixed_power, 'frame_{}.png'.format(i))
        i += 1