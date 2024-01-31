from Physics import Simulator, Particle
from Geometry import Vector
from os import remove
import numpy as np
import cv2
import gc


particle = Particle(position=Vector(0, 0, 772.3434491974568 * 1e-6), radius=10.01e-6, n=1.59)

simulator = Simulator()
simulator.particles.append(particle)
simulator.setup_beam_gaussian_3d(Vector(), Vector(0, 0, 1),
                                 power=0.5,
                                 numerical_aperture=0.1,
                                 n_diagonal_beams=50,
                                 reflection_limit=6)

path = 'c:/users/Dominik Werner/polybox/Master Thesis/Documentation/Movies/'
video_name = 'movie.avi'

fps = 30
start = 0
end = 3
t = np.linspace(start, end, end * fps)
f = 1. / 12.
x_coord = np.cos(2*np.pi*f*t)
y_coord = np.sin(2*np.pi*f*t)
video = None

for i in range(len(t)):
    frame_path = '{}frame{}.png'.format(path, i)
    simulator.simulate(verbose=False)
    simulator.visualize_2d(show_reflected_beams=True,
                           x_axis=Vector(y_coord[i], x_coord[i], 0),
                           y_axis=Vector(0, y_coord[i], x_coord[i]),
                           title='particle at zero displacement',
                           xlabel='',
                           ylabel='',
                           center_particle=particle,
                           figzise=(1, 1),
                           show=False,
                           save_path=frame_path)
    gc.collect()
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    if video is None:
        video = cv2.VideoWriter(path + video_name, 0, fps, (width, height))
    video.write(frame)
    remove(frame_path)
    print('{} %'.format(((i+1) / len(t)) * 100))
if video is not None:
    video.release()
cv2.destroyAllWindows()
