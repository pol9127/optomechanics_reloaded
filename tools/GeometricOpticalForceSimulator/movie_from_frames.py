import cv2
from os import remove


path = 'c:/users/Dominik Werner/polybox/Master Thesis/Documentation/Movies/movie/Movies/'
video_name = 'movie.avi'
frame_count = 150
fps = 30
video = None

for i in range(1, frame_count):
    frame_path = '{}frame{}.png'.format(path, i)
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape
    if video is None:
        video = cv2.VideoWriter(path + video_name, 0, fps, (width, height))
    video.write(frame)
    remove(frame_path)
    print('{} %'.format(((i+1) / frame_count) * 100))

if video is not None:
    video.release()
print('done.')