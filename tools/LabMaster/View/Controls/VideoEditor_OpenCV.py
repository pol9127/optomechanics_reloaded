from PyQt5.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QSizePolicy, QLabel, QStyle, QPushButton, QSpinBox
from PyQt5.QtCore import Qt, QPoint, QSize, QRect, QEvent
from PyQt5.QtGui import QImage
from QtModularUiPack.Widgets.VideoExtensions import ImageRenderWidget, ImageLayer, ImageRectangle
from QtModularUiPack.Widgets.QtExtensions import QJumpSlider
from threading import Thread
from time import sleep, time
import numpy as np
import cv2


class VideoEditor(QFrame):
    """
    This widget allows to load a video file and zoom onto a specific ROI (region of interest)
    """

    @property
    def is_playing(self):
        """
        True if the video is currently being played.
        """
        return self._is_playing

    @property
    def duration(self):
        """
        Duration in seconds of the video clip currently loaded.
        """
        return (self.frame_count - 1) / self.fps

    @property
    def start_trim(self):
        """
        Gets the duration of the trimmed footage at the beginning of the video clip in seconds.
        """
        return self._start_trim

    @start_trim.setter
    def start_trim(self, value):
        """
        Sets how much of the beginning of the clip should be trimmed.
        :param value: Footage to cut in seconds
        """
        if not self.video_file_open:
            return
        if value > self._end_frame / self.fps:
            value = self._end_frame / self.fps
        if value < 0:
            value = 0

        self._start_trim = value
        self._start_frame = int(value * self.fps)
        self._invalidate_cache_()
        self._timeline.setMinimum(self._start_frame)
        self._current_time.setText('{:10.3f}'.format(round(self._timeline.value() / self.fps - self._start_trim, 3)))
        self._total_time.setText('{:10.3f}'.format(self._end_trim - self._start_trim))

    @property
    def end_trim(self):
        """
        Gets the time in seconds until which the video clip is shown. The rest is trimmed.
        """
        return self._end_trim

    @end_trim.setter
    def end_trim(self, value):
        """
        Sets the time in seconds until which the video clip is displayed. Everything beyond is trimmed.
        :param value: Time code in seconds until which the video clip shoud be shown
        """
        if not self.video_file_open:
            return
        if value > (self.frame_count - 1) / self.fps:
            value = (self.frame_count - 1) / self.fps
        if value < self._start_trim:
            value = self._start_trim

        self._end_trim = value
        self._end_frame = int(value * self.fps)
        self._invalidate_cache_()
        self._timeline.setMaximum(self._end_frame)
        self._current_time.setText('{:10.3f}'.format(round(self._timeline.value() / self.fps - self._start_trim, 3)))
        self._total_time.setText('{:10.3f}'.format(self._end_trim - self._start_trim))

    @property
    def fps(self):
        """
        Frames per second of the currently loaded video clip.
        """
        return self._fps

    @property
    def frame_count(self):
        """
        Total amount of frames of the currently loaded video clip.
        """
        return self._frame_count

    @property
    def current_frame(self):
        """
        Gets the current frame
        """
        return self._timeline.value()

    @current_frame.setter
    def current_frame(self, value):
        """
        Sets the current frame
        :param value: frame number
        """
        self._timeline.setValue(value)

    @property
    def start_frame(self):
        """
        Gets the start frame. (Depends on the trimming of the video clip)
        """
        return self._start_frame

    @property
    def end_frame(self):
        """
        Gets the end frame. (Depends on the trimming of the video clip)
        """
        return self._end_frame

    @property
    def video_width(self):
        """
        Horizontal resolution of the currently loaded video clip.
        """
        return self._video_width

    @property
    def video_height(self):
        """
        Vertical resolution of the currently loaded video clip.
        """
        return self._video_height

    @property
    def video_path(self):
        """
        Path of the currently loaded video file.
        """
        return self._capture_path

    @property
    def video_file_open(self):
        """
        True if a video file is currently opened in the editor.
        """
        return self._capture is not None

    @property
    def roi(self):
        """
        Returns a QRect representing the current region of interest. (If None, the entire image is the ROI)
        """
        return self._roi

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.installEventFilter(self)   # start listening for mouse events to capture ROI changes
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)    # make the editor use as much space as possible
        self.video_time = dict()    # cache for numpy arrays containing video data
        self.video_image = dict()   # cache for time data
        self._display_image = dict()    # cache for images that are displayed from the video
        self._fixed_times = None
        self._frame_count = 0
        self._fps = 0
        self._video_width = 0
        self._video_height = 0
        self._capture = None
        self._capture_path = None
        self._is_playing = False
        self._roi = None
        self._start_trim = 0
        self._end_trim = 0
        self._start_frame = 0
        self._end_frame = 0
        self._layers = list()   # layers for visualizations on top of the video footage
        self._selection_layer = ImageLayer(enabled=False)   # layer for selection indicators
        self._selection_rectangle = ImageRectangle(0, 0, filled=False, border_color=Qt.yellow)  # selection rectangle
        self._selection_layer.shapes.append(self._selection_rectangle)
        self._selection_start = None    # start point for selection rectangle
        self.time_code_changed = list()

        self._layout = QGridLayout()
        self._timeline = None
        self._current_time = None
        self._total_time = None
        self._frame_box = None
        self._play_button = None
        self._stop_button = None
        self._next_frame_button = None
        self._previous_frame_button = None
        self._box_roi_x = None
        self._box_roi_y = None
        self._box_roi_width = None
        self._box_roi_height = None
        self._accept_roi_updates_from_boxes = True
        self.caching_enabled = True

        self._play_thread = None

        self.setLayout(self._layout)
        self._setup_()

    def _to_image_space_(self, point: QPoint):
        """
        Convert a point on the editor widget in to a point in the video footage.
        :param point: point in coordinates of the widget
        :return: point in the coordinates of the video footage
        """
        control_position = self._image_control.pos()    # get the position of the video image on the editor
        control_size = self._image_control.size()   # get the size of the video image
        dx = (control_size.width() - self._image_control.image_width) / 2   # get the x offset of the footage in the image
        dy = (control_size.height() - self._image_control.image_height) / 2     # get the y offset of the footage in the image
        x = (point.x() - dx - control_position.x()) / self._image_control.image_scale_x
        y = (point.y() - dy - control_position.y()) / self._image_control.image_scale_y
        return QPoint(x, y)

    def eventFilter(self, obj, event):
        """
        Check for mouse events to edit the ROI
        :param obj: object that caused the event
        :param event: event parameters (i.e. mouse position on the widget)
        """
        if event.type() == QEvent.MouseMove:    # if the mouse was moved, update the selection size
            target = self._to_image_space_(event.pos())
            self._selection_rectangle.position = self._selection_start
            self._selection_rectangle.width = target.x() - self._selection_start.x()
            self._selection_rectangle.height = target.y() - self._selection_start.y()
            self._image_control.update()
        elif event.type() == QEvent.MouseButtonPress:   # if the left mouse button was pressed designate the point as start of the selection
            self._selection_layer.enabled = True
            target = self._to_image_space_(event.pos())
            self._selection_start = target
        elif event.type() == QEvent.MouseButtonRelease and self._selection_start is not None:   # if the button was release the the ROI
            self._selection_layer.enabled = False
            end_point = self._to_image_space_(event.pos())

            # get all possible corner points
            x1 = self._selection_start.x()
            x2 = end_point.x()
            y1 = self._selection_start.y()
            y2 = end_point.y()

            # find upper left corner of the ROI
            roi_x = x1 if x1 < x2 else x2
            roi_y = y1 if y1 < y2 else y2

            # find extent of the ROI
            roi_width = abs(x1 - x2)
            roi_height = abs(y1 - y2)

            # set the ROI if it was not just a click with no extent
            if roi_width > 0 and roi_height > 0:
                # take into account if the footage was already focused onto a previous ROI
                if self._roi is not None:
                    roi_x += self._roi.x()
                    roi_y += self._roi.y()

                # update spin box values
                self._accept_roi_updates_from_boxes = False     # disable ROI changes from the spin boxes
                self._box_roi_x.setValue(roi_x)
                self._box_roi_y.setValue(roi_y)
                self._box_roi_width.setValue(roi_width)
                self._box_roi_height.setValue(roi_height)
                self._accept_roi_updates_from_boxes = True  # enable ROI changes from the spin boxes
                self.set_roi(QRect(roi_x, roi_y, roi_width, roi_height))   # set ROI
                self._selection_start = None    # remove selection start
        return False

    def sizeHint(self):
        """
        Needed for widget to expand properly on the UI (Should be improved)
        """
        return QSize(1200, 1200)

    def load(self, path):
        """
        Load a video file from the given path
        :param path: path of the file
        """
        if self.video_file_open:    # close current video file if one was open
            self.close()

        self._capture = cv2.VideoCapture(path)      # try to get a handle of the video file
        self._capture_path = path     # remember the video file path

        self._invalidate_cache_()
        self._frame_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)         # get total video frames
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)                         # get frames per second of the video
        self._video_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))    # get width of the image
        self._video_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # get height of the image
        self._fixed_times = np.array([i / self._fps for i in range(int(self._frame_count))])     # get array with all time codes
        self._start_trim = 0    # no trimming when video is loaded
        self._start_frame = 0   # first frame is also the first frame of the video
        self._end_frame = self._frame_count - 1     # don't trim the end of the video
        self._end_trim = self.duration              # use the duration of the video as trim mark (no trimming)

        # set maximum values of the ROI spin boxes
        self._box_roi_x.setMaximum(self._video_width)
        self._box_roi_width.setMaximum(self._video_width)
        self._box_roi_y.setMaximum(self._video_height)
        self._box_roi_height.setMaximum(self._video_height)

        # update the data on the UI elements
        self._total_time.setText('{:10.3f}'.format(self._end_trim - self._start_trim))
        self._frame_box.setText('(Frame: 0000)')
        self._timeline.setValue(0)
        self._timeline.setMaximum(self._end_frame)
        self._timeline.setEnabled(True)
        self._play_button.setEnabled(True)
        self._stop_button.setEnabled(True)
        self._next_frame_button.setEnabled(True)
        self._previous_frame_button.setEnabled(True)
        self.reset_roi()

    def set_time(self, seconds):
        """
        Display the frame that is the closest to the given time
        :param seconds: time at which to display the frame in seconds
        """
        if not self.video_file_open:
            return
        frame_number = np.argmin(np.abs(self._fixed_times - self._start_trim - seconds))
        self._timeline.setValue(frame_number)

    def get_time(self, frame_number):
        """
        Return the time code at the specified frame
        :param frame_number: frame number
        :return: time code in seconds
        """
        if not self.video_file_open:    # return zero if no file is opened
            return 0
        frame_number = self._clamp_frame_number_(frame_number)
        if frame_number not in self.video_time:     # check if the data is in the cache, otherwise retrieve it from the file
            self.get_frame(frame_number)
        return self.video_time[frame_number] - self._start_trim

    def get_frame(self, frame_number):
        """
        Returns a numpy array containing the video frame at the given frame number
        :param frame_number: frame number
        :return: numpy array
        """
        if not self.video_file_open:    # return None if no video file is opened
            return None

        frame_number = self._clamp_frame_number_(frame_number)

        if self.caching_enabled:
            if frame_number in self.video_image:     # check if the data is in the cache
                return self.video_image[frame_number]

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)    # set the frame number on the video stream
        _, frame = self._capture.read()  # get the frame from the video file

        if frame is None:   # return None if no data was found
            return None

        if self._roi is not None:   # crop the frame to the ROI if one has been specified
            frame = frame[self._roi.y():self._roi.y()+self._roi.height(), self._roi.x():self._roi.x()+self._roi.width()]

        if self.caching_enabled:
            self.video_image[frame_number] = frame  # store the frame in the cache
        self.video_time[frame_number] = frame_number / self.fps     # store the time code in the cache

        # create an image that can be displayed on the image widget
        height, width, bpc = frame.shape
        bpl = bpc * width
        image = QImage(frame.data.tobytes(), width, height, bpl, QImage.Format_RGB888)
        self._display_image[frame_number] = image   # store the image in the cache
        return frame

    def _clamp_frame_number_(self, frame_number):
        """
        Clamps the given frame number to an allowed range
        :param frame_number: frame number
        :return: frame number between 0 and frame_count - 1
        """
        if frame_number < self._start_frame:
            frame_number = self._start_frame
        elif frame_number > self._end_frame:
            frame_number = self._end_frame
        return int(frame_number)

    def _invalidate_cache_(self):
        """
        Invalidate cached data
        """
        self.video_time = dict()  # reset the video numpy array cache
        self.video_image = dict()  # reset the time cache
        self._display_image = dict()  # reset the video image cache

    def set_roi(self, rect: QRect):
        """
        Sets the region of interest on the video footage.
        :param rect: rectangle representing the region of interest
        """
        self._roi = rect
        self._box_roi_x.setValue(rect.x())
        self._box_roi_y.setValue(rect.y())
        self._box_roi_width.setValue(rect.width())
        self._box_roi_height.setValue(rect.height())

        # invalidate caches that need to be filled up with the new cropped images
        self._invalidate_cache_()

        if self._frame_count > 0:
            self._display_frame_(self._timeline.value())    # update the display

    def reset_roi(self):
        """
        Reset the region of interest
        """
        self._roi = None
        self._invalidate_cache_()

        # set the full image as ROI on the spin boxes
        self._accept_roi_updates_from_boxes = False     # stop the spin boxes from updating the ROI
        self._box_roi_x.setValue(0)
        self._box_roi_y.setValue(0)
        self._box_roi_width.setValue(self._video_width)
        self._box_roi_height.setValue(self._video_height)
        self._accept_roi_updates_from_boxes = True      # re-enable the spin boxes to update the ROI

        if self._frame_count > 0:
            self._display_frame_(self._timeline.value())

    def play(self):
        """
        Start playing the video that is currently loaded.
        """
        if self.is_playing:     # do nothing if the video is already playing
            return

        self._play_thread = Thread(target=self._play_worker_, daemon=True)  # create a thread for playing the footage
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))  # set the pause icon on the play button
        self._play_thread.start()   # start playing the video

    def pause(self):
        """
        Pauses the video.
        """
        self._is_playing = False
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))   # set pause button icon to play

    def stop(self):
        """
        Stop the video and rewind.
        """
        self.pause()    # stop the video from playing
        self._timeline.setValue(0)  # return to the first frame

    def next_frame(self):
        """
        Skip one frame ahead.
        """
        current_frame = self._timeline.value()
        if current_frame < self.frame_count - 1:
            self._timeline.setValue(current_frame + 1)

    def previous_frame(self):
        """
        Skip to the previous frame.
        """
        current_frame = self._timeline.value()
        if current_frame >= 1:
            self._timeline.setValue(current_frame - 1)

    def _play_pause_(self):
        """
        Play if the video is paused or pause if the video is currently playing.
        """
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def _play_worker_(self):
        """
        Plays the video in a separate thread, with the proper speed.
        """
        try:
            self._is_playing = True     # notify the widget that the video is now being played
            frame_delay = 1 / self.fps  # calculate proper delays between frames
            current_frame = self._timeline.value()  # get first frame to play
            while self.is_playing and current_frame <= self._end_frame:     # play the video until finished or paused
                current_frame += 1
                timestamp = time()
                self._timeline.setValue(current_frame)
                remaining_delay = frame_delay - time() + timestamp  # calculate the remaining delay after the frame has been set
                if remaining_delay > 0:
                    sleep(remaining_delay)  # wait until the next frame can be displayed
        except Exception as e:
            print('error while playing. {}'.format(e))
        finally:
            self.pause()    # stop playing

    def _display_frame_(self, frame_number):
        """
        Displays the requested frame on the widget.
        :param frame_number: frame number
        """
        if not self.video_file_open:    # do nothing if no video file is open
            return

        frame_number = self._clamp_frame_number_(frame_number)  # get proper frame number

        if frame_number not in self._display_image:     # check if the data is in the cache, otherwise retrieve it from the file
            self.get_frame(frame_number)

        if frame_number not in self._display_image:
            return

        try:
            self._image_control.set_image(self._display_image[frame_number])    # set the image on the image control
        except:
            print('offending frame number: {}'.format(frame_number))

        # update the UI
        self._current_time.setText('{:10.3f}'.format(round(self.video_time[frame_number] - self._start_trim, 3)))
        self._frame_box.setText('(frame: {:04})'.format(int(frame_number)))

        # send event about frame change
        for callback in self.time_code_changed:
            callback(self.video_time[frame_number] - self._start_trim)

    def _roi_box_value_changed_(self, *args):
        """
        Callback for changes made in the ROI spin boxes. Adjust the ROI accordingly.
        """
        if self._accept_roi_updates_from_boxes:
            roi_x = self._box_roi_x.value()
            roi_y = self._box_roi_y.value()
            roi_width = self._box_roi_width.value()
            roi_height = self._box_roi_height.value()
            self.set_roi(QRect(roi_x, roi_y, roi_width, roi_height))

    def close(self):
        """
        Closes the video file which is  currently opened.
        """
        if self._capture is not None:
            self._capture.release()     # release the stream
            self._capture = None        # se6 the stream to None
            self._capture_path = None   # reset the capture past
            self._fixed_times = None
            self._frame_count = 0       # set the frame count to zero
            self._fps = 0               # set the frames per second to zero
            self._timeline.setValue(0)              # set the timeline to zero
            self._timeline.setEnabled(False)        # disable the timeline
            self._play_button.setEnabled(False)     # disable the play button
            self._stop_button.setEnabled(False)     # disable the stop button
            self._next_frame_button.setEnabled(True)       # disable the skip frame button
            self._previous_frame_button.setEnabled(True)    # disable the previous frame button
            self._current_time.setText('0.000')     # set the current time code to zero
            self._total_time.setText('0.000')       # set the total time to zero
            self._frame_box.setText('(frame: 0000)')    # set the current frame to zero
            self._box_roi_x.setMaximum(0)       # set the ROI maximum to zero
            self._box_roi_width.setMaximum(0)   # set the ROI maximum to zero
            self._box_roi_y.setMaximum(0)       # set the ROI maximum to zero
            self._box_roi_height.setMaximum(0)  # set the ROI maximum to zero
            self.reset_roi()    # reset the ROI (this also empties the cache)

    def _setup_(self):
        self._image_control = ImageRenderWidget()
        self._image_control.overlay_layers.append(self._selection_layer)
        self._layout.addWidget(self._image_control, 0, 0, 1, 10, Qt.AlignCenter)

        self._timeline = QJumpSlider(Qt.Horizontal)
        self._timeline.setEnabled(False)
        self._timeline.valueChanged.connect(self._display_frame_)
        self._layout.addWidget(self._timeline, 1, 4)

        self._current_time = QLabel('0.000')
        self._total_time = QLabel('0.000')
        self._frame_box = QLabel('(Frame: 0000)')
        self._layout.addWidget(self._current_time, 1, 5)
        self._layout.addWidget(QLabel('/'), 1, 6)
        self._layout.addWidget(self._total_time, 1, 7)
        self._layout.addWidget(QLabel(' s'), 1, 8)
        self._layout.addWidget(self._frame_box, 1, 9)

        self._play_button = QPushButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._play_button.setEnabled(False)
        self._play_button.clicked.connect(self._play_pause_)
        self._layout.addWidget(self._play_button, 1, 0)

        self._stop_button = QPushButton()
        self._stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self._stop_button.setEnabled(False)
        self._stop_button.clicked.connect(self.stop)
        self._layout.addWidget(self._stop_button, 1, 1)

        self._next_frame_button = QPushButton()
        self._next_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self._next_frame_button.setEnabled(False)
        self._next_frame_button.clicked.connect(self.next_frame)
        self._layout.addWidget(self._next_frame_button, 1, 3)

        self._previous_frame_button = QPushButton()
        self._previous_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self._previous_frame_button.setEnabled(False)
        self._previous_frame_button.clicked.connect(self.previous_frame)
        self._layout.addWidget(self._previous_frame_button, 1, 2)

        roi_frame = QFrame()
        roi_layout = QHBoxLayout()
        roi_frame.setLayout(roi_layout)
        roi_frame.setFixedHeight(38)
        roi_layout.addWidget(QLabel('ROI: ['))
        roi_layout.addWidget(QLabel('x:'))
        self._box_roi_x = QSpinBox()
        roi_layout.addWidget(self._box_roi_x)
        roi_layout.addWidget(QLabel('y:'))
        self._box_roi_y = QSpinBox()
        roi_layout.addWidget(self._box_roi_y)
        roi_layout.addWidget(QLabel('width:'))
        self._box_roi_width = QSpinBox()
        roi_layout.addWidget(self._box_roi_width)
        roi_layout.addWidget(QLabel('height:'))
        self._box_roi_height = QSpinBox()
        roi_layout.addWidget(self._box_roi_height)
        roi_layout.addWidget(QLabel(']'))
        roi_reset_button = QPushButton('Reset')
        roi_reset_button.clicked.connect(self.reset_roi)
        roi_layout.addWidget(roi_reset_button)
        self._box_roi_x.valueChanged.connect(self._roi_box_value_changed_)
        self._box_roi_y.valueChanged.connect(self._roi_box_value_changed_)
        self._box_roi_width.valueChanged.connect(self._roi_box_value_changed_)
        self._box_roi_height.valueChanged.connect(self._roi_box_value_changed_)
        self._layout.addWidget(roi_frame, 2, 0, 1, 9, Qt.AlignLeft)
