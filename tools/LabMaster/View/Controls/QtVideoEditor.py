from QtModularUiPack.Widgets import EmptyFrame
from QtModularUiPack.Widgets.QtExtensions import QJumpSlider
from QtModularUiPack.Widgets.VideoExtensions import ImageRenderWidget, ImageLayer, ImageRectangle, VideoFrameGrabber
from QtModularUiPack.ViewModels import BaseViewModel
from PyQt5.QtCore import Qt, QUrl, QEvent, QRect, QPoint
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QStyle, QVBoxLayout
from PyQt5.QtWidgets import QWidget, QPushButton
import os


class QtVideoPlayer(EmptyFrame, BaseViewModel):

    name = 'video player'

    @property
    def play_button_icon(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            return self.style().standardIcon(QStyle.SP_MediaPause)
        else:
            return self.style().standardIcon(QStyle.SP_MediaPlay)

    @property
    def video_available(self):
        return self._video_available

    @video_available.setter
    def video_available(self, value):
        self._video_available = value
        self.notify_change('video_available')

    @property
    def video_start(self):
        return self._video_start

    @video_start.setter
    def video_start(self, value):
        self._video_start = value
        self._start_frame = self._time_code_to_frame_number_(value)
        self.notify_change('video_start')

    @property
    def video_end(self):
        return self._video_end

    @video_end.setter
    def video_end(self, value):
        self._video_end = value
        self._end_frame = self._time_code_to_frame_number_(value)
        self.notify_change('video_end')
        self.notify_change('video_duration')

    @property
    def video_duration(self):
        return '{:10.3f}'.format(self.video_end / 1000)

    @property
    def time_code(self):
        return self._time_code

    @time_code.setter
    def time_code(self, value):
        self._time_code = value
        self._frame_number = self._time_code_to_frame_number_(value)
        self.media_player.setPosition(value)
        self.notify_change('time_code')
        self.notify_change('frame_number')
        self.notify_change('time')

    @property
    def time(self):
        return '{:10.3f}'.format(self.time_code / 1000)

    @property
    def fps(self):
        return self._fps

    @property
    def frame_number(self):
        return self._frame_number

    @frame_number.setter
    def frame_number(self, value):
        self._frame_number = value
        self._time_code = self._frame_number_to_time_code(value)
        self.media_player.setPosition(self._time_code)
        self.notify_change('frame_number')
        self.notify_change('time_code')
        self.notify_change('time')

    @property
    def file_path(self):
        return self._file_path

    def open(self, path):
        if os.path.exists(path):
            self._file_path = path
            self.video_available = True
            self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(path)))
            self.media_player.pause()

    def play(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

        self.notify_change('play_button_icon')

    def stop(self):
        self.media_player.pause()
        self.time_code = 0

    def next_frame(self):
        if self.frame_number < self._end_frame:
            self.frame_number += 1

    def previous_frame(self):
        if self.frame_number > self._start_frame:
            self.frame_number -= 1

    def __init__(self, *args, **kwargs):
        EmptyFrame.__init__(self, *args, **kwargs)
        BaseViewModel.__init__(self)
        self.installEventFilter(self)  # start listening for mouse events to capture ROI changes
        self._layers = list()  # layers for visualizations on top of the video footage
        self._selection_layer = ImageLayer(enabled=False)  # layer for selection indicators
        self._selection_rectangle = ImageRectangle(0, 0, filled=False, border_color=Qt.yellow)  # selection rectangle
        self._selection_layer.shapes.append(self._selection_rectangle)
        self._selection_start = None  # start point for selection rectangle
        self._accept_roi_updates_from_boxes = True
        self._roi = None
        self.data_context = self
        self._fps = 30
        self._video_available = False
        self._video_start = 0
        self._video_end = 0
        self._start_frame = 0
        self._end_frame = 0
        self._time_code = 0
        self._frame_number = 0
        self._file_path = None
        self._setup_()
        self.open('Z:/shared/Master Thesis/Experiments/TrappingExperiment/Trapping_18.7.2019/power_walk_1W-350mW_1Bar.avi')

    def _to_image_space_(self, point: QPoint):
        """
        Convert a point on the editor widget in to a point in the video footage.
        :param point: point in coordinates of the widget
        :return: point in the coordinates of the video footage
        """
        control_position = self.video_image.pos()    # get the position of the video image on the editor
        control_size = self.video_image.size()   # get the size of the video image
        dx = (control_size.width() - self.video_image.image_width) / 2   # get the x offset of the footage in the image
        dy = (control_size.height() - self.video_image.image_height) / 2     # get the y offset of the footage in the image
        x = (point.x() - dx - control_position.x()) / self.video_image.image_scale_x
        y = (point.y() - dy - control_position.y()) / self.video_image.image_scale_y
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
            self.video_image.update()
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

    def _time_code_to_frame_number_(self, time_code):
        return int(time_code / 1000 * self.fps)

    def _frame_number_to_time_code(self, f_number):
        return int(f_number / self.fps * 1000)

    def _process_frame_(self, frame):
        roi = frame.copy(QRect(100, 100, 100, 100))
        self.video_image.set_image(roi)

    def _setup_(self):
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.media_player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self.media_player.metaDataAvailableChanged.connect(self._meta_data_changed_)
        self.video_image = ImageRenderWidget()
        self.video_image.overlay_layers.append(self._selection_layer)
        self.grabber = VideoFrameGrabber(self)
        self.grabber.frameAvailable.connect(self._process_frame_)

        play_button = QPushButton()
        play_button.clicked.connect(self.play)
        stop_button = QPushButton()
        stop_button.clicked.connect(self.stop)
        stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        next_frame_button = QPushButton()
        next_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        next_frame_button.clicked.connect(self.next_frame)
        previous_frame_button = QPushButton()
        previous_frame_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        previous_frame_button.clicked.connect(self.previous_frame)

        self.bindings.set_binding('media', self.media_player, 'setMedia')
        self.bindings.set_binding('video_available', play_button, 'setEnabled')
        self.bindings.set_binding('video_available', stop_button, 'setEnabled')
        self.bindings.set_binding('video_available', next_frame_button, 'setEnabled')
        self.bindings.set_binding('video_available', previous_frame_button, 'setEnabled')
        self.bindings.set_binding('play_button_icon', play_button, 'setIcon')

        position_slider = QJumpSlider(Qt.Horizontal)
        self.bindings.set_binding('video_start', position_slider, 'setMinimum')
        self.bindings.set_binding('video_end', position_slider, 'setMaximum')
        self.bindings.set_binding('time_code', position_slider, 'setValue')

        control = QWidget()
        control.setFixedHeight(36)
        control_layout = QHBoxLayout()
        control.setLayout(control_layout)
        control_layout.addWidget(play_button)
        control_layout.addWidget(stop_button)
        control_layout.addWidget(previous_frame_button)
        control_layout.addWidget(next_frame_button)
        control_layout.addWidget(position_slider)
        control_layout.addWidget(self.add_widget(QLabel(), 'time', 'setText', width=50))
        control_layout.addWidget(QLabel('/'))
        control_layout.addWidget(self.add_widget(QLabel(), 'video_duration', 'setText', width=50))
        control_layout.addWidget(QLabel('s'))
        control_layout.addWidget(QLabel(' (frame: '))
        control_layout.addWidget(self.add_widget(QLabel(), 'frame_number', 'setText', width=40))
        control_layout.addWidget(QLabel(')'))

        main_layout.addWidget(self.video_image)
        main_layout.addWidget(control)

        self.media_player.setVideoOutput(self.grabber)
        self.media_player.setNotifyInterval(1)
        self.media_player.stateChanged.connect(self._media_state_changed_)
        self.media_player.positionChanged.connect(self._position_changed_)
        self.media_player.durationChanged.connect(self._duration_changed_)

    def _meta_data_changed_(self, available):
        if self.media_player.isMetaDataAvailable():
            self._fps = self.media_player.metaData('VideoFrameRate')
            self.video_start = 0
            self.video_end = self.media_player.metaData('Duration')
            self.notify_change('fps')

    def _media_state_changed_(self, state):
        self.notify_change('play_button_icon')

    def _position_changed_(self, position):
        self._time_code = position
        self._frame_number = self._time_code_to_frame_number_(position)
        self.notify_change('time_code')
        self.notify_change('frame_number')
        self.notify_change('time')

    def _duration_changed_(self, duration):
        self.video_end = duration


if __name__ == '__main__':
    QtVideoPlayer.standalone_application()
