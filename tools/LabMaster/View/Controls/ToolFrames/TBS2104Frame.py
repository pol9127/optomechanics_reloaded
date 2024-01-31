from QtModularUiPack.Widgets import EmptyFrame, PyGraphWidget, PlotWidget
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import QMetaObject
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from ViewModel.ToolViewModels.TBS2104ViewModel import TBS2104ViewModel, CHANNELS, COUPLING_MODES, TRIGGER_MODES


BACKEND_MATPLOTLIB = 'matplotlib'
BACKEND_PYQTGRAPH = 'pyqtgraph'


class TBS2104Frame(QMetaObject, EmptyFrame):
    """
    This frame allows the control of the TBS2104 Oscilloscope
    """

    name = 'TBS2104 Oscilloscope'
    title_font = QFont('Arial', 10)
    backend = BACKEND_PYQTGRAPH

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.data_context = TBS2104ViewModel()
        self.data_context.widget = self
        self.data_context.property_changed.connect(self._on_property_changed_update_plot_)
        self.layout_parent = QHBoxLayout()
        self.layout_parent.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout_parent)
        self.progress_bar = None
        self._setup_()

    def show_progress(self):
        if self.progress_bar is not None:
            self.progress_bar.hide()
            self.progress_bar.destroy()

        self.progress_bar = QProgressDialog("Saving waveform to file...", None, 0, 100, self)
        self.progress_bar.setWindowTitle('Saving')
        self.progress_bar.setCancelButton(None)
        self.progress_bar.show()

    def set_progress(self, value):
        if self.progress_bar is None:
            self.show_progress()

        self.progress_bar.setValue(value)
        if value >= 100:
            self.progress_bar.hide()
            self.progress_bar.destroy()
            self.progress_bar = None

    def message(self, message, title=None):
        QMessageBox.information(self, title, message)

    def _on_property_changed_update_plot_(self, name):
        """
        Callback to update the plot that displays channel data
        :param name: name of variable that changed
        """
        if name == 'x' or name == 'y':  # make sure that either x or y where changed
            if len(self.data_context.x) == len(self.data_context.y):     # make sure that x and y have the same length
                if self.backend == BACKEND_MATPLOTLIB:
                    self._plot.set_data(self.data_context.x, self.data_context.y)
                    self._ax.relim()
                    self._ax.autoscale_view(True, True, True)
                    self._set_units_()
                    self._plot_control.update()
                else:
                    self._plot_control.set_data(self.data_context.x, self.data_context.y)

    def _set_units_(self):
        """
        Ask the device for the units of the x and y axis -> update the plot
        """
        x_label = 'Time [{}]'.format(self.data_context.x_unit)
        y_label = 'Amplitude [{}]'.format(self.data_context.y_unit)
        if self.backend == BACKEND_MATPLOTLIB:
            self._ax.set_xlabel(x_label)
            self._ax.set_ylabel(y_label)
        elif self.backend == BACKEND_PYQTGRAPH:
            self._plot_control.x_label = x_label
            self._plot_control.y_label = y_label

    def _save_(self):
        """
        Callback to display the save dialog
        """
        qfd = QFileDialog()
        path, _ = QFileDialog.getSaveFileName(qfd, 'Save data to CSV', filter='h5(*.h5)')
        if path == '':
            return
        self.data_context.save_data(path)

    def _setup_(self):
        """
        Generate UI
        """
        left_layout = QVBoxLayout(self)
        left_layout.setAlignment(QtCore.Qt.AlignTop)
        left_frame = QFrame(self)
        left_frame.setLayout(left_layout)
        self.layout_parent.addWidget(left_frame)

        # set title
        title = QLabel(self.name)
        title.setFont(self.title_font)
        left_layout.addWidget(title)

        # add connection section
        connection_layout = QGridLayout()
        connection_frame = QFrame()
        connection_frame.setLayout(connection_layout)
        left_layout.addWidget(connection_frame)

        connection_layout.addWidget(self.add_widget(QLabel('Host:'), width=40), 0, 0)
        host_input = self.add_widget(QLineEdit(), 'host', 'setText', width=100)
        self.bindings.set_binding('connected', host_input, 'setEnabled', operation=lambda value: not value)
        connection_layout.addWidget(host_input, 0, 1)

        connect_button = QPushButton()
        self.bindings.set_binding('connection_button_text', connect_button, 'setText')
        connect_button.clicked.connect(self.data_context.connect_to_device)
        connection_layout.addWidget(self.add_widget(connect_button, width=80), 0, 2)
        connection_layout.addWidget(QLabel('Status:'), 1, 0)
        connection_layout.addWidget(self.add_widget(QLabel(), 'status', 'setText'), 1, 1)
        connection_layout.addWidget(QLabel('Auto-Connect:'), 2, 0)
        connection_layout.addWidget(self.add_widget(QCheckBox(), 'auto_connect', 'setChecked'), 2, 1)

        # data acquisition
        channel_select = self.add_widget(QComboBox(), 'selected_channel', 'setCurrentIndex')
        channel_select.addItems(CHANNELS)
        self.bindings.set_binding('connected', channel_select, 'setEnabled')
        connection_layout.addWidget(QLabel('Channel:'), 3, 0)
        connection_layout.addWidget(channel_select, 3, 1)

        connection_layout.addWidget(QLabel('Start:'), 4, 0)
        start_edit = self.add_widget(QLineEdit(), 'start', 'setText')
        connection_layout.addWidget(start_edit, 4, 1)
        connection_layout.addWidget(QLabel('End:'), 5, 0)
        stop_edit = self.add_widget(QLineEdit(), 'stop', 'setText')
        connection_layout.addWidget(stop_edit, 5, 1)
        connection_layout.addWidget(QLabel('Available:'), 6, 0)
        connection_layout.addWidget(self.add_widget(QLabel(), 'length', 'setText'), 6, 1)
        self.bindings.set_binding('connected', start_edit, 'setEnabled')
        self.bindings.set_binding('connected', stop_edit, 'setEnabled')
        select_all = self.add_widget(QPushButton('Select All'), 'connected', 'setEnabled')
        select_all.clicked.connect(self.data_context.select_all)
        connection_layout.addWidget(select_all, 6, 2)

        get_data_button = QPushButton('Get Data ->')
        get_data_button.clicked.connect(self.data_context.get_data)
        self.bindings.set_binding('get_data_allowed', get_data_button, 'setEnabled')
        connection_layout.addWidget(get_data_button, 3, 2)

        clear_data_button = QPushButton('Clear Data')
        clear_data_button.clicked.connect(self.data_context.clear_data)
        connection_layout.addWidget(clear_data_button, 4, 2)

        save_data_button = self.add_widget(QPushButton('Save Data'), 'is_saving', 'setEnabled', operation=lambda x: not x)
        save_data_button.clicked.connect(self._save_)
        connection_layout.addWidget(save_data_button, 5, 2)

        # instrument configuration
        instrument_layout = QGridLayout()
        instrument_layout.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        instrument_frame = QFrame()
        instrument_frame.setLayout(instrument_layout)
        left_layout.addWidget(instrument_frame)
        self.bindings.set_binding('connected', instrument_frame, 'setEnabled')

        instrument_layout.addWidget(QLabel('V-Scale:'), 0, 0)
        instrument_layout.addWidget(self.add_widget(QLineEdit(), 'vertical_scale', 'setText', width=80), 0, 1)
        instrument_layout.addWidget(self.add_widget(QLabel(), 'vertical_units', 'setText'), 0, 2)
        instrument_layout.addWidget(QLabel('V-Position:'), 1, 0)
        instrument_layout.addWidget(self.add_widget(QLineEdit(), 'vertical_position', 'setText', width=80), 1, 1)

        instrument_layout.addWidget(QLabel('Coupling:'), 0, 3)
        coupling_box = self.add_widget(QComboBox(), 'selected_coupling', 'setCurrentIndex', width=70)
        coupling_box.addItems(COUPLING_MODES)
        instrument_layout.addWidget(coupling_box, 0, 4)

        instrument_layout.addWidget(QLabel('Trigger Mode:'), 1, 3)
        trigger_box = self.add_widget(QComboBox(), 'selected_trigger_mode', 'setCurrentIndex', width=70)
        trigger_box.addItems(TRIGGER_MODES)
        instrument_layout.addWidget(trigger_box)

        instrument_layout.addWidget(self.add_widget(QCheckBox('display when finished'), 'get_data_after_measurement', 'setChecked'), 3, 3, 1, 2, QtCore.Qt.AlignLeft)
        run_stop = self.add_widget(QPushButton(), 'run_button_text', 'setText')
        run_stop.clicked.connect(self.data_context.run_stop)
        single = self.add_widget(QPushButton('single'), 'measurement_in_progress', 'setEnabled', operation=lambda x: not x)
        single.clicked.connect(self.data_context.single)
        instrument_layout.addWidget(run_stop, 3, 0)
        instrument_layout.addWidget(single, 3, 1)

        # Plot section
        if self.backend == BACKEND_MATPLOTLIB:
            self._plot_control = PlotWidget(self)
            self._ax = self._plot_control.add_subplot(111)
            self._ax.set_title('Measurement')
            self._plot = self._ax.plot([], [])[0]
        elif self.backend == BACKEND_PYQTGRAPH:
            self._plot_control = PyGraphWidget(self)
            self._plot_control.title = 'Measurement'
            self._plot_control.set_plot_color('y')
        self._set_units_()
        self.layout_parent.addWidget(self._plot_control)


if __name__ == '__main__':
    tbs = TBS2104Frame.standalone_application()
