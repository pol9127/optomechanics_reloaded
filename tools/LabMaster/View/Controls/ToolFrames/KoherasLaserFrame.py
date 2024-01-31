from PyQt5.QtWidgets import QGridLayout, QCheckBox, QLabel, QLineEdit, QPushButton, QFrame, QSpinBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from QtModularUiPack.Widgets import PlotWidget, EmptyFrame
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel, POWER_METER_ENABLED


class KoherasLaserFrame(EmptyFrame):
    """
    This frame allows the configuration and control of the NKT Photonics Koheras Boostik optical amplifier
    """

    name = 'Koheras Laser'

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.data_context = KoherasLaserViewModel()
        self.data_context.on_update_calibration.append(self._update_calibration_)
        self._layout = QGridLayout(self)
        self._layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setLayout(self._layout)
        self._setup_()

    def _update_calibration_(self):
        """
        Callback if the calibration curve was updated -> updates the plot
        """
        try:  # if there are invalid data points don't crash but reports
            self._plot_meas.set_data(self.data_context.calibration_x, self.data_context.calibration_y)
            self._plot_fit.set_data(self.data_context.fit_x, self.data_context.fit_y)
            self._ax.relim()
            self._ax.autoscale_view(True, True, True)
            self._calibration_plot.update()
        except Exception as e:
            print(e)

    def _values_to_text(self, values):
        """
        Convert values array of floats to text for the line edit widgets
        :param values: array
        :return: text
        """
        if values is None:
            return None
        else:
            return ','.join([str(i) for i in values])

    def _text_to_values(self, text):
        """
        Convert delimiter separated text to a list of float values
        :param text: text
        :return: list
        """
        if text is None:
            return None
        else:
            delimiter = None
            if ',' in text:
                delimiter = ','
            elif ';' in text:
                delimiter = ';'
            elif ' ' in text:
                delimiter = ' '
            elif '\t' in text:
                delimiter = '\t'
            else:
                return [float(text)]

            result = list()
            for i in text.split(delimiter):
                try:
                    result.append(float(i))
                except:
                    pass
            return result

    def _setup_(self):
        """
        Generate UI
        """
        # set title
        title = QLabel(self.name)
        title.setFont(QFont('Arial', 10))
        self._layout.addWidget(title, 0, 0, 1, 2, Qt.AlignLeft)

        # emission control
        emission_check = self.add_widget(QCheckBox('emission'), 'emission_on', 'setChecked')
        self.bindings.set_binding('connected', emission_check, 'setEnabled')
        self._layout.addWidget(emission_check, 3, 2)

        # Serial Port control
        self._layout.addWidget(QLabel('Port:'), 1, 0)
        port_edit = self.add_widget(QLineEdit(), 'port', 'setText', width=80)
        self.bindings.set_binding('connected', port_edit, 'setEnabled', operation=lambda x: not x)
        self._layout.addWidget(port_edit, 1, 1)

        connection_button = self.add_widget(QPushButton(), 'connect_button_text', 'setText', width=80)
        connection_button.clicked.connect(self.data_context.connect_disconnect)
        self._layout.addWidget(connection_button, 1, 2, 1, 1, Qt.AlignLeft)
        self._layout.addWidget(self.add_widget(QCheckBox('show curve'), 'show_calibration', 'setChecked'), 5, 0, 1, 2, Qt.AlignLeft)

        # current control
        self._layout.addWidget(self.add_widget(QLabel('Power [mW]:'), width=65), 2, 0)
        power_edit = self.add_widget(QLineEdit(), 'power', 'setText', width=80)
        self.bindings.set_binding('connected', power_edit, 'setEnabled')
        self._layout.addWidget(power_edit, 2, 1)
        self._layout.addWidget(self.add_widget(QLabel('Current [A]:'), width=65), 3, 0)
        current_edit = self.add_widget(QLineEdit(), 'current_set_point', 'setText', width=80)
        self.bindings.set_binding('connected', current_edit, 'setEnabled')
        self._layout.addWidget(current_edit, 3, 1)
        set_button = self.add_widget(QPushButton('set'), 'connected', 'setEnabled', width=80)
        set_button.clicked.connect(self.data_context.set_on_laser)
        self._layout.addWidget(set_button, 2, 2)

        self._layout.addWidget(QLabel('device current set point:'), 1, 3, 1, 2, Qt.AlignLeft)
        self._layout.addWidget(self.add_widget(QLabel(), 'device_set_point', 'setText'), 2, 3)
        self._layout.addWidget(QLabel('A'), 2, 4)

        # power meter
        if POWER_METER_ENABLED:
            self._layout.addWidget(QLabel('Power Meter:'), 4, 0)
            self._layout.addWidget(self.add_widget(QLineEdit(), 'power_meter_port', 'setText', width=80), 4, 1)
            power_meter_connect_button = self.add_widget(QPushButton(), 'power_meter_button_text', 'setText', width=80)
            power_meter_connect_button.clicked.connect(self.data_context.connect_disconnect_power_meter)
            self._layout.addWidget(power_meter_connect_button, 4, 2)
            self._layout.addWidget(QLabel('Max Power [mW]:'), 4, 3)
            power_meter_max_box = self.add_widget(QSpinBox(), width=80)
            power_meter_max_box.setMinimum(0)
            power_meter_max_box.setMaximum(10000)
            self.bindings.set_binding('power_meter_max_power', power_meter_max_box, 'setValue')
            self._layout.addWidget(power_meter_max_box, 4, 4)

        # calibration control
        calibration_frame = self.add_widget(QFrame(), 'show_calibration', 'setHidden', operation=lambda x: not x)
        calibration_layout = QGridLayout()
        calibration_frame.setLayout(calibration_layout)
        self._layout.addWidget(calibration_frame, 6, 0, 1, 4, Qt.AlignLeft)

        self._calibration_plot = PlotWidget(self)
        self._ax = self._calibration_plot.add_subplot(111)
        self._ax.set_title('Power Calibration Curve')
        self._ax.set_xlabel('Current [A]')
        self._ax.set_ylabel('Power [mW]')
        self._plot_meas = self._ax.plot([], [], '.')[0]
        self._plot_fit = self._ax.plot([], [], '--')[0]
        self._update_calibration_()
        calibration_layout.addWidget(self._calibration_plot, 0, 0, 1, 4, Qt.AlignCenter)

        calibration_layout.addWidget(QLabel('Current [A]:'), 1, 1)
        calibration_layout.addWidget(self.add_widget(QLineEdit(), 'calibration_x', 'setText', operation=self._values_to_text, inv_op=self._text_to_values), 1, 2)
        calibration_layout.addWidget(QLabel('Power [mW]'), 2, 1)
        calibration_layout.addWidget(self.add_widget(QLineEdit(), 'calibration_y', 'setText', operation=self._values_to_text, inv_op=self._text_to_values), 2, 2)
        calibration_layout.addWidget(QLabel('Fit Start [A]:'), 1, 3)
        calibration_layout.addWidget(self.add_widget(QLineEdit(), 'fit_start', 'setText', width=70), 2, 3)
        update_calibration = QPushButton('update')
        update_calibration.clicked.connect(self.data_context.update_calibration)
        calibration_layout.addWidget(update_calibration, 1, 0)
