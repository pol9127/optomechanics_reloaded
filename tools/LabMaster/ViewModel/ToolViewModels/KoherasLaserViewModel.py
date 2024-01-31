from QtModularUiPack.ViewModels import BaseViewModel
from KoherasLaser.KoherasBoostikLaser import KoherasBoostikLaser
from ArduinoPowerMeter.ArduinoPowerMeter import ArduinoPowerMeter
import numpy as np
import json
import os


POWER_METER_ENABLED = False


class KoherasLaserViewModel(BaseViewModel):
    """
    Data context for the Koheras Laser Frame.
    """

    name = 'laser'

    _Koheras_config_file_ = 'Koheras_Laser_configuration.json'

    @property
    def connected(self):
        """
        True if the laser is connected.
        """
        return self._laser.connected

    @property
    def connect_button_text(self):
        """
        The text of the connect / disconnect button
        """
        if self.connected:
            return 'disconnect'
        else:
            return 'connect'

    @property
    def calibration_x(self):
        """
        Gets the x data of the measured calibration curve
        """
        return self._calibration_x

    @calibration_x.setter
    def calibration_x(self, value):
        """
        Sets the x data of the measured calibration curve
        """
        self._calibration_x = value
        self.notify_change('calibration_x')

    @property
    def calibration_y(self):
        """
        Gets the y data of the measured calibration curve
        """
        return self._calibration_y

    @calibration_y.setter
    def calibration_y(self, value):
        """
        Sets the y data of the measured calibration curve
        """
        self._calibration_y = value
        self.notify_change('calibration_y')

    @property
    def show_calibration(self):
        """
        Gets if the calibration curve is shown on the UI
        """
        return self._show_calibration

    @show_calibration.setter
    def show_calibration(self, value):
        """
        Sets if the calibration curve is shown on the UI
        """
        self._show_calibration = value
        self.notify_change('show_calibration')

    @property
    def fit_start(self):
        """
        Gets the first point from where the linear regression of the calibration curve is made
        """
        return self._fit_start

    @fit_start.setter
    def fit_start(self, value):
        """
        Sets the first point from where the linear regression of the calibration curve is made
        """
        self._fit_start = value
        self.notify_change('fit_start')

    @property
    def port(self):
        """
        Gets the serial port of the laser
        """
        return self._laser.port

    @port.setter
    def port(self, value):
        """
        Sets the serial port of the laser
        """
        self._laser.port = value
        self.notify_change('port')

    @property
    def current_set_point(self):
        """
        Gets the current set point of the laser
        """
        return self._current_set_point

    @current_set_point.setter
    def current_set_point(self, value):
        """
        Sets the current set point of the laser
        """
        self._current_set_point = value
        self._power = self.get_fitted_power(value)
        self.notify_change('power')
        self.notify_change('current_set_point')

    @property
    def power(self):
        """
        Gets the power of the laser
        """
        return self._power

    @power.setter
    def power(self, value):
        """
        Sets the power of the laser
        """
        self._power = value
        self._current_set_point = self.from_power_to_current(value)
        self.notify_change('power')
        self.notify_change('current_set_point')

    @property
    def device_set_point(self):
        """
        Gets the current set point from the device.
        """
        if self.connected:
            return self._laser.current_set_point
        else:
            return 0

    @property
    def emission_on(self):
        """
        Gets if the emission is turned on
        :return:
        """
        if self.connected:
            return self._laser.emission
        else:
            return False

    @emission_on.setter
    def emission_on(self, value):
        """
        Turns the emission on or off
        """
        self._laser.emission = value
        if POWER_METER_ENABLED:
            self._set_power_meter_()
        self.notify_change('emission_on')

    @property
    def power_meter_port(self):
        """
        Gets the serial port of the power meter
        """
        if POWER_METER_ENABLED:
            return self._power_meter.port
        else:
            return None

    @power_meter_port.setter
    def power_meter_port(self, value):
        """
        Sets the serial port of the power meter
        :param value: port
        """
        if POWER_METER_ENABLED:
            self._power_meter.port = value
        self.notify_change('power_meter_port')

    @property
    def power_meter_button_text(self):
        """
        Gets the appropriate text for the power meter connect / disconnect button
        :return:
        """
        return 'disconnect' if self._power_meter.connected else 'connect'

    @property
    def power_meter_max_power(self):
        if POWER_METER_ENABLED:
            return self._power_meter.max_power
        else:
            return 0

    @power_meter_max_power.setter
    def power_meter_max_power(self, value):
        if POWER_METER_ENABLED:
            self._power_meter.max_power = value
        self.notify_change('power_meter_max_power')

    def __init__(self):
        super().__init__()
        self.on_update_calibration = list()
        self._power = 0
        self._current_set_point = 0
        self._calibration_x = None
        self._calibration_y = None
        self._show_calibration = False
        self._laser = KoherasBoostikLaser()
        if POWER_METER_ENABLED:
            self._power_meter = ArduinoPowerMeter()
        self.power_meter_max_power = 1000
        self._fit_start = 0
        self.fit_x = None
        self.fit_y = None
        self.fit = None
        self.load_configuration()
        self.update_calibration()

    def __del__(self):
        self._laser.close()
        if POWER_METER_ENABLED:
            self._set_power_meter_()
            self._power_meter.close()

    def _set_power_meter_(self):
        """
        Sets the current power value of the laser on the power meter (zero if emission is turned of or laser not connected)
        """
        if self._power_meter.connected:
            if self._laser.connected and self._laser.emission:
                self._power_meter.power = self.get_fitted_power(self._laser.current)
            else:
                self._power_meter.power = 0

    def save_configuration(self):
        data = {'calibration_x': self.calibration_x, 'calibration_y': self.calibration_y,
                'show_calibration': self.show_calibration, 'fit_start': self.fit_start,
                'port': self.port, 'power_meter': self.power_meter_port, 'power_meter_max': self.power_meter_max_power}
        with open(self._Koheras_config_file_, 'w') as file:
            file.write(json.dumps(data))

    def load_configuration(self):
        if os.path.isfile(self._Koheras_config_file_):
            with open(self._Koheras_config_file_, 'r') as file:
                data = json.loads(file.read())
                self.calibration_x = data['calibration_x']
                self.calibration_y = data['calibration_y']
                self.show_calibration = data['show_calibration']
                self.fit_start = data['fit_start']
                self.port = data['port']
                self.power_meter_port = data['power_meter']
                self.power_meter_max_power = data['power_meter_max']

    def set_on_laser(self):
        """
        Sets the current set point on the device
        """
        if self.connected:
            self._laser.current_set_point = self.current_set_point
            if POWER_METER_ENABLED:
                self._set_power_meter_()
            self.notify_change('device_set_point')

    def get_fitted_point(self, point):
        """
        Evaluate fitted function at a given point
        :param point: point on x axis
        """
        if self.fit is None:
            return point
        else:
            value = 0
            n = len(self.fit) - 1
            for c in self.fit:
                value += point**n * c
                n -= 1
            return value

    def get_fitted_power(self, current):
        """
        Get power for a given current
        :param current: current in A
        :return: power in mW
        """
        if self.fit is None:
            return 0
        else:
            slope = self.fit[0]
            offset = self.fit[1]
            return round(slope * current + offset, 2)

    def from_power_to_current(self, power):
        """
        Get current from power
        :param power: power in mW
        :return: current in A
        """
        if self.fit is None:
            return 0
        else:
            slope = self.fit[0]
            offset = self.fit[1]
            return round((power - offset) / slope, 2)

    def update_calibration(self):
        """
        Update calibration data. Updates plot with new data points and also does a linear regression.
        """
        self.fit_x = list()
        y = list()
        for i in range(len(self.calibration_x)):
            if self.calibration_x[i] > self.fit_start:
                self.fit_x.append(self.calibration_x[i])
                y.append(self.calibration_y[i])

        self.fit = np.polyfit(self.fit_x, y, 1)
        self.fit_y = [self.get_fitted_point(x) for x in self.fit_x]

        for callback in self.on_update_calibration:
            callback()

    def connect_disconnect(self):
        """
        Connect or disconnect
        """
        if self.connected:
            self.disconnect()
        else:
            self.connect()

    def connect_disconnect_power_meter(self):
        """
        Connect or disconnect the power meter
        """
        if self._power_meter.connected:
            self._power_meter.close()
        else:
            self._power_meter.open()
            self._set_power_meter_()
        self.notify_change('power_meter_button_text')

    def flush(self):
        """
        Flush the serial buffer of the device port
        """
        if self.connected:
            self._laser.flush()

    def connect(self):
        """
        Connect to the device
        """
        self._laser.open()

        if self.connected:
            self.current_set_point = self._laser.current_set_point
            self.power = self.get_fitted_power(self.current_set_point)

            if POWER_METER_ENABLED:
                if self.power_meter_port is not None and self.power_meter_port != '':
                    self._power_meter.open()
                    self._set_power_meter_()
                    self.notify_change('power_meter_button_text')

            self.notify_change('connected')
            self.notify_change('connect_button_text')
            self.notify_change('device_set_point')
            self.notify_change('emission_on')
        else:
            print('Unable to connect to laser.')

    def disconnect(self):
        """
        Disconnect the device
        """
        self._laser.close()
        if POWER_METER_ENABLED:
            if self._power_meter.connected:
                self._power_meter.power = 0
                self._power_meter.close()
                self.notify_change('power_meter_button_text')
        self.notify_change('connected')
        self.notify_change('connect_button_text')
