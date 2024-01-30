# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:42:10 2014

@author: Erik Hebestreit
"""

from __future__ import division, unicode_literals

import os
import serial
from threading import Lock
try:
    from PyQt4 import QtCore, QtGui, uic
except:
    from PyQt5 import QtCore, uic
    import PyQt5.QtWidgets as QtGui
from numpy import ndarray, roll, append, delete, nan
from time import time


class MVC3(object):

    read_write_lock = Lock()

    def __init__(self, port, baudrate=38400, timeout=0.1):
        """Initialize serial connection to the gauge controller.

        Parameters
        ----------
        port : str or int
            Identifier for the port the gauge is connected to. A
            list of port names may be retrieved using
            `serial.tools.list_ports`.
        baudrate : int, optional
            Baudrate can be either 9600, 19200, or 38400 (default).
            The same value has to be set on the controller.
        timeout : float, optional
            Timeout for the serial communiaction in seconds,
            defaults to 0.1.
        """
        self.ser = serial.Serial(port=port, baudrate=baudrate,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

        self.channel_1 = Channel(1, self)
        self.channel_2 = Channel(2, self)
        self.channel_3 = Channel(3, self)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to the gauge controller."""
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        """Close serial connection to the gauge controller."""
        if hasattr(self, 'ser'):
            if self.ser.isOpen():
                self.ser.close()

    def _write(self, command, encoding='ascii'):
        """Internal function to handle sending messages to
        controller."""
        self.ser.write(bytearray(command, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'\r', max_length=50):
        """Internal function to handle receiving messages to
        controller."""
        leneol = len(eol)
        line = bytearray()

        i = 0
        while i < max_length:
            i += 1
            c = self.ser.read(1)
            if c:
                line += c
                if line[-leneol:] == eol:
                    break
            else:
                break

        # Handle Error Messages
        if line.decode()[0] == '?':
            if line.decode()[2] == 'X':
                raise RuntimeError('Command not recognized by devices.')
            elif line.decode()[2] == 'P':
                raise RuntimeError(
                    'Parameter number {par} incorrect.'.format(
                        par=line.decode()[5]))
            elif line.decode()[2] == 'C':
                raise RuntimeError(
                    'Channel {chn} on devices not available.'.format(
                        chn=line.decode()[5]))
            elif line.decode()[2] == 'S':
                raise RuntimeError(
                    'No sensor on Channel {chn} connected.'.format(
                        chn=line.decode()[5]))
            elif line.decode()[2] == 'K':
                raise RuntimeError(
                    'No divider in the command available.')

        return bytes(line).decode()[:-1]

    def write_read(self, command, encoding='ascii', eol=b'\r',
                   max_length=50):
        with self.read_write_lock:
            self._write(command, encoding)
            return self._read(eol, max_length)

    def key_lock(self, state):
        command = 'SKL{value}\r'.format(value=int(state))
        return self.write_read(command) == 'OK'

    def safe_actual_configuration(self):
        command = 'SAC\r'
        return self.write_read(command) == 'OK'

    def reset_error(self):
        command = 'SRE\r'
        return self.write_read(command) == 'OK'

    @property
    def version(self):
        command = 'RVN\r'
        return self.write_read(command)

    @property
    def unit(self):
        command = 'RGP\r'
        response = self.write_read(command).split(',\t')
        if response[0] == '0':
            return 'mBar'
        elif response[0] == '1':
            return 'Pa'
        elif response[0] == '2':
            return 'Torr'
        else:
            return None

    @unit.setter
    def unit(self, value):
        if value.lower() == 'mBar'.lower():
            digit = 0
        elif value.lower() == 'Pa'.lower():
            digit = 1
        elif value.lower() == 'Torr'.lower():
            digit = 2
        else:
            digit = None

        if digit is not None:
            command = 'SGP{0},X,X,X,X\r'.format(digit)
            if self.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise ValueError

    @property
    def digits(self):
        command = 'RGP\r'
        response = self.write_read(command).split(',\t')
        if response[1] == '0':
            return 2
        elif response[1] == '1':
            return 3
        else:
            return None

    @digits.setter
    def digits(self, value):
        if value == 2:
            digit = 0
        elif value == 3:
            digit = 1
        else:
            digit = None

        if digit is not None:
            command = 'SGPX,{0},X,X,X\r'.format(digit)
            if self.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise ValueError

    @property
    def brightness(self):
        command = 'RGP\r'
        response = self.write_read(command).split(',\t')
        if response[2] == '0':
            return 'high'
        elif response[2] == '1':
            return 'low'
        else:
            return None

    @brightness.setter
    def brightness(self, value):
        if value.lower() == 'high'.lower():
            digit = 0
        elif value.lower() == 'low'.lower():
            digit = 1
        else:
            digit = None

        if digit is not None:
            command = 'SGPX,X,{0},X,X\r'.format(digit)
            if self.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise ValueError

    @property
    def baudrate(self):
        command = 'RGP\r'
        response = self.write_read(command).split(',\t')
        if response[3] == '0':
            return 9600
        elif response[3] == '1':
            return 19200
        elif response[3] == '2':
            return 38400
        else:
            return None

    @baudrate.setter
    def baudrate(self, value):
        """Setting the baudrate needs a restart of the gauge
        controller."""
        if value == 9600:
            digit = 0
        elif value == 19200:
            digit = 1
        elif value == 38400:
            digit = 2
            # fixme: There seems to be an issue when setting baudrate
            # 38400 --> complains about erroneous parameter 4
        else:
            digit = None

        if digit is not None:
            command = 'SGPX,X,X,{0},X\r'.format(digit)
            if self.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise ValueError

    @property
    def set_point_status(self):
        command = 'RSS\r'
        response = self.write_read(command).split(',')
        return (bool(int(response[0])), bool(int(response[1])),
                bool(int(response[2])), bool(int(response[3])),
                bool(int(response[4])), bool(int(response[5])))

    def read_set_point(self, number):
        if number in range(1, 7):
            command = 'RSP{:d}\r'.format(number)
            response = self.write_read(command).split(',\t')
            return float(response[0]), float(response[1])  # (low, high)
        else:
            raise ValueError

    def set_set_point(self, number, low, high):
        if number in range(1, 7) and low > 0 and high > 0:
            command = 'SSP{num:d},{lo:5.4E},{hi:5.4E}\r'.format(
                num=number, lo=low, hi=high)
            return self.write_read(command) == 'OK'
        else:
            raise ValueError

    def open_gui(self):
        self.gui_app = QtGui.QApplication.instance()
        if self.gui_app is None:
            from sys import argv
            self.gui_app = QtGui.QApplication(argv)
        self.gui = GUI(self)
        self.gui.show()


class Channel(object):
    number = None
    controller = None

    def __init__(self, number, mvc3):
        self.number = number
        self.controller = mvc3

    def hv(self, state):
        command = 'SHV{channel},{value}\r'.format(channel=self.number,
                                                  value=int(state))
        return self.controller.write_read(command) == 'OK'

    def degas(self, state):
        command = 'SDG{channel},{value}\r'.format(channel=self.number,
                                                  value=int(state))
        return self.controller.write_read(command) == 'OK'

    @property
    def pressure(self):
        """Read pressure value from the sensor.

        Returns
        -------
        float or int
            Returns a float value of the read pressure, if the
            measuring value is OK. Returns a negative integer value
            in one of the following error cases:

            -1
                Measuring value < Measuring range
            -2
                Measuring value > Measuring range
            -3
                Measuring range undershooting (Err Lo)
            -4
                Measuring range overstepping (Err Hi)
            -5
                Sensor off (oFF)
            -6
                HV on (HU on)
            -7
                Sensor error (Err S)
            -8
                BA error (Err bA)
            -9
                No Sensor (no Sen)
            -10
                No switch on or switch off point (notriG)
            -11
                Pressure value overstepping (Err P)
            -12
                Pirani error ATMION (Err Pi)
            -13
                Breakdown of operational voltage (Err24)
            -14
                Filament defective (FiLbr)
        """
        command = 'RPV{channel}\r'.format(channel=self.number)
        response = self.controller.write_read(command).split('\t')
        pressure = float(response[1])
        msg = int(response[0][:-1])

        if msg == 0:
            return pressure
        else:
            return -msg

    @property
    def sensor_id(self):
        command = 'RID{channel}\r'.format(channel=self.number)
        response = int(self.controller.write_read(command))

        if response == 0:
            return None
        elif response == 1:
            return 'Ptr'
        elif response == 2:
            return 'ttr1'
        elif response == 3:
            return 'ttr'
        elif response == 4:
            return 'Ctr'
        elif response == 5:
            return 'bA'
        elif response == 6:
            return 'bEE'
        elif response == 7:
            return 'At'
        elif response == 8:
            return 'Ptr90'

    @property
    def filter_factor(self):
        command = 'RFF{channel}\r'.format(channel=self.number)
        response = int(self.controller.write_read(command))

        if response == 0:
            return 1
        elif response == 1:
            return 3
        elif response == 2:
            return 7
        elif response == 3:
            return 15
        else:
            return None

    @filter_factor.setter
    def filter_factor(self, value):
        if value in (1, 3, 7, 15):
            command = 'SFF{channel},{value}\r'.format(
                channel=self.number, value=value)
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError

    @property
    def gas_correction(self):
        command = 'RGC{channel}\r'.format(channel=self.number)
        return float(self.controller.write_read(command))

    @gas_correction.setter
    def gas_correction(self, value):
        if 0.2 <= value <= 8.0:
            command = 'SGC{channel},{value:.2f}\r'.format(
                channel=self.number, value=value)
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError

    @property
    def filament_sensitivity(self):
        command = 'RSF{channel}\r'.format(channel=self.number)
        response = self.controller.write_read(command).split(',\t')
        return float(response[0]), float(response[1])

    @filament_sensitivity.setter
    def filament_sensitivity(self, values):
        if 1 <= values[0] <= 80 and 1 <= values[1] <= 80:
            command = 'SSF{channel},{value1:3f},{value2:3f}\r'.format(
                channel=self.number, value1=values[0], value2=values[1])
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError

    @property
    def filament_mode(self):
        command = 'RFM{channel}\r'.format(channel=self.number)
        return int(self.controller.write_read(command))

    @filament_mode.setter
    def filament_mode(self, value):
        if value in (0, 1, 2):
            command = 'SFM{channel},{value:d}\r'.format(
                channel=self.number, value=value)
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError

    @property
    def sensor_control(self):
        command = 'RSC{channel}\r'.format(channel=self.number)
        response = self.controller.write_read(command).split(',\t')
        return {'switch on': (int(response[0]), float(response[2])),
                'switch off': (int(response[1]), float(response[3]))}

    @sensor_control.setter
    def sensor_control(self, values):
        if (values['switch on'][0] in (0, 1, 2, 3, 4, 5) and
                values['switch off'][0] in (0, 1, 2, 3, 4, 5) and
                values['switch on'][1] > 0 and
                values['switch off'][1] > 0):

            # fixme: format of exponential numbers is wrong. Should
            # have two digits in the exponent. See here:
            # http://stackoverflow.com/a/9911741/1970609
            command = 'SSC{ch},{on},{onval:5.4E},{off},' \
                      '{offval:5.4E}\r'.format(
                           ch=self.number,
                           on=values['switch on'][0],
                           onval=values['switch on'][1],
                           off=values['switch off'][0],
                           offval=values['switch off'][1])
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError

    @property
    def ctr_full_scale(self):
        command = 'RFS{channel}\r'.format(channel=self.number)
        return int(self.controller.write_read(command))

    @ctr_full_scale.setter
    def ctr_full_scale(self, value):
        if value in (0, 1, 2, 3, 4):
            command = 'SFS{channel},{value:d}\r'.format(
                channel=self.number, value=value)
            if self.controller.write_read(command) != 'OK':
                raise AttributeError
        else:
            raise AttributeError


class GUI(QtGui.QMainWindow):
    number_channels = 3

    def __init__(self, gauge):
        super(GUI, self).__init__()
        ui_file_name = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'MVC3_gui.ui')
        uic.loadUi(ui_file_name, self)

        self.gauge = gauge
        self.start_time = time()

        self.history = ndarray((0, self.number_channels))
        self.history_time = ndarray((0, 1))

        self.plot.addLegend()
        self.curve_1 = self.plot.plot(
            self.history[:, 0], pen='r',
            name=self.gauge.channel_1.sensor_id)
        self.curve_2 = self.plot.plot(
            self.history[:, 1], pen='g',
            name=self.gauge.channel_2.sensor_id)
        self.curve_3 = self.plot.plot(
            self.history[:, 2], pen='y',
            name=self.gauge.channel_3.sensor_id)
        self.plot.setLogMode(y=True)
        self.plot.setLabel('left', 'Pressure', 'mBar')
        self.plot.setLabel('bottom', 'Time', 's')

        self.update_pressure()

        self.refresh_values.clicked.connect(self.update_pressure)
        self.refresh_time.valueChanged.connect(self.update_interval)
        self.monitor_button.toggled.connect(self.update_timer)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_pressure)

        self.update_interval()

        self.monitor_button.toggle()

        self.show()

    def update_interval(self):
        self.timer.setInterval(int(self.refresh_time.value()*1000))

    def update_timer(self, state):
        if state:
            self.timer.start()
        else:
            self.timer.stop()

    def update_pressure(self):
        try:
            pressure_1 = self.gauge.channel_1.pressure
            if pressure_1 < 0:
                pressure_1 = nan

            self.pressure_value_1.setText(
                    '{:.2e} mBar'.format(pressure_1))
        except RuntimeError:
            self.pressure_value_1.setText('No Sensor')
            pressure_1 = nan

        try:
            pressure_2 = self.gauge.channel_2.pressure
            if pressure_2 < 0:
                pressure_2 = nan

            self.pressure_value_2.setText(
                    '{:.2e} mBar'.format(pressure_2))
        except RuntimeError:
            self.pressure_value_2.setText('No Sensor')
            pressure_2 = nan

        try:
            pressure_3 = self.gauge.channel_3.pressure
            if pressure_3 < 0:
                pressure_3 = nan

            self.pressure_value_3.setText(
                    '{:.2e} mBar'.format(pressure_3))
        except RuntimeError:
            self.pressure_value_3.setText('No Sensor')
            pressure_3 = nan

        self.append_history([pressure_1, pressure_2, pressure_3])
        self.update_plot()

    def append_history(self, value):
        if len(self.history) < self.history_length.value():
            self.history = append(self.history, [value], axis=0)
            self.history_time = append(self.history_time,
                                       time() - self.start_time)
        else:
            if len(self.history) > self.history_length.value():
                difference = (len(self.history) -
                              self.history_length.value())

                self.history = delete(self.history, range(difference),
                                      axis=0)
                self.history_time = delete(self.history_time,
                                           range(difference))

            self.history = roll(self.history, -1, axis=0)
            self.history_time = roll(self.history_time, -1)

            self.history[-1] = value
            self.history_time[-1] = time() - self.start_time

    def update_plot(self):
        self.curve_1.setData(self.history_time-self.history_time[-1],
                             self.history[:, 0])
        self.curve_2.setData(self.history_time-self.history_time[-1],
                             self.history[:, 1])
        self.curve_3.setData(self.history_time-self.history_time[-1],
                             self.history[:, 2])

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
