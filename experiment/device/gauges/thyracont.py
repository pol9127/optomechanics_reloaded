# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

import os
import serial
from array import array
from threading import Lock
try:
    from PyQt4 import QtCore, QtGui, uic
except:
    from PyQt5 import QtCore, uic
    import PyQt5.QtWidgets as QtGui

from numpy import ndarray, roll, append, delete
from time import time


class Thyracont(object):

    read_write_lock = Lock()

    def __init__(self, port, baudrate=9600, timeout=0.1, address=1):
        """Initialize serial connection to the gauge controller.

        Parameters
        ----------
        port : str or int
            Identifier for the port the gauge is connected to. A
            list of port names may be retrieved using
            `serial.tools.list_ports`.
        baudrate : int, optional
            Baudrate, defaults to 9600.
        timeout : float, optional
            Timeout for the serial communiaction in seconds,
            defaults to 0.1.
        """
        self.ser = serial.Serial(port=port, baudrate=baudrate,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

        self.address = address

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
        send_string = '{:03d}'.format(self.address)+command
        send_string += self._checksum(send_string)
        send_string += '\r'
        self.ser.write(bytearray(send_string, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'\r', max_length=50):
        """Internal function to handle receiving messages to
        controller."""
        # Todo: Check for address
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

        response = bytes(line).decode()

        assert self._checksum(response[:-2]) == response[-2]

        return bytes(line).decode()[4:-2]

    @staticmethod
    def _checksum(string):
        a = array('B')
        a.fromstring(string)  # depricated, but frombytes not
        # available in Py <3.2
        return chr(sum(a) % 64 + 64)

    def write_read(self, command, encoding='ascii', eol=b'\r',
                   max_length=50):
        with self.read_write_lock:
            self._write(command, encoding)
            return self._read(eol, max_length)

    @property
    def type(self):
        command = 'T'
        return self.write_read(command)

    @property
    def pressure(self):
        """Read pressure value from the sensor (in mBar)."""
        command = 'M'
        response = self.write_read(command)
        pressure = float(response[:4])/1000 * 10**(int(response[4:])-20)

        return pressure

    @property
    def degas_status(self):
        command = 'D'
        return bool(int(self.write_read(command)))

    def start_degas(self):
        command = 'd1'
        self.write_read(command)

    def stop_degas(self):
        command = 'd0'
        self.write_read(command)

    def calibration_pirani_athmosphere(self):
        # unlock adjustment point
        command = 'j1'
        self.write_read(command)

        # perform adjustment
        command = 'j100023'
        self.write_read(command)

    def calibration_pirani_zero(self):
        # unlock adjustment point
        command = 'j0'
        self.write_read(command)

        # perform adjustment
        command = 'j000000'
        self.write_read(command)

    def open_gui(self):
        self.gui_app = QtGui.QApplication.instance()
        if self.gui_app is None:
            from sys import argv
            self.gui_app = QtGui.QApplication(argv)
        self.gui = GUI(self)
        self.gui.show()


class GUI(QtGui.QMainWindow):
    def __init__(self, gauge):
        super(GUI, self).__init__()
        ui_file_name = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'thyracont_gui.ui')
        uic.loadUi(ui_file_name, self)

        self.gauge = gauge
        self.start_time = time()

        self.history = ndarray((0, 1))
        self.history_time = ndarray((0, 1))

        self.curve = self.plot.plot(self.history)
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
        pressure = self.gauge.pressure
        self.pressure_value.setText('{:.2e} mBar'.format(pressure))

        self.append_history(pressure)
        self.update_plot()

    def append_history(self, value):
        if len(self.history) < self.history_length.value():
            self.history = append(self.history, value)
            self.history_time = append(self.history_time,
                                       time() - self.start_time)
        else:
            if len(self.history) > self.history_length.value():
                difference = (len(self.history) -
                              self.history_length.value())

                self.history = delete(self.history, range(difference))
                self.history_time = delete(self.history_time,
                                           range(difference))

            self.history = roll(self.history, -1)
            self.history_time = roll(self.history_time, -1)

            self.history[-1] = value
            self.history_time[-1] = time() - self.start_time

    def update_plot(self):
        self.curve.setData(self.history_time-self.history_time[-1],
                           self.history)

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
