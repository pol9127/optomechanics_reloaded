# -*- coding: utf-8 -*-
"""
Created on Sun Dec 14 15:15:08 2014

@author: Erik Hebestreit
"""

from __future__ import division, unicode_literals

import serial
import os
from PyQt4 import QtCore, QtGui, uic


class EXT75DX:
    """Class representing the pump Edwards EXT75DX.

    This class allows control of the Edwards EXT75DX, which is part
    of the pump station T-Station 75 from Edwards.

    Many functions return an error code that is described in the
    manual table 17.

    Note
    ----
    The documentation of the pump and all the serial commands is in
    https://shop.edwardsvacuum.com/Viewers/Document.ashx?id=1171&lcid
    =2057

    Properties
    ----------
    speed : int
        Rotation speed of the turbo pump in Hz.
    full_speed : int
        Setting for rotation speed in 'full-speed' setting in
        percent of the maximum speed (1500 Hz).
    standby_speed : int
        Setting for rotation speed in 'standby' setting in
        percent of the maximum speed (1500 Hz).
    vent_valve_is_closed : bool
        Is venting valve closed?
    venting_option : int
        Venting option as described in manual table 18.
    temperatures : list of int
        Motor and pump module temperature in C.
    temperature_motor : int
        Motor temperature of the pump in C.
    temperature_pump_module : int
        Pump module temperature in C.
    link_parameters : list of int
        Internal voltage, current and motor power of the pump
        in V, A, and W.
    power_limit : int
        Power limit of the pump in W.
    timer_state : bool
        State of the timer (activated/deactivated). See manual
        section 1.4.3.
    timer_value : int
        Value of the timer in minutes.
    system_status : dict
        Status values of the pump as documented in manual table 26.
    is_running : bool
        Is the turbo pump running?
    is_stopped : bool
        Is the turbo pump stopped?
    is_standby : bool
        Is the turbo pump in standby mode?
    """

    def __init__(self, port, timeout=0.1):
        """Initialize serial connection to the pump module.

        Parameters
        ----------
        port : str or int
            Identifier for the port the pump is connected to. A list
            of port names may be retrieved using
            `serial.tools.list_ports`.
        timeout : float, optional
            Timeout for the serial communiaction in seconds,
            defaults to 0.1.
        """
        self.ser = serial.Serial(port=port, baudrate=9600,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to the pump module."""
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        """Close serial connection to the pump module."""
        if hasattr(self, 'ser'):
            if self.ser.isOpen():
                self.ser.close()

    def _write(self, command, encoding='ascii'):
        """Internal function to handle sending messages to pump."""
        self.ser.flushOutput()
        self.ser.flushInput()
        self.ser.write(bytearray(command, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'\r', max_length=80):
        """Internal function to handle receiving messages to pump."""
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

        return bytes(line).decode()[:-1]

    @staticmethod
    def _unpack_response(response, start=' ', delimiter=';'):
        """Extract parameters from response string.

        Parameters
        ----------
        response :str
            String to be splitted.
        start : str, optional
            Parameters start after this string.
        delimiter : str,optional
            Parameters are separated by this string.
        """
        response = response.split(start, 1)[1]
        return response.split(delimiter)

    @staticmethod
    def _convert(string_list, type_list=None):
        """Convert the list of strings to the types in type_list."""
        if type_list is None:
            type_list = ['int' for _ in range(len(string_list))]
        else:
            assert len(string_list) == len(type_list), \
                "string_list and type_list are of different length"

        values = list()
        for i in range(len(string_list)):
            if type_list[i] == 'int':
                value = int(string_list[i])
            elif type_list[i] == 'bool':
                value = bool(int(string_list[i]))
            elif type_list[i] == 'systemstatus':
                # convert hex-string to integer, to bit string,
                # remove '0b' at beginning, fill with zeros at left
                # side
                bits = bin(int(string_list[i], 16))[2:].zfill(32)
                # translate to dictionary given in manual table 26
                value = {'fail': bool(int(bits[-1])),
                         'stopped speed': bool(int(bits[-2])),
                         'normal speed': bool(int(bits[-3])),
                         'vent valve closed': bool(int(bits[-4])),
                         'start': bool(int(bits[-5])),
                         'serial enable': bool(int(bits[-6])),
                         'standby': bool(int(bits[-7])),
                         'half full speed': bool(int(bits[-8])),
                         'parallel control mode': bool(int(bits[-9])),
                         'serial control mode': bool(int(bits[-10])),
                         'invalid Podule software':
                             bool(int(bits[-11])),
                         'Podule upload incomplete':
                             bool(int(bits[-12])),
                         'timer expired': bool(int(bits[-13])),
                         'hardware trip': bool(int(bits[-14])),
                         'thermistor error': bool(int(bits[-15])),
                         'serial control mode interlock':
                             bool(int(bits[-16]))}
            else:
                value = None

            values.append(value)

        return values

    def close_vent_valve(self):
        """Close the vent valve at the pump."""
        command = '!C875 1\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    def start_pump(self):
        """Start the turbo pump.

        Ensure that the rough pump is running.
        """
        command = '!C852 1\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    def stop_pump(self):
        """Stop the turbo pump.

        The vent valve will automatically open at a rotation speed
        of 50% (or as defined by `venting_option`).
        """
        command = '!C852 0\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    def standby(self):
        """Set standby mode for the turbo pump.

        Speed at standby mode can be set using parameter
        `standby_speed`.
        """
        command = '!C869 1\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    def full(self):
        """Set full-speed mode for the turbo pump.

        Speed at full-speed mode can be set using parameter
        `full_speed`.
        """
        command = '!C869 0\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    def reset_pump(self):
        """Reset all pump settings."""
        command = '*S867 1\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @property
    def temperatures(self):
        command = '?V859\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)

    @property
    def temperature_motor(self):
        return self.temperatures[0]

    @property
    def temperature_pump_module(self):
        return self.temperatures[1]

    @property
    def speed_system_status(self):
        command = '?V852\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response, ['int', 'systemstatus'])

    @property
    def speed(self):
        return self.speed_system_status[0]

    @property
    def system_status(self):
        return self.speed_system_status[1]

    @property
    def is_running(self):
        return not self.system_status['stopped speed']

    @property
    def is_stopped(self):
        return self.system_status['stopped speed']

    @property
    def vent_valve_is_closed(self):
        return self.system_status['vent valve closed']

    @property
    def is_standby(self):
        return self.system_status['standby']

    @property
    def link_parameters(self):
        command = '?V860\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        response = self._convert(response)
        return [response[0]/10., response[1]/10., response[2]/10.]

    @property
    def power_limit(self):
        command = '?S855\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @power_limit.setter
    def power_limit(self, value):
        command = '!S855 {val}\r'.format(val=round(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    @property
    def venting_option(self):
        command = '?S853\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @venting_option.setter
    def venting_option(self, value):
        assert 0 <= value <= 7, "invalid option"

        command = '!S853 {val}\r'.format(val=int(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    @property
    def standby_speed(self):
        command = '?S857\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @standby_speed.setter
    def standby_speed(self, value):
        assert 55 <= value <= 100, "invalid speed"

        command = '!S857 {val}\r'.format(val=int(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    @property
    def full_speed(self):
        command = '?S856\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @full_speed.setter
    def full_speed(self, value):
        assert 50 <= value <= 100, "invalid speed"

        command = '!S856 {val}\r'.format(val=int(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    @property
    def timer_state(self):
        command = '?S870\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response, ['bool'])[0]

    @timer_state.setter
    def timer_state(self, value):
        command = '!S870 {val}\r'.format(val=int(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    @property
    def timer_value(self):
        command = '?S854\r'
        self._write(command)
        response = self._read()
        response = self._unpack_response(response)
        return self._convert(response)[0]

    @timer_value.setter
    def timer_value(self, value):
        assert 1 <= value <= 30, "invalid time"

        command = '!S854 {val}\r'.format(val=int(value))
        self._write(command)
        response = self._read()
        response = self._convert(self._unpack_response(response))
        assert response[0] == 0, "Error code {}.".format(response[0])

    def open_gui(self, update_interval=None):
        self.gui_app = QtGui.QApplication.instance()
        if self.gui_app is None:
            from sys import argv
            self.gui_app = QtGui.QApplication(argv)
        self.gui = GUI(self, update_interval)
        self.gui.show()


class GUI(QtGui.QMainWindow):
    def __init__(self, pump, update_interval=None):
        super(GUI, self).__init__()
        ui_file_name = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'EXT75DX_gui.ui')
        uic.loadUi(ui_file_name, self)

        self.pump = pump

        self.update_values()

        self.refresh.clicked.connect(self.update_values)

        self.start_pump.clicked.connect(self.pump.start_pump)
        self.stop_pump.clicked.connect(self.pump.stop_pump)
        self.close_vent_valve.clicked.connect(
            self.pump.close_vent_valve)
        self.reset_pump.clicked.connect(self.pump.reset_pump)
        self.normal.toggled.connect(self.pump.full)
        self.standby.toggled.connect(self.pump.standby)

        self.normal_speed.valueChanged.connect(self.set_normal_speed)
        self.standby_speed.valueChanged.connect(self.set_standby_speed)
        self.vent_mode.currentIndexChanged.connect(self.set_vent_mode)
        self.power_limit.valueChanged.connect(self.set_power_limit)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_values)

        if update_interval is not None:
            self.update_interval(1)
            self.update_timer(True)

        self.show()

    def update_interval(self, time):
        self.timer.setInterval(int(time*1000))

    def update_timer(self, state):
        if state:
            self.timer.start()
        else:
            self.timer.stop()

    def update_values(self):
        speed_system_status = self.pump.speed_system_status
        temperatures = self.pump.temperatures

        self.speed.setValue(speed_system_status[0])
        self.temp_motor.setValue(temperatures[0])
        self.temp_module.setValue(temperatures[1])
        self.power_consumption.setValue(self.pump.link_parameters[2])

        if speed_system_status[1]['vent valve closed']:
            self.vent_valve_state.setText('CLOSED')
        else:
            self.vent_valve_state.setText('OPEN')

        if speed_system_status[1]['standby']:
            self.standby.setChecked(True)
        else:
            self.normal.setChecked(True)

        self.normal_speed.setValue(self.pump.full_speed)
        self.standby_speed.setValue(self.pump.standby_speed)
        self.vent_mode.setCurrentIndex(self.pump.venting_option)
        self.power_limit.setValue(self.pump.power_limit)

    def set_normal_speed(self, value):
        self.pump.full_speed = value

    def set_standby_speed(self, value):
        self.pump.standby_speed = value

    def set_vent_mode(self, mode):
        self.pump.venting_option = mode

    def set_power_limit(self, value):
        self.pump.power_limit = value

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
