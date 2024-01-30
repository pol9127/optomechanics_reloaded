# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 08:36:02 2014

@author: ehebestreit
"""

from __future__ import division, unicode_literals

import serial
import threading


class NIOPS3:
    """Class representing the ion-getter pump controller NIOPS 3.

    This class allows control of the ion-getter pump controller NIOPS 3, for
    the ion-getter pump NEXTorr D200-5 from SAES.

    Attributes
    ----------
    version : str
        Version and date of the controller software.
    status : dict
        Status of different parts of the pump system.
    temperatures : tuple of int
        Temperatures of IP and NP generator in Celsius.
    working_time : tuple of tuple of int
        working times of IP and NP in the format ``(hrs, min)``.
    IP_voltage : int
        IP voltage in V.
    IP_current : float
        IP current in A.
    IP_pressure : float
        Pressure meausred by IP in mBar.
    """
    ACTIVATION = 1
    TIMED_ACTIVATION = 2
    CONDITIONING = 3
    TIMED_CONDITIONING = 4

    lok = threading.Lock()

    def __init__(self, port, timeout=0.2):
        """Initialize serial connection to the pump controller.

        Parameters
        ----------
        port : str or int
            Identifier for the port the controller is connected to. A list of
            port names may be retrieved using `serial.tools.list_ports`.
        timeout : float, optional
            Timeout for the serial communiaction in seconds, defaults to 0.2.
        """
        self.ser = serial.Serial(port=port, baudrate=115200,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to the pump controller."""
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        """Close serial connection to the pump controller."""
        if hasattr(self, 'ser'):
            if self.ser.isOpen():
                self.ser.close()

    def _write(self, command, encoding='ascii'):
        """Internal function to handle sending messages to controller."""
        self.ser.write(bytearray(command, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'\r', max_length=128):
        """Internal function to handle receiving messages to controller."""
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

    def _write_read(self, command, encoding='ascii', eol=b'\r',
                    max_length=128, lines=1):
        with self.lok:
            self._write(command, encoding)

            if lines is 1:
                response = self._read(eol, max_length)
            else:
                response = []
                for i in range(lines):
                    response.append(self._read(eol, max_length))

        return response

    def start_ip(self):
        """Start ion pump module."""
        command = 'G\r'
        return self._write_read(command) == '$'

    def stop_ip(self):
        """Stop ion pump module."""
        command = 'B\r'
        return self._write_read(command) == '$'

    def start_np(self, mode):
        """Start non evaporable getter pump module.

        Parameters
        ----------
        mode : int
            Pump mode as described in manual.
        """
        command_mode = 'M{x}\r'.format(x=mode)
        command_np = 'GN\r'
        return (self._write_read(command_mode) == '$' and
                self._write_read(command_np) == '$')

    def stop_np(self):
        """Stop non evaporable getter pump module."""
        command = 'BN\r'
        return self._write_read(command) == '$'

    # TODO: implement control for comporators/switches

    @property
    def version(self):
        command = 'V\r'
        return self._write_read(command)

    @property
    def status(self):
        command = 'TS\r'
        response = self._write_read(command).split(', ')

        status = dict()
        for part in response:
            part = part.rsplit(' ', 1)
            if part[1] == 'ON':
                status[part[0]] = True
            elif part[1] == 'OFF':
                status[part[0]] = False
            else:
                status[part[0]] = None
        return status

    @property
    def temperatures(self):
        command = 'TC\r'
        response = self._write_read(command)
        _, ip_temp, _, np_temp, _ = response.split(' ')
        return int(ip_temp), int(np_temp)

    @property
    def ip_temperature(self):
        return self.temperatures[0]

    @property
    def np_temperature(self):
        return self.temperatures[1]

    @property
    def working_time(self):
        command = 'TM\r'
        response = self._write_read(command, lines=2)
        _, _, _, hours, _, minutes, _ = response[0].split(' ')
        ip_time = (int(hours), int(minutes))
        _, _, _, hours, _, minutes, _ = response[1].split(' ')
        np_time = (int(hours), int(minutes))
        return ip_time, np_time

    @property
    def ip_voltage(self):
        command = 'u\r'
        response = self._write_read(command)
        return int(response, 16)

    @property
    def ip_current(self):
        command = 'i\r'
        response = self._write_read(command)
        binary_string = bin(int(response, 16))[2:].zfill(16)
        if binary_string[0:2] == '00':
            return int(binary_string[2:], 2) * 1e-9
        elif binary_string[0:2] == '01':
            return int(binary_string[2:], 2) * 1e-7
        elif binary_string[0:2] == '10':
            return int(binary_string[2:], 2) * 1e-5
        else:
            return None

    @property
    def ip_pressure(self):
        command = 'Tb\r'
        response = self._write_read(command)
        return float(response)
