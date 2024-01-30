# -*- coding: utf-8 -*-
"""
:Created on: Mon Dec 15 14:39:02 2014

:Authors: Erik Hebestreit
"""

from __future__ import division, unicode_literals

import os
import serial
from struct import unpack
from threading import Lock
from numpy import ndarray, roll, append, delete, nan
from time import time


class K204:
    """Class representing the thermometer Voltcraft K204.

    This class allows control and readout of the Voltcraft K204 4
    channel thermometer.

    Note
    ----
    Documentation of the API can be found in:

    - http://www.produktinfo.conrad.com/datenblaetter/100000-124999
    /100806-da-01-en-SCHNITTSTELLEN_VOLTCRAFT_304_K204_DGT__T.pdf
    - http://azug.minpet.unibas.ch/~lukas/bricol/100357-da-01-en
    -Schnittstelle_Digitalth_302K_J_202K_J.pdf

    Properties
    ----------
    version : int
        Version number of the software.
    scale : str
        Temperature Scale set on the thermometer (Celsius ``C`` or
        Fahrenheit ``F``).
    temperature_1 : float
        Temperature measured on channel 1.
    temperature_2 : float
        Temperature measured on channel 2.
    temperature_3 : float
        Temperature measured on channel 3.
    temperature_4 : float
        Temperature measured on channel 4.
    """

    read_write_lock = Lock()

    def __init__(self, port, timeout=0.5):
        """Initialize serial connection to the thermometer.

        Parameters
        ----------
        port : str or int
            Identifier for the port the thermometer is connected to.
            A list of port names may be retrieved using
            `serial.tools.list_ports`.
        timeout : float, optional
            Timeout for the serial communiaction in seconds, defaults
            to 0.5.
        """
        self.ser = serial.Serial(port=port, baudrate=9600,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to the thermometer."""
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        """Close serial connection to the thermometer."""
        if hasattr(self, 'ser'):
            if self.ser.isOpen():
                self.ser.close()

    def _write(self, command, encoding='ascii'):
        """Internal function to handle sending messages to
        thermometer."""
        self.ser.write(bytearray(command, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'', max_length=128):
        """Internal function to handle receiving messages to
        thermometer."""
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

        return bytes(line)

    def _write_read(self, command, encoding='ascii', eol=b'',
                    max_length=128, lines=1):
        with self.read_write_lock:
            self._write(command, encoding)

            if lines is 1:
                return self._read(eol, max_length)
            else:
                response = []
                for i in range(lines):
                    response.append(self._read(eol, max_length))
                return response

    def clear_buffer(self):
        response = None

        while response is not b'':
            response = self._read(eol=b'')

    @property
    def state(self):
        """Status of the thermometer (settings and temperatures).

        Returns
        -------
        dict
            Status dictionary containing all information from the
            devices::

                {
                    'mode recording': is recording (bool)
                    'mode max-min': `normal`, `max`, `min`,
                    or `background`
                    'mode T1-T2': in T1-T2 mode (bool)
                    'mode rel': in relative measurement mode (bool)
                    'mode hold': in hold mode (bool)
                    'battery low': battery is low (bool)
                    'scale': `C` or `F`
                    'memory full': memory is full (bool)
                    'auto off': devices will automatically turn off
                    (bool)
                }

        tuple of float
            measured temperatures values
        tuple of float
            measured measured relative temperatures values
        tuple of float
            measured measured minimum temperatures values
        tuple of float
            measured measured maximum temperatures values
        """
        response = self._write_read('A', max_length=45)

        # Check start and end byte for frame error
        if response[0] != b'\x02'[0] or response[44] != b'\x03'[0]:
            return None

        status = dict()

        # Byte 2
        try:
            byte = bin(response[1])[2:].zfill(8)
        except TypeError:
            byte = bin(ord(response[1]))[2:].zfill(8)

        status['mode recording'] = bool(int(byte[-1]))
        if byte[-3:-1] == '00':
            status['mode max-min'] = 'normal'
        elif byte[-3:-1] == '01':
            status['mode max-min'] = 'max'
        elif byte[-3:-1] == '10':
            status['mode max-min'] = 'min'
        elif byte[-3:-1] == '11':
            status['mode max-min'] = 'background'
        status['mode T1-T2'] = bool(int(byte[-4]))
        status['mode rel'] = bool(int(byte[-5]))
        status['mode hold'] = bool(int(byte[-6]))
        status['battery low'] = bool(int(byte[-7]))
        if bool(int(byte[-8])):
            status['scale'] = 'C'
        else:
            status['scale'] = 'F'

        # Byte 3
        try:
            byte = bin(response[2])[2:].zfill(8)
        except TypeError:
            byte = bin(ord(response[2]))[2:].zfill(8)

        status['memory full'] = bool(int(byte[-1]))
        status['auto off'] = bool(int(byte[-8]))

        # Thermometer returns 32767 if no sensor is connected.
        # Translate this to None, others have to be devided by 10.

        # Byte 8-15 - Temperatures
        temp = unpack('HHHH', response[14:6:-1])
        temperature = (temp[3]/10 if temp[3] < 32767 else None,
                       temp[2]/10 if temp[2] < 32767 else None,
                       temp[1]/10 if temp[1] < 32767 else None,
                       temp[0]/10 if temp[0] < 32767 else None)

        # Byte 16-23 - Relative Temperatures
        temp = unpack('HHHH', response[22:14:-1])
        rel_temperature = (temp[3]/10 if temp[3] < 32767 else None,
                           temp[2]/10 if temp[2] < 32767 else None,
                           temp[1]/10 if temp[1] < 32767 else None,
                           temp[0]/10 if temp[0] < 32767 else None)

        # Byte 24-31 - Minimum Temperatures
        temp = unpack('HHHH', response[30:22:-1])
        min_temperature = (temp[3]/10 if temp[3] < 32767 else None,
                           temp[2]/10 if temp[2] < 32767 else None,
                           temp[1]/10 if temp[1] < 32767 else None,
                           temp[0]/10 if temp[0] < 32767 else None)

        # Byte 32-39 - Maximum Temperatures
        temp = unpack('HHHH', response[38:30:-1])
        max_temperature = (temp[3]/10 if temp[3] < 32767 else None,
                           temp[2]/10 if temp[2] < 32767 else None,
                           temp[1]/10 if temp[1] < 32767 else None,
                           temp[0]/10 if temp[0] < 32767 else None)

        return (status, temperature, rel_temperature, min_temperature,
                max_temperature)

    def _btn_hold(self):
        self._write('H')

    def _btn_t1_t2(self):
        self._write('T')

    def _btn_avg_max_min(self):
        self._write('M')

    def _btn_avg_max_min_hold(self):
        self._write('N')

    def _btn_rel(self):
        self._write('R')

    def _btn_c_f(self):
        self._write('C')

    @property
    def version(self):
        """Get `Center Model Number` of thermometer."""
        response = int(self._write_read('K')[0:3])
        return response

    @property
    def scale(self):
        return self.state[0]['scale']

    @scale.setter
    def scale(self, value):
        sc = self.state[0]['scale']
        if sc == 'C' and value == 'F':
            self._btn_c_f()
        elif sc == 'F' and value == 'C':
            self._btn_c_f()
        else:
            pass

    @property
    def temperature_1(self):
        return self.state[1][0]

    @property
    def temperature_2(self):
        return self.state[1][1]

    @property
    def temperature_3(self):
        return self.state[1][2]

    @property
    def temperature_4(self):
        return self.state[1][3]

