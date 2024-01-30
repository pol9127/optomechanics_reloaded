from __future__ import division, unicode_literals

import serial
import time


class ValveControl(object):
    """Class representing the Arduino with the ValveControl software.

    This class allows control of the valves using Arduino.
    """

    def __init__(self, port, timeout=0.5):
        """Initialize serial connection to Arduino.

        Parameters
        ----------
        port : str or int
            Identifier for the port Arduino is connected to. A list of port
            names may be retrieved using `serial.tools.list_ports`.
        timeout : float, optional
            Timeout for the serial communication in seconds, defaults to 0.5.
        """
        baudrate=9600

        self.ser = serial.Serial(port=port, baudrate=baudrate,
                                 bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE,
                                 stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to Arduino."""
        if not self.ser.isOpen():
            self.ser.open()

    def close(self):
        """Close serial connection to Arduino."""
        if hasattr(self, 'ser'):
            if self.ser.isOpen():
                self.ser.close()

    def _write(self, command, encoding='ascii'):
        """Internal function to handle sending messages to Arduino."""
        self.ser.write(bytearray(command, encoding=encoding))
        self.ser.flush()

    def _read(self, eol=b'\r\n', max_length=80):
        """Internal function to handle receiving messages from Arduino."""
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

        return bytes(line).decode()[:-leneol]

    def drive_motor(self, motor_number, motor_speed=50, motor_time=500):
        """Drive specified motor to control valve.

        Parameters
        ----------
        motor_number : int
            Identifier for the motor. Must be in (0,1,2).
        motor_speed : int, optional
            Speed of the motor between -255 and 255, defaults to 50. Positive
            values mean opening the valve, negative values mean closing.
        motor_time : int, optional
            Duration for how long motor should run in milliseconds, defaults to
            500.
        """
        assert motor_number in (0,1,2)
        assert -255 <= motor_speed <= 255
        assert motor_time < 10000

        command = "R{num:1d}{speed:0=+4d}{time:04d}".format(num=motor_number,
                                                            speed=motor_speed,
                                                            time=motor_time)

        self._write(command)
        time.sleep(motor_time/1000.)
        assert self._read() == 'OK'