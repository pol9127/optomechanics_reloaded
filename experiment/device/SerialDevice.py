from serial import Serial
from time import sleep


class SerialDevice(object):
    """
    This is a base class for devices that utilize the serial port
    """

    @property
    def port(self):
        """
        Gets the serial port of the computer the device is connected to
        """
        return self._port

    @port.setter
    def port(self, value):
        """
        Sets the serial port of the device. (Closes open connections)
        :param value: port
        """
        self.close()
        self._port = value

    @property
    def connected(self):
        """
        True if a connection is currently open.
        """
        if self._connection is not None:
            return self._connection.is_open
        else:
            return False

    def __init__(self, port=None):
        self._port = port
        self._connection = None
        self.encoding = 'ASCII'
        self.line_end = '\n'
        self.command_pause = 0.05

    def flush(self):
        """
        Flushes the input and output buffers
        """
        if self.connected:
            self._connection.reset_input_buffer()
            sleep(self.command_pause)
            self._connection.reset_output_buffer()
            sleep(self.command_pause)

    def write(self, command):
        """
        Writes to the devices
        :param command: command to write
        """
        if self.connected:
            self._connection.write(bytes(command, self.encoding))
            sleep(self.command_pause)

    def read(self):
        """
        Reads the output buffer of the device
        :return: bytes from the output buffer
        """
        if self.connected:
            return self._connection.read_all()
        else:
            return None

    def query(self, command):
        """
        Sends a command to the device and returns the response
        :param command: query
        :return: reply from the device
        """
        if self.connected:
            self.flush()
            self.write(command)
            return self.read()
        else:
            return None

    def open(self):
        """
        Opens the connection to the device
        :return:
        """
        self._connection = Serial(self.port)
        answer = self._connection.read_all()
        self._connection.timeout = 1
        self._connection.write_timeout = 1
        return answer

    def close(self):
        """
        Closes the connection to the device
        """
        if self.connected:
            self._connection.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
