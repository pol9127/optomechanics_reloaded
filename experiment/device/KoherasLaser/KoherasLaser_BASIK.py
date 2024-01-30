from NKTP_DLL import openPorts, getAllPorts, getOpenPorts, closePorts, \
    registerWriteU8, registerWriteU16, registerReadU16, deviceGetModuleSerialNumberStr

AUTO_MODE = 1
LIVE_MODE = 1


class KoherasLaser(object):
    """
    This laser can be controlled via UART
    """

    @property
    def port(self):
        """
        Gets the serial port the laser is connected to
        """
        return self._port

    @port.setter
    def port(self, value):
        """
        Sets the serial port the laser should be connected to and tries to connect to it
        """
        self._port = value
        self.open()

    @property
    def connected(self):
        """
        True if the laser is currently connected
        """
        if self._port is not None:
            return self._port in getOpenPorts()
        else:
            return False

    @property
    def serial_number(self):
        """
        Gets the serial number of the laser
        """
        if self.connected:
            return deviceGetModuleSerialNumberStr(self._port, 1)

    @property
    def status(self):
        """
        Gets a 16 bit unsigned integer that represents the status of the device
        """
        if self.connected:
            return registerReadU16(self._port, 1, 0x66, 0)

    @property
    def emission_on(self):
        """
        True if the emission is turned on
        """
        return self.get_status_bit(0)

    @emission_on.setter
    def emission_on(self, value):
        """
        Sets the emission on or off
        """
        if self.connected:
            registerWriteU8(self._port, 1, 0x30, int(value), 0)

    @property
    def power_mW(self):
        """
        Gets the power in mW
        """
        if self.connected:
            return registerReadU16(self._port, 1, 0x17, 0) * 100

    @power_mW.setter
    def power_mW(self, value):
        """
        Sets the power in mW
        """
        if self.connected:
            registerWriteU16(self._port, 1, 0x22, value / 100, 0)

    @property
    def power_dBm(self):
        """
        Gets the power in dBm
        """
        if self.connected:
            return registerReadU16(self._port, 1, 0x17, 0) * 100

    @power_dBm.setter
    def power_dBm(self, value):
        """
        Sets the power in dBm
        """
        if self.connected:
            registerWriteU16(self._port, 1, 0xA0, value / 100, 0)

    def __init__(self, port=None, verbose=True):
        self._port = port
        self.verbose = verbose

    def close(self):
        """
        Closes the connection to the laser
        """
        if self.connected:
            closePorts(self._port)

    def open(self):
        """
        Tries to open the connection to a laser
        """
        self.close()

        if self.verbose and self._port is None:
            print('no port provided, find automatically...')
        openPorts(getAllPorts() if self._port is None else self._port, AUTO_MODE, LIVE_MODE)
        open_ports = getOpenPorts()

        if len(open_ports) > 0:
            if self.verbose:
                print('found open port: ', open_ports[0])
            self._port = open_ports[0]
        elif self.verbose:
            print('unable to open port')

    def get_status_bit(self, bit):
        if self.connected:
            return (self.status & 1 << bit) != 0
        else:
            return False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == '__main__':
    with KoherasLaser('COM3') as laser:
        print(laser.port)

