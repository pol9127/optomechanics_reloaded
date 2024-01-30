from SerialDevice import SerialDevice


MAX_CURRENT_LIMIT = 8   # maximum current that the device can handle according to the data sheet


class KoherasBoostikLaser(SerialDevice):
    """
    This class allows the control of the NKT Photonics Koheras Boostik optical amplifier
    """

    @property
    def current_set_point(self):
        """
        Gets the current set point [A]
        """
        return self._query_('ACC\r\n')

    @current_set_point.setter
    def current_set_point(self, value):
        """
        Sets the current set point [A]
        """
        if value > MAX_CURRENT_LIMIT:
            value = MAX_CURRENT_LIMIT
        self._write_('ACC {}\r\n'.format(value))
        self._read_()

    @property
    def current(self):
        """
        Gets the current currently measured in the device [A]
        """
        return self._query_('AMC\r\n')

    @property
    def emission(self):
        """
        Returns true if the emission is turned on
        """
        result = int(self._query_('CDO\r\n')) == 1
        return result

    @emission.setter
    def emission(self, enable):
        """
        Turns the emission on or off
        """
        if enable:
            self._write_('CDO 1\r\n')
            self._read_()
            if not self.emission:
                print('Unable to turn on emission. Make sure that the key on the front panel of the device is turned on.')
        else:
            self.write('CDO 0\r\n')
            self._read_()

    def __init__(self, port=None):
        super().__init__(port)
        self.read_buffer_after_write = True

    # some strange exceptions can happen during multi-threading, use these wrappers to be save
    def _query_(self, command):
        """
        Sends a query to the device and does error handling
        :param command: query command
        :return: float response
        """
        try:
            result = self.query(command)

            if b'\r\n' in result:
                arr = result.split(b'\r\n')
                if len(arr) > 2:
                    result = arr[-2]

            return float(result)
        except Exception as e:
            print('Warning, query "{}" was not successful. {}'.format(command, e))
            return 0

    def _write_(self, command):
        """
        Sends a command to the device and does error handling
        :param command: command
        """
        try:
            self.write(command)
        except Exception as e:
            print(e)

    def _read_(self):
        """
        Tries to read a response from the device
        :return: response
        """
        try:
            return self.read()
        except:
            pass    # does rarely happen since the flushing was added


if __name__ == '__main__':
    with KoherasBoostikLaser('COM3') as laser:
        print(laser.current)
        print(laser.current_set_point)
        laser.emission = True
        print(laser.current)
        laser.emission = False
        print(laser.current)
