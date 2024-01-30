from SerialDevice import SerialDevice
from time import time, sleep


class ArduinoPowerMeter(SerialDevice):
    """
    The power meter can be used to indicate how much power a device is using
    """

    @property
    def power(self):
        """
        Gets the power that the power meter is displaying
        """
        return self._power

    @power.setter
    def power(self, value):
        """
        Sets the power that the power meter is displaying
        :param value: power
        """
        self._power = value
        power_range = abs(self.max_power - self.min_power)
        percentage = (value - self.min_power) / power_range * 100
        self.write('{}\r\n'.format(percentage))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line_end = ''
        self.max_power = 1
        self.min_power = 0
        self.timeout = 3
        self._power = 0

    def open(self):
        answer = super().open()

        # make sure the device is ready
        start_time = time()
        while answer != b'OK':
            answer = self.query('R\n')
            sleep(0.1)
            if time() - start_time > self.timeout:
                raise TimeoutError('Timeout while trying to connect to Arduino.')
        print('Connected to arduino power meter.')


if __name__ == '__main__':
    meter = ArduinoPowerMeter('COM8')
    meter.open()
    for i in range(0, 11):
        meter.power = i * 0.1
        sleep(0.01)
    meter.power = 250e-3
