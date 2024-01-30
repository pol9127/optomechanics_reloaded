from ProcessHandler import ProcessHandler
from time import sleep
import os


EXECUTABLE = 'PeakTech4055MV.exe'

WF_SQUARE = 'square'
WF_SINE = 'sin'
WF_RAMP = 'ramp'
WF = [WF_RAMP, WF_SINE, WF_SQUARE]


class PeakTech4055MV(object):
    """
    This class allows the control of the PeakTech 4055MV signal generator.
    This class only implements a small portion of the capabilities of the real device. For more functions refer to the
    Manual of the device and use the write() and query() methods to use custom commands.

        ATTENTION: Since the API for this device is only available for windows systems this file only works on windows

    """

    @property
    def output_enabled(self):
        """
        Gets if output is enabled
        """
        return self._output_enabled

    @output_enabled.setter
    def output_enabled(self, value):
        """
        Sets the output enabled or disabled
        """
        self._output_enabled = value
        if value:
            self._process_handler.write('write;OUTP ON')
        else:
            self._process_handler.write('write;OUTP OFF')
        sleep(0.05)     # make this faster than the usual commands

    @property
    def frequency(self):
        """
        Gets the frequency (in Hz)
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """
        Sets the frequency (in Hz)
        """
        self._frequency = value
        self.write('write;FREQ {}Hz'.format(value))

    @property
    def amplitude(self):
        """
        Gets the amplitude (in Volts)
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        """
        Sets the amplitude (in Volts)
        """
        self._amplitude = value
        self.write('write;VOLT {}Vpp'.format(value))

    @property
    def offset(self):
        """
        Get the offset (in Volts)
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        """
        Sets the offset (in Volts)
        """
        self._offset = value
        self.write('write;VOLT:OFFS {}Vdc'.format(value))

    @property
    def waveform(self):
        """
        Gets the waveform
        """
        return self._wf

    @waveform.setter
    def waveform(self, value):
        """
        Sets the waveform
        """
        if value not in WF:
            print('The given value is not a valid waveform. Valid waveforms are: {}'.format(WF))
            value = WF_SINE
        self._wf = value
        self._process_handler.write('write;FUNC {}'.format(value))

    @property
    def connected(self):
        """
        True if a device is connected and responsive.
        """
        return self._process_handler.running

    def __init__(self, index=0):
        self._output_enabled = False
        self._frequency = 1000
        self._amplitude = 1
        self._offset = 0
        self._wf = WF_SINE
        self._index = index
        path, _ = os.path.split(__file__)
        exe_path = '{}/{}'.format(path, EXECUTABLE)
        self._process_handler = ProcessHandler(exe_path, verbose=False)
        self.connect()
        self.output_enabled = False
        self.write('Source:Apply:{} {}Hz, {}Vpp, {}Vdc'.format(self._wf, self._frequency, self._amplitude, self._offset))

    def __del__(self):
        self.write('exit')
        self._process_handler.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.write('exit')
        self._process_handler.terminate()

    def connect(self):
        """
        Try to connect to device
        """
        sleep(0.1)  # make sure that the buffer is empty
        result = self.query('connect;{}'.format(self._index))[0]
        if 'Connected device' in result:
            print('Connection to PeakTech4055MV established.')
        else:
            print('Connection to PeakTech4055MV failed.')

    def close(self):
        """
        Close the connection to the device
        :return:
        """
        self._process_handler.terminate()

    def write(self, command):
        """
        Send a command to the device
        :param command: command to send
        """
        self._process_handler.write(command)
        sleep(0.2)

    def query(self, command, result_count=1):
        """
        Query the device
        :param command: query command
        :param result_count: length of expected result list
        :return: list of results
        """
        result = self._process_handler.query(command, result_count)
        sleep(0.2)
        return result


if __name__ == '__main__':
    device = PeakTech4055MV(0)

    ver = 1

    if ver == 0:
        device.waveform = WF_SINE
        device.frequency = 112.45*1e3
        device.amplitude = 0.3
        device.offset = -1
        device.output_enabled = True
    else:
        device.waveform = WF_SQUARE
        device.frequency = 12.45 * 1e3
        device.amplitude = 1.3
        device.offset = 1.120
        device.output_enabled = False

    for i in range(8):
        device.output_enabled = not device.output_enabled

    device.close()

