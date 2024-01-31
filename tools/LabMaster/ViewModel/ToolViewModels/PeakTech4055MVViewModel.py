from QtModularUiPack.ViewModels import BaseViewModel
from PeakTech4055MV.PeakTech4055MV import PeakTech4055MV, WF_SINE, WF
import json
import os


CONFIG_FILE = 'PeakTech4055_config.json'


class PeakTech4055MVViewModel(BaseViewModel):
    """
    The view model for the PeakTech signal generator frame
    """

    name = 'sig_gen'

    @property
    def output_enabled(self):
        """
        Gets if the output is enabled
        """
        return self._device.output_enabled

    @output_enabled.setter
    def output_enabled(self, value):
        """
        Sets if the output is enabled
        """
        self._device.output_enabled = value
        self.notify_change('output_enabled')

    @property
    def amplitude(self):
        """
        Gets the amplitude (in Volts)
        """
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        """
        Sets the amplitude (in Volts, does not apply the setting to the device -> use apply() for that)
        """
        self._amplitude = value
        self.notify_change('amplitude')

    @property
    def offset(self):
        """
        Gets the offset (in Volts)
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        """
        Sets the offset (in Volts, does not apply the setting to the device -> use apply() for that)
        """
        self._offset = value
        self.notify_change('offset')

    @property
    def frequency(self):
        """
        Gets the frequency (in Hz)
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """
        Sets the amplitude (in Hz, does not apply the setting to the device -> use apply() for that)
        """
        self._frequency = value
        self.notify_change('frequency')

    @property
    def waveform(self):
        """
        Gets the waveform
        """
        return self._waveform

    @waveform.setter
    def waveform(self, value):
        """
        Sets the waveform (does not apply the setting to the device -> use apply() for that)
        """
        if value not in WF:
            print('The value given is not a valid waveform. Valid waveforms are: {}'.format(WF))
            value = WF_SINE
        self._waveform = value
        self._waveform_index = WF.index(value)
        self.notify_change('waveform_index')
        self.notify_change('waveform')

    @property
    def waveform_index(self):
        """
        Gets the current index of the selected waveform
        """
        return self._waveform_index

    @waveform_index.setter
    def waveform_index(self, value):
        """
        Sets the waveform by via index (does not apply the setting to the device -> use apply() for that)
        """
        self._waveform_index = value
        self._waveform = WF[value]
        self.notify_change('waveform_index')
        self.notify_change('waveform')

    @property
    def connected(self):
        """
        True if a device is connected
        """
        return self._device.connected

    def __init__(self):
        super().__init__()
        self._device = PeakTech4055MV()
        self._amplitude = self._device.amplitude
        self._offset = self._device.offset
        self._frequency = self._device.frequency
        self._waveform = self._device.waveform
        self._waveform_index = WF.index(self._waveform)
        self.load_configuration()
        self.apply()

    def set_frequency(self, frequency):
        """
        Sets frequency and apply it to the signal generator
        :param frequency: frequency to apply [Hz]
        """
        self.frequency = frequency
        self.apply()

    def save_configuration(self):
        """
        Save the configuration of the signal generator
        """
        data = {'amplitude': self.amplitude, 'offset': self.offset, 'frequency': self.frequency, 'waveform': self.waveform}
        with open(CONFIG_FILE, 'w') as file:
            file.write(json.dumps(data))

    def load_configuration(self):
        """
        Load the configuration of the signal generator
        """
        if os.path.isfile(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as file:
                data = json.loads(file.read())
                self.amplitude = data['amplitude']
                self.offset = data['offset']
                self.frequency = data['frequency']
                self.waveform = data['waveform']

    def connect(self):
        """
        Connect the device
        """
        self._device.connect()

    def apply(self):
        """
        Apply the current settings to the device
        """
        self._device.write('write;Source:Apply:{} {}Hz, {}Vpp, {}Vdc'.format(self._waveform, self._frequency, self._amplitude, self._offset))
