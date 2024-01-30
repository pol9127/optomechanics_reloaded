import numpy as np
from visa import VisaTypeError
from VisaDevice import VisaDevice


CH1 = 'CH1'
CH2 = 'CH2'
CH3 = 'CH3'
CH4 = 'CH4'
MATH = 'MATH'
REF1 = 'REF1'
REF2 = 'REF2'

POINT_INT8 = 1
POINT_INT16 = 2

COUPLING_AC = 'AC'
COUPLING_DC = 'DC'
COUPLING_GND = 'GND'

SUPPORTED_UNITS = ['%', '/Hz', 'A', 'A/A', 'A/V', 'A/W', 'A/dB', 'A/s', 'AA', 'AW', 'AdB', 'As', 'B', 'Hz', 'IRE', 'S/s', 'V', 'V/A', 'V/V', 'V/W', 'V/dB', 'V/s', 'VV', 'VW', 'VdB', 'Volts', 'Vs', 'W', 'W/A', 'W/V', 'W/W', 'W/dB', 'W/s,WA', 'WV', 'WW', 'WdB', 'Ws', 'dB', 'dB/A', 'dB/V', 'dB/W', 'dB/dB', 'dBA', 'dBV,dBW', 'dBdB', 'day', 'degrees', 'div', 'hr', 'min', 'ohms', 'percent', 's']
SUPPORTED_ACQUIRE_MODES = ['SAMPLE', 'PEAKDETECT' 'AVERAGE']
SUPPORTED_ACQUISITION_STATES = ['ON', 'OFF', 'STOP', 'RUN', '0']

TRIGGER_MODES = ['AUTO', 'NORMAL']


class TBS2104(VisaDevice):
    """
    TBS 2104 Oscilloscope
    """

    _verification_string = 'TEKTRONIX,TBS2104,'
    _channels_ = [CH1, CH2, CH3, CH4, MATH, REF1, REF2]
    _measurement_channels = [CH1, CH2, CH3, CH4]
    _data_point_formats_ = [POINT_INT8, POINT_INT16]

    @property
    def trigger_mode(self):
        """
        Gets the current trigger mode of the scope
        """
        return self._query_('TRIGGER:A:MODE?')

    @trigger_mode.setter
    def trigger_mode(self, value):
        """
        Sets the current trigger mode of the scope
        """
        if value not in TRIGGER_MODES:
            print('The value "{}" is not a valid trigger mode. Valid modes are: {}'.format(value, TRIGGER_MODES))
        else:
            self.write('TRIGGER:A:MODE {}'.format(value))

    @property
    def channels(self):
        return self._channels_

    @property
    def source(self):
        return self._query_('DATA:SOURCE?')

    @source.setter
    def source(self, value):
        if value not in self.channels:
            print('Invalid channel "{}", setting source to CH1.'.format(value))
            self.source = CH1
        else:
            self.write('DATA:SOURCE {}'.format(value))

    @property
    def coupling(self):
        src = self.source
        if src in self._measurement_channels:
            return self._query_('{}:COUP?'.format(src))
        else:
            return COUPLING_GND

    @coupling.setter
    def coupling(self, value):
        src = self.source
        if src in self._measurement_channels and value in [COUPLING_GND, COUPLING_AC, COUPLING_DC]:
            self.write('{}:COUP {}'.format(src, value))
        else:
            print('Cannot set coupling on "{}".'.format(src))

    @property
    def start(self):
        return int(self.query('DATA:START?'))

    @start.setter
    def start(self, value):
        if value < 1 or value > self.stop:
            print('The start point {} lies outside of the acceptable range, setting it to 1.'.format(value))
            self.start = 1
        else:
            self.write('DATA:START {}'.format(value))

    @property
    def stop(self):
        return int(self.query('DATA:STOP?'))

    @stop.setter
    def stop(self, value):
        if value < self.start or value > self.length:
            print('The end point {} lies outside of the acceptable range, setting it to 1.'.format(value))
            self.stop = 2
        else:
            self.write('DATA:STOP {}'.format(value))

    @property
    def length(self):
        try:
            return int(self.query('WFMO:RECO?'))
        except Exception as e:
            print(e)
            return 1

    @property
    def point_format(self):
        return int(self.query('DATA:WIDTH?'))

    @point_format.setter
    def point_format(self, value):
        if value != POINT_INT8 or value != POINT_INT16:
            print('Invalid data point setting "{}", setting data points to 1 byte.'.format(value))
            self.point_format = POINT_INT8
        else:
            self.write('DATA:WIDTH {}'.format(value))

    @property
    def x_unit(self):
        """
        Gets units of x axis
        """
        return self._query_('WFMO:XUN?')

    @property
    def x_zero(self):
        """
        Gets zero offset of x axis
        """
        return float(self.query('WFMO:XZE?'))

    @property
    def x_spacing(self):
        """
        Gets step size on x axis
        """
        return float(self.query('WFMO:XIN?'))

    @property
    def y_unit(self):
        """
        Gets unit of y axis
        """
        return self._query_('WFMO:YUN?')

    @property
    def y_scale(self):
        """
        Gets scaling factor of y axis
        """
        return float(self.query('WFMOUTPRE:YMULT?'))

    @property
    def y_offset_levels(self):
        """
        Gets y offset
        """
        return float(self.query('WFMO:YOF?'))

    @property
    def y_zero(self):
        """
        Gets zero of y axis
        """
        return float(self.query('WFMO:YZE?'))

    @property
    def vertical_scale(self):
        src = self.source
        if src in self._measurement_channels:
            return float(self.query('{}:SCALE?'.format(src)))
        else:
            return 1

    @vertical_scale.setter
    def vertical_scale(self, value):
        src = self.source
        if src in self._measurement_channels:
            self.write('{}:SCALE {}'.format(src, value))

    @property
    def vertical_position(self):
        src = self.source
        if src in self._measurement_channels:
            return float(self.query('{}:POSITION?'.format(src)))
        else:
            return 0

    @vertical_position.setter
    def vertical_position(self, value):
        src = self.source
        if src in self._measurement_channels:
            self.write('{}:POSITION {}'.format(src, value))

    @property
    def sensitivity(self):
        src = self.source
        if src in self._measurement_channels:
            return float(self.query('{}:VOLTS?'.format(src)))
        else:
            return 0

    @sensitivity.setter
    def sensitivity(self, value):
        src = self.source
        if src in self._measurement_channels:
            self.write('{}:VOLTS {}'.format(src, value))

    @property
    def channel_unity_y(self):
        src = self.source
        if src in self._measurement_channels:
            return self._query_('{}:YUNIT?'.format(src))
        else:
            return None

    @channel_unity_y.setter
    def channel_unity_y(self, value):
        src = self.source
        if src in self._measurement_channels and value in SUPPORTED_UNITS:
            self.write('{}:YUNIT {}'.format(src, value))
        else:
            print('Cannot set "{}" to unit "{}". Supported units are: {}'.format(src, value, SUPPORTED_UNITS))

    @property
    def acquisition_state(self):
        return self._query_('ACQuire:STATE?')

    @acquisition_state.setter
    def acquisition_state(self, value):
        if value in SUPPORTED_ACQUISITION_STATES:
            self.write('ACQuire:STATE {}'.format(value))
        else:
            print('"{}" is not a supported acquisition state. Supported states are: {}'.format(value, SUPPORTED_ACQUISITION_STATES))

    @property
    def acquisition_mode(self):
        return self._query_('ACQuire:MODE?')

    @acquisition_mode.setter
    def acquisition_mode(self, value):
        if value in SUPPORTED_ACQUIRE_MODES:
            self.write('ACQuire:MODE {}'.format(value))
        else:
            print('"{}" is not a supported acquisition mode. Supported modes are: {}'.format(value, SUPPORTED_ACQUIRE_MODES))

    def get_data(self):
        """
        Gets data from the device according to the current settings
        """
        data = self.query('CURV?')

        dx = self.x_spacing
        zx = self.x_zero
        yaxis_scale = self.y_scale
        yaxis_offset = self.y_offset_levels * yaxis_scale
        yaxis_zero = self.y_zero
        y_data = np.fromstring(data, dtype=float, sep=',') * yaxis_scale - yaxis_offset - yaxis_zero
        x_data = np.asarray([i * dx for i in range(len(y_data))]) - zx

        return x_data, y_data

    def run(self):
        self.acquisition_state = 'RUN'
        self.write('ACQuire:STOPAfter RUNSTop')

    def single(self):
        self.acquisition_state = 'RUN'
        self.write('ACQuire:STOPAfter SEQuence')

    def stop_acquisition(self):
        self.acquisition_state = '0'

    def __init__(self, *args):
        super().__init__(*args)

    def _query_(self, message):
        return self._prepare_value_(self.query(message).replace('\n', '').replace('"', ''))

    def _prepare_value_(self, value):
        return value.replace('\n', '').replace('"', '')

    def verify_instance(self, loaded_instance):
        if self.idn[0:len(self._verification_string)] != self._verification_string:
            raise VisaTypeError('Connected device is not a {}.'.format(self._verification_string))

        self.write('WFMO:ENC ASC')  # set output to ASCII
        self.write('WFMI:ENC ASC')  # set input to ASCII


if __name__ == '__main__':
    with TBS2104('129.132.1.157') as tbs:
        x, y = tbs.get_data()

