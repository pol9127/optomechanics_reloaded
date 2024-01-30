from __future__ import division, unicode_literals

import visa
from time import sleep


class PeakTech4055MV:
    """Class representing the function generator PeakTech 4055MV.

    Note
    ----
    Communication is established by PyVISA through the NI VISA
    library. Refer to the Programmers Guide of the function
    generator on how to set it up (see CD).
    The connection might be tested using VISA Interactive Control
    from NI.

    The commands use strings or values given in the units (s, Hz, V,
    dB, %).

    String values are also defined as constants.
    """

    MIN = 'min'
    MAX = 'max'
    AUTO = 'auto'

    def __init__(self, address):
        """Initialize the USB connection to the function generator.

        Parameters
        ----------
        address : str
            Address of the devices as stated in the NI VISA program.
        """

        rm = visa.ResourceManager()
        self.device = rm.open_resource(address)

    def __del__(self):
        self.close()

    def open(self):
        """Open USB connection to the function generator."""
        self.device.open()

    def close(self):
        """Close USB connection to the function generator and return
        to local control."""
        self.local_control()
        self.device.close()

    # OUTPUT CONFIGURATION COMMANDS
    @property
    def output(self):
        return bool(int(self.device.query('output:state?')[1]))

    @output.setter
    def output(self, state):
        if state:
            self.device.write('output:state on')
        else:
            self.device.write('output:state off')

    @property
    def polarity(self):
        return self.device.query('output:polarity?')[1:-2]\
            .lower().strip()

    @polarity.setter
    def polarity(self, polarity_state):
        self.device.write('output:polarity {}'.format(polarity_state))

    NORMAL = 'norm'
    INVERTED = 'inv'

    @property
    def function(self):
        return self.device.query('function?')[1:-2].lower().strip()

    @function.setter
    def function(self, function_string):
        self.device.write('function {:s}'.format(function_string))

    SINE = 'sin'
    SQUARE = 'squ'
    RAMP = 'ramp'
    NOISE = 'noise'
    POSITIVE_PULSE = 'ppuls'
    NEGATIVE_PULSE = 'npuls'
    STAIR = 'stair'
    HALF_SINE = 'hsine'
    LIMITED_SINE = 'lsine'
    EXPONENTIAL = 'rexp'
    LOGARITHM = 'rlog'
    TANGENT = 'tang'
    SINC = 'sinc'
    HALF_ROUND = 'round'
    CARDIAC = 'card'
    QUAKE = 'quake'

    @property
    def frequency(self):
        return float(self.device.query('frequency?')[1:-2])

    @frequency.setter
    def frequency(self, frequency):
        self.device.write('frequency {}'.format(frequency))

    @property
    def period(self):
        return float(self.device.query('period?')[1:-2])

    @period.setter
    def period(self, period):
        self.device.write('period {}'.format(period))

    @property
    def amplitude(self):
        return float(self.device.query('voltage:amplitude?')[1:-2])

    @amplitude.setter
    def amplitude(self, amplitude):
        self.device.write('voltage:amplitude {}'.format(amplitude))

    @property
    def offset(self):
        return float(self.device.query('voltage:offset?')[1:-2])

    @offset.setter
    def offset(self, offset):
        self.device.write('voltage:offset {}'.format(offset))

    @property
    def attenuation(self):
        try:
            return float(
                self.device.query('voltage:attenuation?')[1:-2])
        except ValueError:
            return self.device.query('voltage:attenuation?')[1:-2]\
                .lower().strip()

    @attenuation.setter
    def attenuation(self, attenuation):
        self.device.write('voltage:attenuation {}'.format(attenuation))

    @property
    def voltage_unit(self):
        return self.device.query('voltage:unit?')[1:-2].lower().strip()

    # TODO: Subsequent query returns last set voltage_unit
    @voltage_unit.setter
    def voltage_unit(self, voltage_unit):
        self.device.write('voltage:unit {}'.format(voltage_unit))

    VPP = 'vpp'
    VRMS = 'vrms'

    @property
    def duty_cycle(self):
        return float(self.device.query('function:square:dcycle?')[1:-2])

    @duty_cycle.setter
    def duty_cycle(self, duty_cycle):
        self.device.write(
            'function:square:dcycle {}'.format(duty_cycle))

    @property
    def ramp_symmetry(self):
        return float(self.device.query('function:ramp:symmetry?')[1:-2])

    @ramp_symmetry.setter
    def ramp_symmetry(self, symmetry):
        self.device.write('function:ramp:symmetry {}'.format(symmetry))

    # SYSTEM
    def reset(self):
        self.device.write('*RST')

    def local_control(self):
        self.device.write('system:local')

    def error_queue(self):
        return self.device.query('system:error?')[1:-2]

    def clear_error_queue(self):
        self.device.write('*CLS')

    # FREQUENCY MODULATION
    @property
    def fm_state(self):
        return bool(int(self.device.query('fm:state?')[1:-2]))

    @fm_state.setter
    def fm_state(self, state):
        if state:
            self.device.write('fm:state on')
        else:
            self.device.write('fm:state off')

    @property
    def fm_function(self):
        return self.device.query('fm:internal:function?')[1:-2]\
            .lower().strip()

    @fm_function.setter
    def fm_function(self, function_string):
        self.device.write('fm:internal:function {:s}'.format(
            function_string))

    @property
    def fm_frequency(self):
        return float(self.device.query('fm:internal:frequency?')[1:-2])

    @fm_frequency.setter
    def fm_frequency(self, frequency):
        self.device.write('fm:internal:frequency {}'.format(frequency))

    @property
    def fm_deviation(self):
        return float(self.device.query('fm:deviation?')[1:-2])

    @fm_deviation.setter
    def fm_deviation(self, deviation):
        self.device.write('fm:deviation {}'.format(deviation))

    # AMPLITDUE MODULATION
    @property
    def am_state(self):
        return bool(int(self.device.query('am:state?')[1:-2]))

    @am_state.setter
    def am_state(self, state):
        if state:
            self.device.write('am:state on')
        else:
            self.device.write('am:state off')

    @property
    def am_function(self):
        return self.device.query('am:internal:function?')[1:-2]\
            .lower().strip()

    @am_function.setter
    def am_function(self, function_string):
        self.device.write('am:internal:function {:s}'.format(
            function_string))

    @property
    def am_frequency(self):
        return float(self.device.query('am:internal:frequency?')[1:-2])

    @am_frequency.setter
    def am_frequency(self, frequency):
        self.device.write('am:internal:frequency {}'.format(frequency))

    @property
    def am_depth(self):
        return float(self.device.query('am:depth?')[1:-2])

    @am_depth.setter
    def am_depth(self, deviation):
        self.device.write('am:depth {}'.format(deviation))

    # PHASE MODULATION
    @property
    def pm_state(self):
        return bool(int(self.device.query('pm:state?')[1:-2]))

    @pm_state.setter
    def pm_state(self, state):
        if state:
            self.device.write('pm:state on')
        else:
            self.device.write('pm:state off')

    @property
    def pm_function(self):
        return self.device.query('pm:internal:function?')[1:-2]\
            .lower().strip()

    @pm_function.setter
    def pm_function(self, function_string):
        self.device.write('pm:internal:function {:s}'.format(
            function_string))

    @property
    def pm_frequency(self):
        return float(self.device.query('pm:internal:frequency?')[1:-2])

    @pm_frequency.setter
    def pm_frequency(self, frequency):
        self.device.write('pm:internal:frequency {}'.format(frequency))

    @property
    def pm_deviation(self):
        return float(self.device.query('pm:deviation?')[1:-2])

    @pm_deviation.setter
    def pm_deviation(self, deviation):
        self.device.write('pm:deviation {}'.format(deviation))

    # PULSE WIDTH MODULATION
    @property
    def pwm_state(self):
        return bool(int(self.device.query('pwm:state?')[1:-2]))

    @pwm_state.setter
    def pwm_state(self, state):
        if state:
            self.device.write('pwm:state on')
        else:
            self.device.write('pwm:state off')

    @property
    def pwm_function(self):
        return self.device.query('pwm:internal:function?')[1:-2]\
            .lower().strip()

    @pwm_function.setter
    def pwm_function(self, function_string):
        self.device.write('pwm:internal:function {:s}'.format(
            function_string))

    @property
    def pwm_frequency(self):
        return float(self.device.query('pwm:internal:frequency?')[1:-2])

    @pwm_frequency.setter
    def pwm_frequency(self, frequency):
        self.device.write('pwm:internal:frequency {}'.format(frequency))

    @property
    def pwm_pulse_width_deviation(self):
        return float(self.device.query('pwm:deviation:dcycle?')[1:-2])

    @pwm_pulse_width_deviation.setter
    def pwm_pulse_width_deviation(self, deviation):
        self.device.write('pwm:deviation:dcycle {}'.format(deviation))

    # FREQUENCY SHIFT KEYING
    @property
    def fsk_state(self):
        return bool(int(self.device.query('fsk:state?')[1:-2]))

    @fsk_state.setter
    def fsk_state(self, state):
        if state:
            self.device.write('fsk:state on')
        else:
            self.device.write('fsk:state off')

    @property
    def fsk_frequency(self):
        return float(self.device.query('fsk:frequency?')[1:-2])

    @fsk_frequency.setter
    def fsk_frequency(self, frequency):
        self.device.write('fsk:frequency {}'.format(frequency))

    @property
    def fsk_rate(self):
        return float(self.device.query('fsk:internal:rate?')[1:-2])

    @fsk_rate.setter
    def fsk_rate(self, rate):
        self.device.write('fsk:internal:rate {}'.format(rate))

    @property
    def fsk_source(self):
        return self.device.query('fsk:source?')[1:-2].lower().strip()

    @fsk_source.setter
    def fsk_source(self, source):
        self.device.write('fsk:source {}'.format(source))

    # SWEEP
    @property
    def sweep_state(self):
        return bool(int(self.device.query('sweep:state?')[1:-2]))

    @sweep_state.setter
    def sweep_state(self, state):
        if state:
            self.device.write('sweep:state on')
        else:
            self.device.write('sweep:state off')

    @property
    def sweep_range(self):
        sweep_range = list()
        sweep_range.append(float(
            self.device.query('frequency:start?')[1:-2]))
        sweep_range.append(float(
            self.device.query('frequency:stop?')[1:-2]))
        return sweep_range

    @sweep_range.setter
    def sweep_range(self, sweep_range):
        self.device.write('frequency:start {}'.format(sweep_range[0]))
        sleep(0.2)
        self.device.write('frequency:stop {}'.format(sweep_range[1]))

    @property
    def sweep_spacing(self):
        return self.device.query('sweep:spacing?')[1:-2].lower().strip()

    @sweep_spacing.setter
    def sweep_spacing(self, sweep_spacing):
        self.device.write('sweep:spacing {}'.format(sweep_spacing))

    LIN = 'lin'
    LOG = 'log'

    @property
    def sweep_time(self):
        return float(self.device.query('sweep:time?')[1:-2])

    @sweep_time.setter
    def sweep_time(self, sweep_time):
        self.device.write('sweep:time {}'.format(sweep_time))

    # BURST
    @property
    def burst_state(self):
        return bool(int(self.device.query('burst:state?')[1:-2]))

    @burst_state.setter
    def burst_state(self, state):
        if state:
            self.device.write('burst:state on')
        else:
            self.device.write('burst:state off')

    @property
    def burst_cycles(self):
        return int(float(self.device.query('burst:ncycles?')[1:-2]))

    @burst_cycles.setter
    def burst_cycles(self, cycles):
        self.device.write('burst:ncycles {}'.format(cycles))

    @property
    def burst_phase(self):
        return float(self.device.query('burst:phase?')[1:-2])

    @burst_phase.setter
    def burst_phase(self, phase):
        self.device.write('burst:phase {}'.format(phase))

    @property
    def burst_period(self):
        return float(self.device.query('burst:internal:period?')[1:-2])

    @burst_period.setter
    def burst_period(self, period):
        self.device.write('burst:internal:period {}'.format(period))

    # TRIGGER
    @property
    def trigger_source(self):
        return self.device.query('trigger:source?')[1:-2]\
            .lower().strip()

    @trigger_source.setter
    def trigger_source(self, source):
        self.device.write('trigger:source {}'.format(source))

    IMMEDIATE = 'imm'
    EXTERNAL = 'ext'

    def trigger(self):
        self.device.write('*TRG')
