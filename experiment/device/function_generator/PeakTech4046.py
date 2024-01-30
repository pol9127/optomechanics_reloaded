# author: Andrei Militaru
# date: 24th November 2021

import pyvisa
import time


def query(instrument, scpi):
    """
    Written by Massi
    Query an instrument, it returns the response if scpi asked for information.
    --------------------
    instrument : instance of pyvisa.resources
    scpi : string
    --------------------
    message : bytes
    """
    instrument.write(scpi)
    if '?' in scpi:
        message = b''
        while True:
            charac = instrument.read_bytes(1)
            message += charac
            if charac == b'\n':
                break
        return message
    else:
        return b'-1'


class PeakTech4046:
    """
    Written by Andrei.
    Class able to control function generators model PeakTech 4046.
    So far, only a limited subset of the capabilities have been implemented.
    More properties and options will be added gradually, as soon as the need arises.
    """

    def __init__(self, device):
        """
        :param device: str, MAC address of desired instrument.
                       For the cryo setup: peaktech-01 is the LO_beam control
                                           peaktech-02 is the (former) trigger control
                                           peaktech-03 is the trap_beam control
        """
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource(device)

    def send_command(self, command):
        response = query(self.instrument, command)
        time.sleep(0.01)
        return response

    def all_on(self):
        """
        Turns on both outputs and the sync signal.
        Warning: the sync signal refers to the last channel
        that has been updated!
        :return: response of instrument.
        """
        command = 'OUTPut1 ON;OUTPut2 ON;OUTP:SYNC ON'
        return self.send_command(command)

    def all_off(self):
        """
        Turns off both outputs and the sync signal.
        :return: response of instrument.
        """
        command = 'OUTPut1 OFF;OUTPut2 OFF;OUTP:SYNC OFF'
        return self.send_command(command)

    @property
    def sync(self):
        command = 'OUTP:SYNC?'
        state = self.send_command(command)
        if state == b'0\n':
            output = False
        elif state == b'1\n':
            output = True
        else:
            raise Exception('Response not recognized.')
        return output

    @sync.setter
    def sync(self, value):
        """
        Switches sync channel on/off.
        :param value: bool, True means ON and False means OFF
        :return: response of instrument
        """
        state = 'ON' if value else 'OFF'
        command = 'OUTP:SYNC ' + state
        self.send_command(command)

    @property
    def out1_frequency(self):
        command = 'source1:frequency?'
        return float(self.send_command(command))

    @out1_frequency.setter
    def out1_frequency(self, new_value):
        """
        Setting the frequency of output1.
        :param new_value: float, value (in Hz) of the output frequency.
        :return:
        """
        if new_value < 0:
            raise Exception('Frequencies must be positive.')
        else:
            command = 'source1:frequency {:f}Hz'.format(new_value)
            self.send_command(command)

    @property
    def out2_frequency(self):
        command = 'source2:frequency?'
        return float(self.send_command(command))

    @out2_frequency.setter
    def out2_frequency(self, new_value):
        """
        Setting the frequency of output2.
        :param new_value: float, value (in Hz) of the output frequency.
        :return:
        """
        if new_value < 0:
            raise Exception('Frequencies must be positive.')
        else:
            command = 'source2:frequency {:f}Hz'.format(new_value)
            self.send_command(command)

    @property
    def out1_waveform(self):
        command = 'source1:function?'
        return self.send_command(command)

    @out1_waveform.setter
    def out1_waveform(self, new_wave):
        """
        Setting the waveform of output1.
        :param new_wave: str, must be in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']
        :return: response of instrument
        """
        if new_wave not in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']:
            raise Exception('Waveform not recognized.')
        else:
            command = 'source1:function ' + new_wave
            self.send_command(command)

    @property
    def out2_waveform(self):
        command = 'source2:function?'
        return self.send_command(command)

    @out2_waveform.setter
    def out2_waveform(self, new_wave):
        """
        Setting the waveform of output2.
        :param new_wave: str, must be in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']
        :return: response of instrument
        """
        if new_wave not in ['sinusoid', 'square', 'ramp', 'pulse', 'noise']:
            raise Exception('Waveform not recognized.')
        else:
            command = 'source2:function ' + new_wave
            self.send_command(command)

    @property
    def out1(self):
        command = 'OUTPut1?'
        state = self.send_command(command)
        if state == b'0\n':
            output = False
        elif state == b'1\n':
            output = True
        else:
            raise Exception('Response not recognized.')
        return output

    @out1.setter
    def out1(self, value):
        """
        Switches output1 on/off.
        :param value: bool, True means ON and False means OFF.
        :return: response of the instrument.
        """
        state = 'ON' if value else 'OFF'
        command = 'OUTPut1 ' + state
        self.send_command(command)

    @property
    def out2(self):
        command = 'OUTPut2?'
        state = self.send_command(command)
        if state == b'0\n':
            output = False
        elif state == b'1\n':
            output = True
        else:
            raise Exception('Response not recognized.')
        return output

    @out2.setter
    def out2(self, value):
        """
        Switches output2 on/off.
        :param value: bool, True means ON and False means OFF.
        :return: response of the instrument.
        """
        state = 'ON' if value else 'OFF'
        command = 'OUTPut2 ' + state
        self.send_command(command)

    @property
    def out2_amplitude(self):
        command = 'SOURce2:VOLT?'
        rsp = self.send_command(command)
        return float(rsp)

    @out2_amplitude.setter
    def out2_amplitude(self, value):
        """
        Amplitude (in volts) of output2.
        :param value: float, desired amplitude in V.
        :return: response of the instrument.
        """
        command = 'SOURce2:VOLT {:f}Vpp'.format(value)
        self.send_command(command)

    @property
    def out1_amplitude(self):
        command = 'SOURce1:VOLT?'
        rsp = self.send_command(command)
        return float(rsp)

    @out1_amplitude.setter
    def out1_amplitude(self, value):
        """
        Amplitude (in volts) of output1.
        :param value: float, desired amplitude in V.
        :return: response of the instrument.
        """
        command = 'SOURce1:VOLT {:f}Vpp'.format(value)
        self.send_command(command)

    @property
    def out2_phase(self):
        command = 'SOURce2:PHAS?'
        rsp = self.send_command(command)
        return float(rsp)

    @out2_phase.setter
    def out2_phase(self, value):
        """
        Phase delay (in deg) of output2.
        :param value: float, desired phase in deg.
        :return: response of the isntrument.
        """
        command = 'SOURce2:PHAS {:f}deg'.format(value)
        self.send_command(command)

    @property
    def out1_phase(self):
        command = 'SOURce1:PHAS?'
        rsp = self.send_command(command)
        return float(rsp)

    @out1_phase.setter
    def out1_phase(self, value):
        """
        Phase delay (in deg) of output1.
        :param value: float, desired phase in deg.
        :return: response of the instrument.
        """
        command = 'SOURce1:PHAS {:f}deg'.format(value)
        self.send_command(command)

    @property
    def out1_pulse_width(self):
        command = 'source1function:pulse:width?'
        rsp = self.send_command(command)
        return float(rsp)

    @out1_pulse_width.setter
    def out1_pulse_width(self, value):
        """
        Pulse width (in seconds) of output1. Useful when
        the output waveform is Pulse mode.
        :param value: float, desired pulse width in s.
        :return: response of the instrument.
        """
        command = 'source1:function:pulse:width {:f}s'.format(value)
        self.send_command(command)

    @property
    def out2_pulse_width(self):
        command = 'source2function:pulse:width?'
        rsp = self.send_command(command)
        return float(rsp)

    @out2_pulse_width.setter
    def out2_pulse_width(self, value):
        """
        Pulse width (in seconds) of output2. Useful when
        the output waveform is Pulse mode.
        :param value: float, desired pulse width in s.
        :return: response of the instrument.
        """
        command = 'source2:function:pulse:width {:f}s'.format(value)
        self.send_command(command)
