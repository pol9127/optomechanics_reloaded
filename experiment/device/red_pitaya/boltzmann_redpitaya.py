"""
@author: Andrei Militaru
@date: 30th January 2020
"""

import sys
from io import StringIO
from warnings import warn

from paramiko import client


def data_parser(data):
    chunks = data.split(' ')
    outputs = []
    for chunk in chunks:
        try:
            outputs.append(int(chunk))
        except:
            pass
    channel1, channel2 = outputs[::2], outputs[1::2]
    return channel1, channel2


class SSH:
    """
    Class that opens an ssh connection with a client, sends commands and receives information.
    """

    def __init__(self, address, username, password):
        """
        --------------------
        Parameters:
            address: str,
                Address of the target device.
            username: str,
                Username connecting to the target device.
            password: str,
                Password corresponding to the user.
        -------------------
        """
        self.client = client.SSHClient()
        self.client.set_missing_host_key_policy(client.AutoAddPolicy())
        self.client.connect(address, username=username, password=password, look_for_keys=False)
        self.stored_data = 1

    def __del__(self):
        self.client.close()

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()
        self.client.__exit__(type, value, traceback)

    def sendCommand(self, command, store_lines=False, verbose=False):
        if self.client:
            stdin, stdout, stderr = self.client.exec_command(command)
            while not stdout.channel.exit_status_ready():
                # Print data when available
                if stdout.channel.recv_ready():
                    alldata = stdout.channel.recv(1024)
                    prevdata = b"1"
                    while prevdata:
                        prevdata = stdout.channel.recv(1024)
                        alldata += prevdata
                    if verbose:
                        print(str(alldata, "utf8"))
            if store_lines:
                self.stored_data = stdout.readlines()
        else:
            raise Exception('Connection not opened.')


class Redpitaya(SSH):
    """
    Child class of SSH. It automatically opens a connection with the
    requested Redpitaya and provides methods for initializing a new bitfile
    and for changing the values on the registers.
    """

    def __init__(self, address, username, password):
        """
        See parent class for information about input parameters.
        """
        self.address = address
        self.username = username
        self.password = password
        super().__init__(address, username, password)

    @property
    def bitfile(self):
        return self._bitfile

    @bitfile.setter
    def bitfile(self, new_bitfile):
        self._bitfile = new_bitfile
        self.load_bitfile(new_bitfile)

    @staticmethod
    def gpio_address(gpio):
        if gpio == 0:
            return '0x41200000'
        elif gpio == 1:
            return '0x41210000'
        elif gpio == 2:
            return '0x41220000'
        elif gpio == 3:
            return '0x41230000'
        elif gpio == 4:
            return '0x41240000'
        elif gpio == 5:
            return '0x41250000'
        else:
            raise Exception('Requested GPIO not available. Only choice are 0, 1, 2, 3, 4, 5.')

    @staticmethod
    def signed2unsigned(n, bits=16, base=2):
        """
        Uses the twos complement to return the negative value in the decimal base.
        --------------------
        Parameters:
            n: int,
                Number to be converted in twos complement.
            bits: int, optional,
                Number of bits (or digits more generally) used to encode the number.
                Defaults to 16.
            base: int, optional
                Base used for the computation of the twos complement.
                Defaults to 2.
        --------------------
        Returns:
            If n is negative, returns the twos complement.
        --------------------
        """
        if 0 <= n < base ** (bits - 1):
            return n
        elif 0 > n >= -base ** (bits - 1):
            return base ** bits + 1 + n
        else:
            raise Exception('Number of bits too low to encoding the input.')

    def update_GPIO(self, GPIO, new_value):
        register = self.gpio_address(GPIO)
        command = '/opt/redpitaya/bin/monitor ' + register + ' ' + str(new_value)
        self.sendCommand(command)

    def load_bitfile(self, bitfile):
        command = 'cat ' + bitfile + ' > /dev/xdevcfg'
        self.sendCommand(command)


class SwimmerManager(Redpitaya):
    """
    Class that takes care of the parameters needed in the Mexican Swimmer project.
    Every parameter is a property whose setter automatically changes the registers on the Redpitaya.
    """

    def __init__(self,
                 address='red-pitaya-12.ee.ethz.ch',
                 username='root',
                 password='root',
                 bitfile='bitfiles/mexican_swimmer_cos.bit',
                 gain=1,
                 output1=4,
                 output2=0,
                 input_to_use=1,  # 1 means input 1 and 0 means input 2
                 divider=1,
                 DC=0,
                 DC_adjuster=0,
                 cosine_adjuster=0,
                 sine_adjuster=None,
                 load_bitfile=True):
        """
        ---------------------------
        Parameters:
            address: see SSH class,
                Defaults to 'red-pitaya-12.ee.ethz.ch'.
            username: see SSH class,
                Defaults to 'root'.
            password: see SSH class,
                Defaults to 'root'.
            bitfile: str, optional
                String that contains the path to the bitfile that needs to be loaded.
                Defaults to 'bitfiles/mexican_swimmer_cos.bit'
            gain: float, optional
                Rotational damping, values from 0 to 1.
                Defaults to 1.
            output1: int, optional
                Output of the Digital Analog Converter1:
                    - 0 --> input signal
                    - 1 --> DC value
                    - 2 --> wrapped Wiener process
                    - 3 --> sine
                    - 4 --> cosine
                    - 5 --> amplified wrapped Wiener process
                    - 6 --> amplified rescaled input
                    - 7 --> input b
                Defaults to 4.
            output2, int, optional
                Output of the Digital Analog Converter2 (see output1).
                Defaults to 1.
            input_to_use: int, optional
                Input channel where the white noise is present.
                1 for Input1 (A) and 0 for Input2 (B).
                Defaults to 1.
            divider: int, optional
                A counter is used to down sample the input noise, thus allowing
                a lower cutoff frequency of the active noise.
                Defaults to 1.
            DC: int, optional
                DC values given for the output option "1". Values can be between
                -2**13 and 2**13-1, defaults to 0.
            DC_adjuster: int, optional
                DC adjuster, a redundant degree of freedom that can perhaps reduce the noise when relevant.
                Defaults to 0.
            cosine_adjuster: int, optional
                Attenuator factor of the cosine output. Options from 0 to 7 (meant as exponents of 2).
                Defaults to 0.
            sine_adjuster: int, optional
                Attenuator factor of the sine output. Options from 0 to 7 (meant as exponents of 2).
                If None, the same value of cosine_adjuster is used.
            load_bitfile: bool, optional
                If True, the bitfile is loaded on the Redpitaya. Defaults to True.
        ---------------------------
        """
        super().__init__(address, username, password)
        if load_bitfile:
            self.bitfile = bitfile
        else:
            self._bitfile = bitfile
        self.gain = gain
        self.divider = divider
        self._output1 = output1
        self._output2 = output2
        self._DC = DC
        self._DC_adjuster = DC_adjuster
        self._cosine_adjuster = cosine_adjuster
        if sine_adjuster is None:
            self._sine_adjuster = cosine_adjuster
        else:
            self._sine_adjuster = sine_adjuster
        self._input_to_use = input_to_use
        self.update_gpio1()

    def update_gpio1(self):
        adjusted_DC = self.signed2unsigned(self.DC, bits=14)
        converted_to_bits = self.output1*2**1 + self.input_to_use + self.output2*2**4 + self.DC_adjuster*2**27 +(
                            adjusted_DC*2**7 + self.cosine_adjuster*2**21 + self.sine_adjuster*2**24)
        self.update_GPIO(1, converted_to_bits)

    @property
    def DC_adjuster(self):
        return self._DC_adjuster

    @DC_adjuster.setter
    def DC_adjuster(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value < 8:
            raise Exception('Values of DC_adjuster available only between 0 and 7.')
        else:
            self._DC_adjuster = safe_value
            self.update_gpio1()

    @property
    def cosine_adjuster(self):
        return self._cosine_adjuster

    @cosine_adjuster.setter
    def cosine_adjuster(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value < 8:
            raise Exception('Values of cosine_adjuster available only between 0 and 7.')
        else:
            self._cosine_adjuster = safe_value
            self.update_gpio1()

    @property
    def sine_adjuster(self):
        return self._sine_adjuster

    @sine_adjuster.setter
    def sine_adjuster(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value < 8:
            raise Exception('Values of sine_adjuster available only between 0 and 7.')
        else:
            self._sine_adjuster = safe_value
            self.update_gpio1()

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, new_value):
        if not 0 <= new_value <= 1:
            raise Exception('Gain can be only between 0 and 1.')
        else:
            if new_value < 0.4:
                warn('Usually the behaviour is not correct for gain values smaller than 0.4.')
            else:
                self._gain = new_value
                converted_to_bits = int(round(new_value * 2 ** 32))
                self.update_GPIO(0, converted_to_bits)

    @property
    def divider(self):
        return self._divider

    @divider.setter
    def divider(self, new_value):
        if new_value < 1:
            raise Exception('Divider must be bigger than 1.')
        else:
            safe_value = int(new_value)
            self._divider = safe_value
            self.update_GPIO(2, safe_value)

    @property
    def DC(self):
        return self._DC

    @DC.setter
    def DC(self, new_value):
        safe_value = int(new_value)
        if not -2**13 <= safe_value <= 2**13-1:
            raise Exception('DC value can only go from ' + str(-2**13) + ' to ' + str(2**13-1) + '.')
        else:
            self._DC = safe_value
            self.update_gpio1()

    @property
    def output1(self):
        return self._output1

    @output1.setter
    def output1(self, new_value):
        safe_new_value = int(new_value)
        if not 0 <= safe_new_value <= 7:
            raise Exception('Output channels available only from 0 to 7.')
        else:
            self._output1 = safe_new_value
            self.update_gpio1()

    @property
    def output2(self):
        return self._output2

    @output2.setter
    def output2(self, new_value):
        safe_new_value = int(new_value)
        if not 0 <= safe_new_value <= 7:
            raise Exception('Output channels available only from 0 to 7.')
        else:
            self._output2 = safe_new_value
            self.update_gpio1()

    @property
    def input_to_use(self):
        return self._input_to_use

    @input_to_use.setter
    def input_to_use(self, new_input):
        if not (new_input == 0 or new_input == 1):
            raise Exception('Input values can only be 1 (for input A) or 0 (for input B).')
        else:
            self._input_to_use = new_input
            self.update_gpio1()


class ItoStratoManager(Redpitaya):
    """
    Class that takes care of the parameters needed in the Ito Stratonovich project.
    Every parameter is a property whose setter automatically changes the registers on the Redpitaya.
    """

    def __init__(self,
                 address='red-pitaya-12.ee.ethz.ch',
                 username='root',
                 password='root',
                 bitfile='bitfiles/itostrato2.bit',
                 gain=10000,
                 offset=800,
                 delay=0,
                 final_offset=0,
                 DC=0,
                 coarse_gain=3,
                 gamma=1,
                 output1=1,
                 output2=1,
                 load_bitfile=True):
        """
        ---------------------------
        Parameters:
            address: see SSH class,
                Defaults to 'red-pitaya-12.ee.ethz.ch'.
            username: see SSH class,
                Defaults to 'root'.
            password: see SSH class,
                Defaults to 'root'.
            bitfile: str, optional
                String that contains the path to the bitfile that needs to be loaded.
                Defaults to 'bitfiles/mexican_swimmer_cos.bit'
            gain: int, optional
                Final scaling of noise (16 bits).
                Defaults to 10000.
            offset: int, optional
                Singularity position of the multiplicative noise.
                Defaults to 800.
            delay: int, optional
                Number of delay cycles (in steps of 8ns).
                WARNING: long cables translate in uncontrolled delay of the signal.
                Defaults to 0.
            final_offset: int, optional
                Offset to be applied to the output noise.
                Defaults to 0.
            DC: int, optional
                DC value corresponding to the output channel 7. (1 bit = 122 uV, roughly)
                Defaults to 0.
            coarse_gain: int, optional
                Bit shifts of the final output towards left (corresponding to amplification factors of
                powers of twos).
                Defaults to 3.
            gamma: float, optional
                Digital cutoff of the exponential average filter used to correlate the input noise.
                Defaults to 1 (meaning that no averaging is applied).
            output1: int, optional
                Output of the Digital Analog Converter1:
                    - 0 ADC_a
                    - 1 gain*ADC_b*sqrt(abs(ADC_a(delayed) + offset))
                    - 2 ADC_b
                    - 3 ADC_a + offset
                    - 4 sqrt(abs(ADC_a + offset))
                    - 5 sqrt(abs(ADC_a(delayed) + offset))
                    - 6 correlated noise
                    - 7 DC
                Defaults to 1.
            output2: int, optional
                Output of the Digital Analog Converter2 (see output1).
                Defaults to 1.
            load_bitfile: bool, optional
                If True, the bitfile is loaded on the Redpitaya. Defaults to True.
        -------------------------------
        """
        super().__init__(address, username, password)
        if load_bitfile:
            self.bitfile = bitfile
        else:
            self._bitfile = bitfile
        self._gain = gain
        self._offset = offset
        self._delay = delay
        self._final_offset = final_offset
        self._DC = DC
        self._coarse_gain = coarse_gain
        self._gamma = gamma
        self._output1 = output1
        self._output2 = output2
        self.update_gpio0()
        self.update_gpio1()
        self.update_gpio2()

    @staticmethod
    def volt_to_bit(value):
        if value >= 0:
            return int(value * (2 ** 13 - 1))
        else:
            return int(value * (2 ** 13))

    @staticmethod
    def bit_to_volt(value):
        if value >= 0:
            return value / (2 ** 13 - 1)
        else:
            return value / 2 ** 13

    def update_gpio0(self):
        dac1 = self.output1
        dac2 = self.output2
        delta = self.delay
        offset = self.signed2unsigned(self.final_offset, bits=16)
        new_value = dac1 + 2 ** 3 * dac2 + 2 ** 6 * delta + 2 ** 16 * offset
        self.update_GPIO(0, new_value)

    def update_gpio1(self):
        offset = self.signed2unsigned(self.offset, bits=16)
        new_value = self.gain + 2 ** 16 * offset
        self.update_GPIO(1, new_value)

    def update_gpio2(self):
        DC = self.signed2unsigned(self.DC, bits=14)
        gamma = int(self.gamma*2**15)
        new_value = DC + 2 ** 14 * self.coarse_gain + 2 ** 17 * gamma
        self.update_GPIO(2, new_value)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, new_value):
        if not 0 <= new_value <= 1:
            raise Exception('gamma must be between 0 and 1.')
        else:
            self._gamma = new_value
            self.update_gpio2()

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, new_value):
        safe_value = int(new_value)
        if safe_value < 0:
            raise Exception('gain must be positive.')
        elif safe_value > 2 ** 16 - 1:
            raise Exception('gain cannot be more than {:d}, due to limited number of bits.'.format(2 ** 16 - 1))
        else:
            self._gain = safe_value
            self.update_gpio1()

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, new_value):
        safe_value = int(new_value)
        if not -2 ** 15 <= safe_value <= 2 ** 15 - 1:
            raise Exception(
                'offset must be between {:d} and {:d}, due to limited number of bits.'.format(-2 ** 15, 2 ** 15 - 1))
        else:
            self._offset = safe_value
            self.update_gpio1()

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 2 ** 8 - 1:
            raise Exception('delay must be between {:d} and {:d}, due to limited number of bits.'.format(0, 2 ** 8 - 1))
        else:
            self._delay = safe_value
            self.update_gpio0()

    @property
    def final_offset(self):
        return self._final_offset

    @final_offset.setter
    def final_offset(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 2 ** 16 - 1:
            raise Exception(
                'final_offset must be between {:d} and {:d}, due to limited number of bits.'.format(0, 2 ** 16 - 1))
        else:
            self._final_offset = safe_value
            self.update_gpio0()

    @property
    def DC(self):
        return self._DC

    @DC.setter
    def DC(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 2 ** 14 - 1:
            raise Exception('DC must be between {:d} and {:d}, due to limited number of bits.'.format(0, 2 ** 14 - 1))
        else:
            self._DC = safe_value
            self.update_gpio2()

    @property
    def coarse_gain(self):
        return self._coarse_gain

    @coarse_gain.setter
    def coarse_gain(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 2 ** 3 - 1:
            raise Exception(
                'coarse_grain must be between {:d} and {:d}, due to limited number of bits.'.format(0, 2 ** 3 - 1))
        else:
            self._coarse_gain = safe_value
            self.update_gpio2()

    @property
    def output1(self):
        return self._output1

    @output1.setter
    def output1(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 7:
            raise Exception('Output must be between {:d} and {:d}.'.format(0, 7))
        else:
            self._output1 = safe_value
            self.update_gpio0()

    @property
    def output2(self):
        return self._output2

    @output2.setter
    def output2(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 7:
            raise Exception('Output must be between {:d} and {:d}.'.format(0, 7))
        else:
            self._output2 = safe_value
            self.update_gpio0()


class KovacsManager(Redpitaya):
    """
    Class that takes care of the parameters needed in the Kovacs project.
    Every parameter is a property whose setter automatically changes the registers on the Redpitaya.
    """

    def __init__(self,
                 address='red-pitaya-12.ee.ethz.ch',
                 username='root',
                 password='root',
                 bitfile='bitfiles/kovacs.bit',
                 timescale0=2e-3,
                 timescale1=25e-6,
                 protocol=0,
                 DC=0,
                 hot_rescaling=1.0,
                 warm_rescaling=0.7071,
                 cold_rescaling=0.0,
                 load_bitfile=True):
        """
        ---------------------------
        Parameters:
            address: see SSH class,
                Defaults to 'red-pitaya-12.ee.ethz.ch'.
            username: see SSH class,
                Defaults to 'root'.
            password: see SSH class,
                Defaults to 'root'.
            bitfile: str, optional
                String that contains the path to the bitfile that needs to be loaded.
                Defaults to 'bitfiles/kovacs.bit'
            timescale0: float, optional
                Time interval of the equilibration timescale (in seconds).
                Defaults to 2ms.
            timescale1: float, optional
                Time interval of the short timescale (in seconds).
                Defaults to 25us.
            protocol: int, optional
                Protocol to be run:  0 provides a constant DC value.
                                     1 alternates between hot and cold temperature states.
                                     2 is the proper Kovacs protocol, with 3 values of the temperature.
                                     3 same as protocol=1 but with medium-hot temperature instead of hot.
                                     4 switching between hot temperature and medium temperature.
                                     5 inverse Kovacs protocol: cold-hot-warm
                Defaults to 0.
            DC: int, optional
                DC value corresponding to the output channel 7. (1 bit = 122 uV, roughly)
                Defaults to 0.
            hot_rescaling: float, optional
                rescaling factor (between 0 and 1) that is applied to the hot temperature signal.
                Defaults to 1.
            warm_rescaling: float, optional
                rescaling factor (between 0 and 1) that is applied to the warm temperature signal.
                Defaults to 0.7071, approximately sqrt(1/2).
            cold_rescaling: float, optional
                rescaling factor (between 0 and 1) that is applied to the cold temperature signal.
                Defaults to 0, meaning that the lab temperature applies to the particle.
            load_bitfile: bool, optional
                If True, the bitfile is loaded on the Redpitaya. Defaults to True.
        -------------------------------
        """
        super().__init__(address, username, password)
        if load_bitfile:
            self.bitfile = bitfile
        else:
            self._bitfile = bitfile
        self._timescale0 = timescale0
        self._timescale1 = timescale1
        self._protocol = protocol
        self._DC = DC
        self._cold_rescaling = cold_rescaling
        self._warm_rescaling = warm_rescaling
        self._hot_rescaling = hot_rescaling
        self.update_gpio0()
        self.update_gpio1()
        self.update_gpio2()
        self.update_gpio3()
        self.update_gpio4()

    @staticmethod
    def seconds_to_clock(time):
        """
        Conversion between seconds and clock cycles for a 125MHz frequency.
        """
        return 125e6*time

    @staticmethod
    def volt_to_bit(value):
        if value >= 0:
            return int(value * (2 ** 13 - 1))
        else:
            return int(value * (2 ** 13))

    @staticmethod
    def bit_to_volt(value):
        if value >= 0:
            return value / (2 ** 13 - 1)
        else:
            return value / 2 ** 13

    def update_gpio0(self):
        timescale0_bits = self.seconds_to_clock(self.timescale0)
        self.update_GPIO(0, timescale0_bits)

    def update_gpio1(self):
        timescale1_bits = self.seconds_to_clock(self.timescale1)
        self.update_GPIO(1, timescale1_bits)

    def update_gpio2(self):
        DC = self.signed2unsigned(self.DC, bits=14)
        new_value = DC + 2 ** 14 * self.protocol
        self.update_GPIO(2, new_value)

    def update_gpio3(self):
        hot_coeff = int( self.hot_rescaling * (2**16-1) )
        warm_coeff = int( self.warm_rescaling * (2**16-1) )
        new_value = hot_coeff + 2 ** 16 * warm_coeff
        self.update_GPIO(3, new_value)

    def update_gpio4(self):
        cold_coeff = int( self.cold_rescaling*(2**16-1) )
        new_value = cold_coeff
        self.update_GPIO(4, new_value)

    @property
    def cold_rescaling(self):
        return self._cold_rescaling

    @cold_rescaling.setter
    def cold_rescaling(self, new_value):
        if not 0.0 <= new_value <= 1.0:
            raise Exception('All rescaling factors must be between 0 and 1.')
        else:
            self._cold_rescaling = new_value
            self.update_gpio4()

    @property
    def warm_rescaling(self):
        return self._warm_rescaling

    @warm_rescaling.setter
    def warm_rescaling(self, new_value):
        if not 0.0 <= new_value <= 1.0:
            raise Exception('All rescaling factors must be between 0 and 1.')
        else:
            self._warm_rescaling = new_value
            self.update_gpio3()

    @property
    def hot_rescaling(self):
        return self._hot_rescaling

    @hot_rescaling.setter
    def hot_rescaling(self, new_value):
        if not 0.0 <= new_value <= 1.0:
            raise Exception('All rescaling factors must be between 0 and 1.')
        else:
            self._hot_rescaling = new_value
            self.update_gpio3()

    @property
    def timescale0(self):
        return self._timescale0

    @timescale0.setter
    def timescale0(self, new_value):
        if not 0 <= new_value < 34:
            raise Exception('Number of bits insufficient for required timescale. Maximum value of 34s.')
        else:
            self._timescale0 = new_value
            self.update_gpio0()

    @property
    def timescale1(self):
        return self._timescale1

    @timescale1.setter
    def timescale1(self, new_value):
        if not 0 <= new_value < 34:
            raise Exception('Number of bits insufficient for required timescale. Maximum value of 34s.')
        else:
            self._timescale1 = new_value
            self.update_gpio1()

    @property
    def protocol(self):
        return self._protocol

    @protocol.setter
    def protocol(self, new_value):
        new_value = int(new_value)
        if new_value not in [0, 1, 2, 3, 4, 5]:
            raise Exception('Only allowed protocols are 0, 1, 2, 3, 4 and 5 (see class documentation for details).')
        else:
            self._protocol = new_value
            self.update_gpio2()

    @property
    def DC(self):
        return self._DC

    @DC.setter
    def DC(self, new_value):
        safe_value = int(new_value)
        if not 0 <= safe_value <= 2 ** 14 - 1:
            raise Exception('DC must be between {:d} and {:d}, due to limited number of bits.'.format(0, 2 ** 14 - 1))
        else:
            self._DC = safe_value
            self.update_gpio2()


class TriggerManager(Redpitaya):

    def __init__(self,
                 address='red-pitaya-23.ee.ethz.ch',
                 username='root',
                 password='root',
                 bitfile='bitfiles/trigger.bit',
                 period=30e-3,
                 beam_off=10e-6,
                 feedback_off=1e-3,
                 delay=1e-6,
                 beam_enable=False,
                 feedback_enable=False,
                 trigger_enable=False,
                 load_bitfile=False):
        """
        Initialization of parameters.
        :param address: str, optional
                        IP address of Redpitaya.
                        Defaults to 'red-pitaya-23.ee.ethz.ch'.
        :param username: str, optional
                         Defaults to 'root'.
        :param password: str, optional
                         Defaults to 'root'.
        :param bitfile: str, optional
                        Path+filename corresponding to the bitstream file
                        that needs to be flashed on the Redpitaya.
                        Note: the bitfile should already be present on the board.
                        Defaults to 'bitfiles/trigger.bit'.
        :param period: float, optional
                       Period of oscillation (in seconds) of the digital output signals.
                       Defaults to 30ms.
        :param beam_off: float, optional
                         Time interval (in seconds) for which the digital output
                         corresponding to the beam trigger is switched up (= beam off).
                         Defaults to 10us.
        :param feedback_off: float, optional
                             Time interval (in seconds) for which the digital output
                             corresponding to the feedback trigger is switched up (= feedback off).
                             Defaults to 1ms.
        :param delay: float, optional
                      Time delay (in seconds) of the beam signal compared to the feedback signal,
                      in other words: how much earlier does the feedback switch off compared to the laser.
                      In principle any timescale should be fine, since all that matters is that the
                      switch of the laser does not introduce strong forces on the particle.
                      It can be negative, meaning that the beam is turned off before.
                      Defaults to 1us.
        :param beam_enable: bool, optional
                            If False, the digital output of the beam is kept at 0V even if the input B is
                            high voltage.
                            Defaults to False.
        :param feedback_enable: bool, optional
                            If False, the digital output of the feedback is kept at 0V even if the input B is
                            high voltage.
                            Defaults to False.
        :param trigger_enable: bool, optional
                            If False, the digital output of the trigger is kept at 0V even if the input B is
                            high voltage.
                            Defaults to False.
        :param load_bitfile: bool, optional
                       If True, the bitfile is flashed on the Redpitaya.
                       Defaults to False.
        """

        super().__init__(address, username, password)
        if load_bitfile:
            self.bitfile = bitfile
        else:
            self._bitfile = bitfile
        self._period = period
        self._beam_off = beam_off
        self._feedback_off = feedback_off
        self._beam_enable = beam_enable
        self._feedback_enable = feedback_enable
        self._trigger_enable = trigger_enable
        self.delay = delay
        self.update_gpio0()
        self.update_gpio1()
        self.update_gpio2()
        self.update_gpio3()
        self.update_gpio4()

    @property
    def sampling_frequency(self):
        return 31.25e6

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        max_val = (2**33-1)/self.sampling_frequency
        if 0 < value < max_val:
            self._period = value
            self.update_gpio0()
            self.update_gpio1()
            self.update_gpio2()
            self.update_gpio3()
        else:
            raise Exception('Maximum period is {:.2f} seconds.'.format(max_val))

    @property
    def beam_off(self):
        return self._beam_off

    @beam_off.setter
    def beam_off(self, value):
        max_value = (2 ** 32 - 1)/self.sampling_frequency
        if 0 < value < max_value:
            self._beam_off = value
            self.update_gpio0()
            self.update_gpio1()
        else:
            raise Exception('Maximum laser switch-off is {:.2f} seconds.'.format(max_value))

    @property
    def feedback_off(self):
        return self._feedback_off

    @feedback_off.setter
    def feedback_off(self, value):
        max_value = (2 ** 32 - 1) / self.sampling_frequency
        if 0 < value < max_value:
            self._feedback_off = value
            self.update_gpio2()
            self.update_gpio3()
        else:
            raise Exception('Maximum feedback switch-off is {:.2f} seconds.'.format(max_value))

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, value):
        max_delay = (2**9-1)/self.sampling_frequency
        if 0 < abs(value) < max_delay:
            self._delay = value
            self.update_gpio4()
        else:
            raise Exception('Maximum delay is {:.2f} seconds.'.format(max_delay))

    @property
    def period_cycles(self):
        return int(self.sampling_frequency * self.period)

    @property
    def beam_counter(self):
        # on_duration = self.period - self.beam_off
        # on_cycles = int(on_duration * self.sampling_frequency)
        on_cycles = self.period_cycles - self.beam_up_counter
        return on_cycles

    @property
    def beam_up_counter(self):
        off_cycles = int(self.beam_off * self.sampling_frequency)
        return off_cycles

    @property
    def feedback_counter(self):
        # on_duration = self.period - self.feedback_off
        # on_cycles = int(on_duration * self.sampling_frequency)
        on_cycles = self.period_cycles - self.feedback_up_counter
        return on_cycles

    @property
    def feedback_up_counter(self):
        off_cycles = int(self.feedback_off * self.sampling_frequency)
        return off_cycles

    @property
    def beam_enable(self):
        return self._beam_enable

    @beam_enable.setter
    def beam_enable(self, new_value):
        if type(new_value) is not bool:
            new_value = True if (int(new_value) != 0) else False
        self._beam_enable = new_value
        self.update_gpio4()

    @property
    def feedback_enable(self):
        return self._feedback_enable

    @feedback_enable.setter
    def feedback_enable(self, new_value):
        if type(new_value) is not bool:
            new_value = True if (int(new_value) != 0) else False
        self._feedback_enable = new_value
        self.update_gpio4()

    @property
    def trigger_enable(self):
        return self._trigger_enable

    @trigger_enable.setter
    def trigger_enable(self, new_value):
        if type(new_value) is not bool:
            new_value = True if (int(new_value) != 0) else False
        self._trigger_enable = new_value
        self.update_gpio4()

    def enable_all(self, new_state):
        if type(new_state) is not bool:
            new_state = True if (int(new_state) != 0) else False
        self._beam_enable = new_state
        self._feedback_enable = new_state
        self._trigger_enable = new_state
        self.update_gpio4()

    def update_gpio0(self):
        new_value = self.beam_counter
        self.update_GPIO(0, new_value)

    def update_gpio1(self):
        new_value = self.beam_up_counter
        self.update_GPIO(1, new_value)

    def update_gpio3(self):
        new_value = self.feedback_counter
        self.update_GPIO(3, new_value)

    def update_gpio2(self):
        new_value = self.feedback_up_counter
        self.update_GPIO(2, new_value)

    def update_gpio4(self):
        delay = self.delay
        negativity = 1 if delay < 0 else 0
        cycles = int(abs(delay) * self.sampling_frequency)
        beam_enable_bit = 1 if self.beam_enable else 0
        feedback_enable_bit = 1 if self.feedback_enable else 0
        trigger_enable_bit = 1 if self.trigger_enable else 0
        new_value = cycles + 2**9 * beam_enable_bit + 2**10 * feedback_enable_bit + 2**11 * trigger_enable_bit + 2**12 * negativity
        self.update_GPIO(4, new_value)


class Daq(Redpitaya):

    def __init__(self,
                 address='red-pitaya-06.ee.ethz.ch',
                 username='root',
                 password='root',
                 load_bitfile=False):

        super().__init__(address, username, password)
        if load_bitfile:
            self.reload()

    def reload(self):
        command = 'cat /opt/redpitaya/fpga/fpga_0.94.bit > /dev/xdevcfg'
        self.sendCommand(command)
        return self

    def read_data(self, samples=16384, decimation=8, parsed=False):
        old_stdout = sys.stdout
        temp_stdout = StringIO()
        sys.stdout = temp_stdout
        command = '/opt/redpitaya/bin/acquire ' + str(samples) + ' ' + str(decimation)
        self.sendCommand(command, store_lines=True, verbose=True)
        response = temp_stdout.getvalue()
        sys.stdout = old_stdout
        return data_parser(response) if parsed else response


if __name__ == '__main__':

    params = {'address':'red-pitaya-21.ee.ethz.ch',
              'timescale0':20e-6,
              'timescale1':10e-6,
              'protocol':2,
              'hot_rescaling' : 1,
              'warm_rescaling' : 0.7071,
              'cold_rescaling' : 0,
              'load_bitfile':False}
    with KovacsManager(**params) as tester:
        pass

    print('Done.')
