"""
Written by Andrei on 3rd December 2021.
This class groups together some of the previously sparsed functions that control
the cold damping filter, with the goal of making control scripts cleaner.
Warning: it is meant exclusively for the cryo-trapping setup and it assumes its
specific cables and relais configuration.
"""

from .cold_damp_filter_ctrl import cold_damp_filter_ctrl
from .biquad_calculator import calcBiquad
from ..demodulation_recorder.demodulation_recorder import zhinst_demod_recorder
from warnings import warn
import os


class ColdDamper(cold_damp_filter_ctrl):

    def __init__(self,
                 bitfile='biquad_x6.bit',
                 load_bitfile=False,
                 input_channel=2,
                 mdrec=None,
                 switch='RF1',
                 output1='biquad1',
                 output2='biquad2',
                 zero_output=False,
                 gain=-1000,
                 freq_x=146e3,
                 freq_y=165e3,
                 sign=1,
                 Qx=5,
                 Qy=5,
                 use_mfli_params=True,
                 verbose=True,
                 delay_raw=0,
                 delay_fine=20,
                 enable=False,
                 PLL_enable=True,
                 **kwargs):
        """
        :param bitfile: str, optional.
                        Name of bitfile. Defaults to biquad_x6.bit
        :param load_bitfile: bool, optional
                             If True, the bitfile is loaded. Defaults to False
        :param input_channel: int, optional
                              Input to be used for the filter.
                              Defaults to 2.
        :param mdrec: zhinst.demod.recorder instance, optional
                      It allows the ColdDamper instance to read the resonance
                      frequency from the lock-in box and to activate/deactivate
                      the enable relais. Defaults to None, meaning that its functionalities
                      are not available until a proper instance is provided.
        :param switch: see documentation of cold_damp_filter_ctrl
        :param output1: see documentation of cold_damp_filter_ctrl
        :param output2: see documentation of cold_damp_filter_ctrl
        :param zero_output: bool, optional
                            If True, the outputs are set to zero.
                            Defaults to True.
        :param gain: float, optional
                     gain (in dB) of the filter.
                     Defaults to -1000 (meaning that is is practically turned off).
        :param freq_x: float, optional
                       Value (in Hz) of the x resonance frequency.
                       Defaults to 146e3. It is recommended to read this function
                       directly from the MFLI device, i.e., to set use_mfli_params=True.
        :param freq_y: float, optional
                       Value (in Hz) of the y resonance frequency.
                       Defaults to 165e3. It is recommended to read this function
                       directly from the MFLI device, i.e., to set use_mfli_params=True.
        :param sign: int, either 1 or -1, optional,
                     determines the sign of the feedback gain. It depends on the
                     particle charge, one should always start from lower gains to
                     understand its correct value.
                     Defaults to +1.
        :param Qx: float, optional
                   Q factor of the notch filter for the X mode.
                   Defaults to 5.
        :param Qy: float, optional
                   Q factor of the notch filter for the Y mode.
                   Defaults to 5.
        :param use_mfli_params: bool, optional
                                if True, the parameters freq_x and freq_y are
                                ignored and their value are extracted from the
                                MFLI PID loops (which are assumed to be set
                                correctly).
        :param verbose: bool, optional
                        if True, information about the state of the Redpitaya is
                        provided while the filter parameters are set.
                        Defaults to True.
        :param delay_fine: int, optional
                           clock cycles of delay (1 corresponds to 32ns).
                           Defaults to 20.
        :param delay_raw: int, optional
                          ray delay cycles (1 corresponds to 32x32ns=1024ns).
                          Defaults to 0.
        :param enable: bool, optional
                        if False, the feedback is disconnected from setup via relais.
                        Defaults to False.
        :param PLL_enable: bool, optional
                        if False, the PLL output corresponding to the z peak is turned off.
                        Defaults to True (i.e., initializing an instance does not mess up PLL).
        :param kwargs: address, hostname and password of the Redpitaya of interest.
        """

        super().__init__(**kwargs)
        self.BITFILENAME = bitfile
        self.BITFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.BITFILENAME)
        if load_bitfile:
            self.load_bitfile()
        self.nonzero_output1 = output1
        self.nonzero_output2 = output2
        self._zero_output = zero_output
        self.input_channel = input_channel
        self.mdrec = mdrec
        self.switch = switch
        self.output1 = output1
        self.output2 = output2
        self._enabled = None
        self.use_mfli_params = use_mfli_params
        self.verbose = verbose
        if self.use_mfli_params:
            self.load_from_mfli()
        else:
            self._freq_x = freq_x
            self._freq_y = freq_y
        self._Qx = Qx
        self._Qy = Qy
        self._sign = sign
        self._gain = gain
        self.config_cd_filter(verbose=verbose, use_mfli_params=self.use_mfli_params)
        self.delay_raw = delay_raw
        self.delay_fine = delay_fine
        self.enable = enable
        self.PLL_enable = PLL_enable

    @property
    def PLL_enable(self):
        if self.mdrec is None:
            raise Exception('An instance of zhinst_demod_recorder must be first provided.')
        else:
            return self._PLL_enable

    @PLL_enable.setter
    def PLL_enable(self, new_state):
        if self.mdrec is None:
            raise Exception('An instance of zhinst_demod_recorder must be first provided.')
        else:
            if type(new_state) is not bool:
                new_state = False if (int(new_state) == 0) else True
            self._PLL_enable = new_state
            enable = 1 if new_state else 0
            self.mdrec.lock_in.setInt('/dev3714/sigouts/0/enables/1', enable)
            
    @property
    def delay_raw(self):
        return self._delay_raw

    @delay_raw.setter
    def delay_raw(self, new_value):
        new_value = int(new_value)
        if new_value < 0:
            raise Exception('Delay can only be positive.')
        else:
            self._delay_raw = new_value
            self.set_delay_raw(new_value)

    @property
    def delay_fine(self):
        return self._delay_fine

    @delay_fine.setter
    def delay_fine(self, new_value):
        new_value = int(new_value)
        if new_value < 0:
            raise Exception('Delay can only be positive.')
        else:
            self._delay_fine = new_value
            self.set_delay_fine(new_value)

    @property
    def delay(self):
        return 32e-9*self.delay_fine + 32e-9*32*self.delay_raw

    @property
    def freq_x(self):
        return self._freq_x

    @freq_x.setter
    def freq_x(self, new_value):
        if new_value <= 0:
            raise Exception('Resonance frequency must be greater than 0!')
        else:
            self._freq_x = new_value
            self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    @property
    def freq_y(self):
        return self._freq_y

    @freq_y.setter
    def freq_y(self, new_value):
        if new_value <= 0:
            raise Exception('Resonance frequency must be greater than 0!')
        else:
            self._freq_y = new_value
            self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, new_value):
        if new_value not in [1, -1]:
            raise Exception('sign parameter can only be 1 or -1.')
        else:
            self._sign = new_value
            self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    @property
    def Qx(self):
        return self._Qx

    @Qx.setter
    def Qx(self, new_value):
        if new_value <= 0:
            raise Exception('Quality factor must be greater than 0!')
        else:
            self._Qx = new_value
            self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    @property
    def Qy(self):
        return self._Qy

    @Qy.setter
    def Qy(self, new_value):
        if new_value <= 0:
            raise Exception('Quality factor must be greater than 0!')
        else:
            self._Qy = new_value
            self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, new_value):
        self._gain = new_value
        self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    def config_cd_filter(self, verbose=True, use_mfli_params=False):
        if use_mfli_params:
            self.load_from_mfli()
        gain = self.gain
        x_notch_freq = self.freq_x
        y_notch_freq = self.freq_y
        sign = self.sign
        Qx = self.Qx
        Qy = self.Qy
        settings = self.cd_fc_get_settings(gain)
        add_att_linear = 10 ** (-settings['AddAtten'] / 20)
        if verbose:
            print('Linear Gain in RPi:', add_att_linear)
            print('Amplifier Gain (dB):', settings['Amplifier'])
            print('External Attenuator (dB):', settings['Attenuator'])

        self.configure_biquad(1, *calcBiquad(biquad_type='notch',
                                             Fs=self.fs, Fc=x_notch_freq, Q=Qx, gain=sign))
        self.configure_biquad(2, *calcBiquad(biquad_type='notch',
                                             Fs=self.fs, Fc=y_notch_freq, Q=Qy, gain=add_att_linear))
        self.reset_and_start()
        self.set_amplifier_gain(settings['Amplifier'])
        self.set_attenuator(settings['Attenuator'])
        return self

    def load_new_params(self, gain=None, freq_x=None, freq_y=None,
                        Qx=None, Qy=None, sign=None, delay_fine=None, delay_raw=None):
        warn('No control on the new values is implemented in this method. At your own risk!')
        if gain is not None:
            self._gain = gain
        if freq_x is not None:
            if freq_x > 0:
                self._freq_x = freq_x
            else:
                raise Exception('All values must be positive or in [-1, 1] for the sign.')
        if freq_y is not None:
            if freq_y > 0:
                self._freq_y = freq_y
            else:
                raise Exception('All values must be positive or in [-1, 1] for the sign.')
        if Qx is not None:
            if Qx > 0:
                self._Qx = Qx
            else:
                raise Exception('All values must be positive or in [-1, 1] for the sign.')
        if Qy is not None:
            if Qy > 0:
                self._Qy = Qy
            else:
                raise Exception('All values must be positive or in [-1, 1] for the sign.')
        if sign is not None:
            if sign in [-1, 1]:
                self._sign = sign
            else:
                raise Exception('All values must be positive or in [-1, 1] for the sign.')
        if delay_fine is not None:
            delay_fine = int(delay_fine)
            if delay_fine < 0:
                raise Exception('Delay can only be positive.')
            else:
                self._delay_fine = delay_fine
                self.set_delay_fine(delay_fine)
        if delay_raw is not None:
            delay_raw = int(delay_raw)
            if delay_raw < 0:
                raise Exception('Delay can only be positive.')
            else:
                self._delay_raw = delay_raw
                self.set_delay_raw(delay_raw)
        self.config_cd_filter(verbose=self.verbose, use_mfli_params=self.use_mfli_params)

    def load_from_mfli(self):
        if self.mdrec is None:
            raise Exception('No instance of zhinst.demod.recorder has been provided yet.')
        else:
            self._freq_x = self.mdrec.lock_in.getDouble('/dev3714/pids/2/center')
            self._freq_y = self.mdrec.lock_in.getDouble('/dev3714/pids/3/center')
        return self

    @property
    def zero_output(self):
        return self._zero_output

    @zero_output.setter
    def zero_output(self, new_state):
        if type(new_state) is not bool:
            new_state = False if (int(new_state) == 0) else True
        if new_state:
            self.nonzero_output1 = self.output1
            self.nonzero_output2 = self.output2
            self.set_outputs('biquad6')
            self.configure_biquad(6, 0, 0, 0, 0, 0)
        else:
            self.output1 = self.nonzero_output1
            self.output2 = self.nonzero_output2
        self.reset_and_start()
        self._zero_output = new_state

    @staticmethod
    def cd_fc_get_settings(total_gain_db):
        db_left = total_gain_db
        att = 0
        if total_gain_db < 10:
            att = 36
            db_left += 36
        amplifier_db = int((db_left + 9) / 10) * 10
        if amplifier_db > 60:
            amplifier_db = 60
        elif amplifier_db < 10:
            amplifier_db = 10
        db_left -= amplifier_db
        if db_left > 0:
            raise NotImplementedError('Too much gain, can\'t do!, Gain=', total_gain_db)
        else:
            add_att_db = -db_left
            return {'Attenuator': att,
                    'Amplifier': amplifier_db,
                    'AddAtten': add_att_db}

    @property
    def switch(self):
        return self._switch

    @switch.setter
    def switch(self, new_value):
        try:
            self.set_switch(new_value)
            self._switch = new_value
        except:
            raise Exception('Invalid switch value.')

    @property
    def input_channel(self):
        return self._input_channel

    @input_channel.setter
    def input_channel(self, new_value):
        if new_value not in [1, 2]:
            raise Exception('Only allowed inputs are 1 and 2.')
        else:
            self._input_channel = new_value
            self.set_input(new_value)

    def set_outputs(self, new_value):
        self.output1 = new_value
        self.output2 = new_value
        return self

    @property
    def nonzero_output1(self):
        return self._nonzero_output1

    @nonzero_output1.setter
    def nonzero_output1(self, new_value):
        self._nonzero_output1 = new_value

    @property
    def nonzero_output2(self):
        return self._nonzero_output2

    @nonzero_output2.setter
    def nonzero_output2(self, new_value):
        self._nonzero_output2 = new_value

    @property
    def output1(self):
        return self._output1

    @output1.setter
    def output1(self, new_value):
        if self.zero_output:
            warn('The outputs are set to zero from past intructions. Changing the state...')
            self._zero_output = False
        try:
            self.set_output(1, new_value)
            self._output1 = new_value
        except:
            raise Exception('Invalid output option.')

    @property
    def output2(self):
        return self._output2

    @output2.setter
    def output2(self, new_value):
        if self.zero_output:
            warn('The outputs are set to zero frompast intructions. Changing the state...')
            self._zero_output = False
        try:
            self.set_output(2, new_value)
            self._output2 = new_value
        except:
            raise Exception('Invalid output option.')

    @property
    def enable(self):
        if self.mdrec is None:
            raise Exception('An instance of zhinst_demod_recorder must be first provided.')
        else:
            return self._enable

    @enable.setter
    def enable(self, new_state):
        if self.mdrec is None:
            raise Exception('An instance of zhinst_demod_recorder must be first provided.')
        else:
            if type(new_state) is not bool:
                new_state = False if (int(new_state) == 0) else True
            self._enable = new_state
            enable = 1 if new_state else 0
            self.mdrec.lock_in.setDouble('/dev3714/auxouts/3/offset', enable * 5)
