from __future__ import division
import optomechanics.experiment.lab_control.devices.oscilloscope.gage._gage as _gage
import os
from memory_profiler import profile
from glob import glob
import json
import numpy as np
from copy import copy, deepcopy
from collections import Iterable
from timeit import default_timer
import ast
from collections import OrderedDict

class GageCard(object):
    def initialize(self, board_type=0, channels=0, sample_bits=0, index=0):
        _gage.init(board_type, channels, sample_bits, index)
        _gage.reset()
        self.configuration_order = {'Acquisition': ['Channels', 'SampleRate', 'DepthPostTrigger', 'DepthPreTrigger',
                                                    'SegmentCount', 'ConvertToVolt', 'TriggerDelay', 'TriggerHoldoff',
                                                    'TriggerTimeout', 'TrigEnginesEn', 'TimeStampConfig', 'SampleSize',
                                                    'SampleRes', 'SampleOffset', 'ExtClkSampleSkip', 'ExtClk', 'SampleBits'],
                                    'Channel' : ['InputRange', 'Impedance', 'Term', 'DcOffset', 'Filter'],
                                    'Trigger' : ['Source', 'Level', 'Condition', 'Relation', 'ExtCoupling',
                                                 'ExtImpedance', 'ExtTriggerRange']}
        self.released = False
        self.system_info = deepcopy(_gage.system_info())
        self.num_channels = self.system_info['ChannelCount']
        self.num_triggers = self.system_info['TriggerMachineCount']
        self.ini_location = os.path.dirname(os.path.realpath(__file__))
        ini_files = glob(os.path.join(self.ini_location, '*.ini'))
        if ini_files != []:
            ini_files.sort(key=lambda x: os.path.getmtime(x))
            ini_latest = ini_files[-1]
            try:
                self.read_ini(ini_latest)
            except:
                self.write_ini(os.path.join(self.ini_location, 'default.ini'))
        else:
            self.write_ini(os.path.join(self.ini_location, 'default.ini'))

    def read_ini(self, filename, return_config=False):
        with open(filename) as f:
            if return_config:
                return json.load(f)
            else:
                self.configuration = json.load(f)

    def write_ini(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.configuration, f, indent=4, sort_keys=True)

    def acquire(self):
        _gage.acquire()

    def abort(self):
        _gage.abort()

    # @profile
    def get(self, channel=None, segment=None):
        config = self.configuration
        channel = np.array(ast.literal_eval(config['Acquisition']['Channels']))
        segment = range(1, config['Acquisition']['SegmentCount'] + 1)
        convert_to_volt = config['Acquisition']['ConvertToVolt']
        if convert_to_volt == 'True':
            channel_configs = np.array(config['Channel'])[channel - 1]
            offset = config['Acquisition']['SampleOffset']
            resolution = config['Acquisition']['SampleRes']
            input_range = np.array([cfg['InputRange'] for cfg in channel_configs])
            dc_offset = np.array([cfg['DcOffset'] for cfg in channel_configs])
            factor = input_range / (2 * resolution)
            _gage.wait()
            data_calc = np.swapaxes(np.array([[np.copy(_gage.get(c, s)) for s in segment] for c in channel]).astype(float), 0, 2)
            data_calc -= offset
            data_calc *= -1 * factor

            data_calc += dc_offset
            data = np.swapaxes(data_calc, 1, 2)
        else:
            _gage.wait()
            data = np.swapaxes(np.array([[np.copy(_gage.get(c, s)) for c in channel] for s in segment]), 0, 2)
            # print('offset', config['Acquisition']['SampleOffset'])
            # print('res', config['Acquisition']['SampleRes'])
            # channel_configs = np.array(config['Channel'])[channel - 1]
            # print('range', np.array([cfg['InputRange'] for cfg in channel_configs]))
            # print('dc-offset', np.array([cfg['DcOffset'] for cfg in channel_configs]))
        return data

    def release(self):
        if self.released:
            return
        else:
            _gage.close()
            self.released = True

    def __del__(self):
        self.release()

    @property
    def configuration(self):
        acq_conf = _gage.get_configuration(6, 2)
        trig_conf = [_gage.get_configuration(5, 2, i + 1) for i in range(self.num_triggers)]
        chan_conf = [_gage.get_configuration(4, 2, i + 1) for i in range(self.num_channels)]
        _configuration = {'Acquisition' : OrderedDict([(k, acq_conf[k]) for k in self.configuration_order['Acquisition']]),
                          'Trigger' : [OrderedDict([(k, cfg[k]) for k in self.configuration_order['Trigger']]) for cfg in trig_conf],
                          'Channel' : [OrderedDict([(k, cfg[k]) for k in self.configuration_order['Channel']]) for cfg in chan_conf]}
        return self.configuration_translation(deepcopy(_configuration))

    @configuration.setter
    def configuration(self, config):
        config_copy = deepcopy(config)
        config_trans = self.configuration_translation_inv(config_copy)
        _gage.set_configuration(config_trans)
        config_tmp = self.configuration
        if config != config_tmp:
            print('WARNING: Desired Configuration deviates from Scope Configuration')

    @property
    def configuration_caps(self):
        _caps = {'Acquisition' : {},
                 'Trigger' : [{} for i in range(self.num_triggers)],
                 'Channel' : [{} for i in range(self.num_channels)]}
        possible_modes_num = [2147483648, 1073741824, 4096, 2048, 1024, 512, 128, 8, 4, 2, 1]
        relevant_modes = [4, 2, 1]
        available_modes = []
        rel_modes_bit_mask = _gage.get_caps(0)[0]
        for mode_n in possible_modes_num:
            rel_modes_bit_mask -= mode_n
            if rel_modes_bit_mask >= 0:
                if mode_n in relevant_modes:
                    available_modes.append(mode_n)
            else:
                rel_modes_bit_mask += mode_n

        _caps['Acquisition']['Channels'] = available_modes
        _caps['Acquisition']['ConvertToVolt'] = [0, 1]
        for i in range(self.num_channels):
            _caps['Channel'][i]['InputRange'] = list(_gage.get_caps(1, i + 1))
            _caps['Channel'][i]['Term'] = list(_gage.get_caps(3, i + 1))
            _caps['Channel'][i]['Impedance'] = list(_gage.get_caps(2, i + 1))
        return self.configuration_translation(deepcopy(_caps))

    translation_dict = {'Acquisition' : {},
                        'Trigger' : {},
                        'Channel' : {}}
    translation_dict_inv = {'Acquisition' : {},
                            'Trigger' : {},
                            'Channel' : {}}

    dict_acquisition_mode = {1: '[1]', 2: '[1, 3]', 4: '[1, 2, 3, 4]', 8: 'Oct', 128: 'PowerOn', 512: 'PretTrigMulRec',
                             1024: 'Reference_Clk', 2048: 'Cs3200ClkInvert', 4096: 'SwAveraging', 1073741824: 'User1',
                             2147483648: 'User2'}
    translation_dict['Acquisition']['Channels'] = lambda x, dict_=dict_acquisition_mode: dict_[x]
    translation_dict_inv['Acquisition']['Channels'] = lambda x, dict_=dict_acquisition_mode: {v : k for k,v in dict_.items()}[x]

    dict_channel_term = {1: 'DC', 2: 'AC'}
    translation_dict['Channel']['Term'] = lambda x, dict_=dict_channel_term: dict_[x]
    translation_dict_inv['Channel']['Term'] = lambda x, dict_=dict_channel_term: {v : k for k,v in dict_.items()}[x]

    dict_acquisition_convert_to_volt = {0: 'False', 1: 'True'}
    translation_dict['Acquisition']['ConvertToVolt'] = lambda x, dict_=dict_acquisition_convert_to_volt: dict_[x]
    translation_dict_inv['Acquisition']['ConvertToVolt'] = lambda x, dict_=dict_acquisition_convert_to_volt: {v : k for k,v in dict_.items()}[x]

    def configuration_translation(self, dict_):
        for key_major in self.translation_dict:
            if key_major in dict_:
                for key_minor in self.translation_dict[key_major]:
                    if isinstance(dict_[key_major], list):
                        for i in range(len(dict_[key_major])):
                            if key_minor in dict_[key_major][i]:
                                if isinstance(dict_[key_major][i][key_minor], list):
                                    for j in range(len(dict_[key_major][i][key_minor])):
                                        dict_[key_major][i][key_minor][j] = self.translation_dict[key_major][key_minor](dict_[key_major][i][key_minor][j])
                                else:
                                    dict_[key_major][i][key_minor] = self.translation_dict[key_major][key_minor](dict_[key_major][i][key_minor])

                    else:
                        if key_minor in dict_[key_major]:
                            if isinstance(dict_[key_major][key_minor], list):
                                for j in range(len(dict_[key_major][key_minor])):
                                    dict_[key_major][key_minor][j] = self.translation_dict[key_major][key_minor](dict_[key_major][key_minor][j])
                            else:
                                dict_[key_major][key_minor] = self.translation_dict[key_major][key_minor](dict_[key_major][key_minor])
        return dict_

    def configuration_translation_inv(self, dict_):
        for key_major in self.translation_dict_inv:
            if key_major in dict_:
                for key_minor in self.translation_dict_inv[key_major]:
                    if isinstance(dict_[key_major], list):
                        for i in range(len(dict_[key_major])):
                            if key_minor in dict_[key_major][i]:
                                if isinstance(dict_[key_major][i][key_minor], list):
                                    for j in range(len(dict_[key_major][i][key_minor])):
                                        dict_[key_major][i][key_minor][j] = self.translation_dict_inv[key_major][key_minor](dict_[key_major][i][key_minor][j])
                                else:
                                    dict_[key_major][i][key_minor] = self.translation_dict_inv[key_major][key_minor](dict_[key_major][i][key_minor])
                    else:
                        if key_minor in dict_[key_major]:
                            if isinstance(dict_[key_major][key_minor], list):
                                for j in range(len(dict_[key_major][key_minor])):
                                    dict_[key_major][key_minor][j] = self.translation_dict_inv[key_major][key_minor](dict_[key_major][key_minor][j])
                            else:
                                dict_[key_major][key_minor] = self.translation_dict_inv[key_major][key_minor](dict_[key_major][key_minor])
        return dict_


if __name__ == '__main__':
    Gage = GageCard()
    Gage.initialize()
    config = copy(Gage.configuration_caps)
    print(config)
    # print(Gage.configuration_translation(config))
    Gage.release()
