import numpy as np
import struct
from timeit import default_timer
import pandas as pd
from binascii import hexlify
from datetime import datetime as dt
from optomechanics.post_processing.spectrum import derive_psd
import h5py

class OsciData:
    def __init__(self, filename_, convert_to_volt=False, read_only_header_=False, info=False):
        header_parameters = ['file_version',
                             'crlf1',
                             'name',
                             'crlf2',
                             'comment',
                             'crlf3',
                             'control_z',
                             'sample_rate_index',
                             'operation_mode',
                             'trigger_depth',
                             'trigger_slope',
                             'trigger_source',
                             'trigger_level',
                             'sample_depth',
                             'captured_gain',
                             'captured_coupling',
                             'current_mem_ptr',
                             'starting_address',
                             'trigger_address',
                             'ending_address',
                             'trigger_time',
                             'trigger_date',
                             'trigger_coupling',
                             'trigger_gain',
                             'probe',
                             'inverted_data',
                             'board_type',
                             'resolution_12_bits',
                             'multiple_record',
                             'trigger_probe',
                             'sample_offset',
                             'sample_resolution',
                             'sample_bits',
                             'extended_trigger_time',
                             'imped_a',
                             'imped_b',
                             'external_tbs',
                             'external_clock_rate',
                             'file_options',
                             'version',
                             'eeprom_options',
                             'trigger_hardware',
                             'record_depth',
                             'sample_offset_32',
                             'sample_resolution_32',
                             'multiple_record_count',
                             'dc_offset',
                             'random_factor']

        self.captured_gain_index_conversion = {-28032 : 50,
                                               -28416 : 20,
                                               0 : 10,
                                               1 : 5,
                                               2 : 2,
                                               3 : 1,
                                               4 : 0.5,
                                               5 : 0.2,
                                               6 : 0.1}

        self.trigger_gain_index_conversion = {0 : 10,
                                              1 : 5,
                                              2 : 2,
                                              3 : 1,
                                              4 : 0.5,
                                              5 : 0.2,
                                              6 : 0.1}

        self.probe_index_conversion = {0 : 1,
                                       1 : 10,
                                       2 : 20,
                                       3 : 50,
                                       4 : 100,
                                       5 : 200,
                                       6 : 500,
                                       7 : 1000}


        t1 = default_timer()
        with open(filename_, mode='rb') as f:
            if read_only_header_:
                file_complete = f.read(401)
            else:
                file_complete = f.read()
        t2 = default_timer()

        header_data = struct.unpack('<14s2s9s2s256s2s2s2hi3hi2h4i2s2s4hH6hI2h2fiH3I2iI2h', file_complete[:401])
        self.header = {h_p : h_d for h_p,h_d in zip(header_parameters, header_data)}

        self.sample_rate_index_conversion = {0 : 1,
                                             1 : 2,
                                             2 : 5,
                                             3 : 10,
                                             4 : 20,
                                             5 : 50,
                                             6 : 100,
                                             7 : 200,
                                             8 : 500,
                                             9 : 1000,
                                             10 : 2000,
                                             11 : 5000,
                                             12 : 10000,
                                             13 : 20000,
                                             14 : 50000,
                                             15 : 100000,
                                             16 : 200000,
                                             17 : 500000,
                                             18 : 1000000,
                                             19 : 2000000,
                                             20 : 2500000,
                                             21 : 5000000,
                                             22 : 10000000,
                                             23 : 12500000,
                                             24 : 20000000,
                                             25 : 25000000,
                                             26 : 30000000,
                                             27 : 40000000,
                                             28 : 50000000,
                                             29 : 60000000,
                                             30 : 65000000,
                                             31 : 80000000,
                                             32 : 100000000,
                                             33 : 120000000,
                                             34 : 125000000,
                                             35 : 130000000,
                                             36 : 150000000,
                                             37 : 200000000,
                                             38 : 250000000,
                                             39 : 300000000,
                                             40 : 500000000,
                                             41 : 1000000000,
                                             42 : 2000000000,
                                             43 : 4000000000,
                                             44 : 5000000000,
                                             45 : 8000000000,
                                             46 : 10000000000,
                                             47 : self.header['external_clock_rate']}

        if read_only_header_:
            return
        t3 = default_timer()

        if self.header['resolution_12_bits'] == 1:
            self.measured_data_raw = np.frombuffer(file_complete[512:], dtype=np.int16)
        else:
            self.measured_data_raw = np.frombuffer(file_complete[512:], dtype=np.uint8)

        t4 = default_timer()
        if(convert_to_volt):
            self.measured_data_volt = self.raw_to_volts(self.measured_data_raw)
            t5 = default_timer()
        if info:
            print('##########################################################')
            print('read file in', t2 - t1)
            print('constructed header in ', t3 - t2)
            print('unpacked measured data in ', t4 - t3)
            if(convert_to_volt):
                print('converted to volt in ', t5 - t4)
            print('##########################################################')

    @property
    def sample_data(self):
        sample_rate = self.sample_rate_index_conversion[self.header['sample_rate_index']]
        sample_depth = self.header['sample_depth']
        coupling = {1 : 'DC', 2 : 'AC'}[self.header['captured_coupling']]
        sample_dat = {'sample_rate [Hz]' : sample_rate,
                      'sample_depth' : sample_depth,
                      'coupling' : coupling}
        return sample_dat

    @property
    def channel(self):
        return self.header['name'].split(b'\x00')[0].decode()

    def header_to_csv(self, save_location=None):
        if save_location is not None:
            pd.Series(self.header).to_csv(save_location, sep=';')
        else:
            print('Please specify location to save header to.')

    def raw_to_volts(self, raw):
        sample_offset = self.header['sample_offset']
        sample_resolution = self.header['sample_resolution']
        if sample_offset == 0 or sample_resolution == 0:
            sample_offset = self.header['sample_offset_32']
            sample_resolution = self.header['sample_resolution_32']

        volts = (sample_offset - raw) /  sample_resolution * self.captured_gain_index_conversion[self.header['captured_gain']] + self.header['dc_offset'] * 1000
        return volts

    @property
    def timestamp(self):
        trigger_date_bin = bin(int(hexlify(self.header['trigger_date'][::-1]), 16))[2:].zfill(16)
        day = int(trigger_date_bin[-5:], 2)
        month = int(trigger_date_bin[-9:-5], 2)
        year = int(trigger_date_bin[:-9], 2) + 1980

        trigger_time_bin = bin(int(hexlify(self.header['trigger_time'][::-1]), 16))[2:].zfill(16)
        if self.header['trigger_time'] == 0:
            trigger_time_bin = bin(int(hexlify(self.header['extended_trigger_time'][::-1]), 16))[2:].zfill(32)
            hour = int(trigger_time_bin[:5], 2)
            minute = int(trigger_time_bin[5:11], 2)
            second = int(trigger_time_bin[11:17], 2)
            microsecond = int(trigger_time_bin[17:], 2) * 10
            ts = dt(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=microsecond)
        else:
            second = int(trigger_time_bin[-5:], 2)
            minute = int(trigger_time_bin[-11:-5], 2)
            hour = int(trigger_time_bin[:-11], 2)
            ts = dt(year=year, month=month, day=day, hour=hour, minute=minute, second=second)
        return ts

    @property
    def comment(self):
        return self.header['comment'].split(b'\x00')[0].decode()

    @property
    def trigger_data(self):
        timestamp = self.timestamp
        pre_trigger_samples = self.header['sample_depth'] - self.header['trigger_depth']
        post_trigger_samples = self.header['trigger_depth']
        trigger_slope = {1 : 'positive', 2 : 'negative'}[self.header['trigger_slope']]
        trigger_source = {1 : 'Ch A', 2 : 'Ch B', 3 : 'external', 4 : 'automatic', 5 : 'keyboard'}[self.header['trigger_source']]
        trigger_level = self.header['trigger_level']
        trigger_coupling = {1 : 'DC', 2 : 'AC'}[self.header['trigger_coupling']]
        trigger_gain = self.trigger_gain_index_conversion[self.header['trigger_gain']]
        trigger_dat = {'timestamp' : timestamp,
                       'pre_trigger_samples' : pre_trigger_samples,
                       'post_trigger_samples' : post_trigger_samples,
                       'trigger_slope' : trigger_slope,
                       'trigger_source' : trigger_source,
                       'trigger_level' : trigger_level,
                       'trigger_coupling' : trigger_coupling,
                       'trigger_gain' : trigger_gain}
        return trigger_dat

    @property
    def runtime(self):
        rt = np.linspace(0, self.header['sample_depth'] / self.sample_rate_index_conversion[self.header['sample_rate_index']], self.header['sample_depth'])
        rt -= rt[self.header['sample_depth'] - self.header['trigger_depth']]
        return rt

    @property
    def impedance(self):
        imp_a = {0 : '1 MOhm', 16 : '50 Ohm'}[self.header['imped_a']]
        imp_b = {0 : '1 MOhm', 16 : '50 Ohm'}[self.header['imped_b']]
        return {'Channel A' : imp_a, 'Channel B' : imp_b}

    def power_spectral_density(self, **kwargs):
        from optomechanics.post_processing.spectrum import derive_psd
        psd, frequency = derive_psd(self.measured_data_raw, self.sample_rate_index_conversion[self.header['sample_rate_index']], **kwargs)
        sample_resolution = self.header['sample_resolution']
        if sample_resolution == 0 or self.header['sample_offset'] == 0:
            sample_resolution = self.header['sample_resolution_32']
        psd_scaling = -1 / sample_resolution * self.captured_gain_index_conversion[self.header['captured_gain']]
        psd *= psd_scaling**2
        return psd, frequency


class MetaData(OsciData):
    def __init__(self, filename_):
        super().__init__(filename_, convert_to_volt=False, read_only_header_=True)

        self.data = {'timestamp' : self.timestamp,
                     'channel' : self.channel,
                     'comment' : self.comment,
                     'pre trigger samples' : self.trigger_data['pre_trigger_samples'],
                     'post trigger samples' : self.trigger_data['post_trigger_samples'],
                     'trigger source' : self.trigger_data['trigger_source'],
                     'sample rate': self.sample_data['sample_rate [Hz]'] / 1000,
                     'coupling' : self.sample_data['coupling'],
                     'capture gain' : self.captured_gain_index_conversion[self.header['captured_gain']]}

def export_psds(filenames, export_filename, method='rfft', subdivision_factor=1, pad_zeros_pow2=False):
    psds = []
    values = []

    for fn in filenames:
        dat = OsciData(fn, convert_to_volt=False)

        psd, frequency = derive_psd(data=dat.measured_data_raw, method=method,
                   sampling_rate=1 / (dat.runtime[1] - dat.runtime[0]),
                   pad_zeros_pow2=pad_zeros_pow2, subdivision_factor=subdivision_factor)

        psds.append(psd)
        values.append(dat.measured_data_raw)
        runtime = dat.runtime

    with h5py.File(export_filename, 'w') as f:
        f.create_dataset("psds", data=np.array(psds).T)
        f.create_dataset("frequencies", data=frequency)
        f.create_dataset("timetraces_value", data=np.array(values).T)
        f.create_dataset("timetraces_runtime", data=runtime)

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # import matplotlib

    # matplotlib.rcParams['agg.path.chunksize'] = 10000

    filename = '/media/cavitydata/Cavity Lab/data/2016/2016-11/2016-11-24_aom-test/GaGe-measurements/gage_test/test0.sig'
    print(MetaData(filename).data)
    # data = OsciData(filename, convert_to_volt=True)
    #
    # print('sample_data :', data.sample_data)
    # print('channel :', data.channel)
    # print('comment :', data.comment)
    # print('trigger_data :', data.trigger_data)
    # print('impedance :', data.impedance)
    # print('file_version :', data.header['file_version'])
    # print('sample_offset_32 :', data.header['sample_offset_32'])
    # print('sample_resolution_32 :', data.header['sample_resolution_32'])
    # print('sample_offset :', data.header['sample_offset'])
    # print('sample_resolution :', data.header['sample_resolution'])
    #
    # plt.figure()
    # frequency, psd = data.power_spectral_density(subdivision_factor_=50)
    # plt.semilogy(frequency, psd)
    # plt.grid()
    # plt.show()
