import sys
import time
import sched
from optomechanics.experiment.cryolev.experiment_monitor.monitor import h5_saver
from optomechanics.experiment.cryolev.experiment_monitor.drivers.zhinst_monitor import zhinst_monitor
from optomechanics.experiment.cryolev.experiment_monitor.drivers.attoDRY800 import client as attoDry800Client
from optomechanics.experiment.cryolev.experiment_monitor.drivers.Thyra import measure
from optomechanics.experiment.device.daq.daq_hat.mcc_118 import daq_client
import datetime
import os
import numpy as np
import queue


class ExperimentMonitor(object):
    rpi_channel_header = [  # None means nothing is connected -> those are not stored.
        None,  # Ch 0
        None,  # Ch 1
        None,  # Ch 2
        None,  # Ch 3
        None,  # Ch 4
        None,  # Ch 5
        'LO power (V)',  # Ch 6
        'Trap power (V)']  # Ch 7
    rpi_filename = 'rpi_monitor.h5'

    # The python server which we talk to here, returns a dictionary with two entries: 'pressure' and '4KstageTemp'
    attoheaders = {'pressure': 'Pressure (mbar)',
                   '4KstageTemp': '4Kstage Temp (K)',
                   'SampleTemp': 'Sample Temp (K)'}
    atto_filename = 'attoDRY800_monitor_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'.h5'

    zimon_headers = [
        'Z-PLL Demod R', 'Z-Freq (Hz)', 'X-PLL Demod R', 'X-Freq (Hz)', 'Y-PLL Demod R', 'Y-Freq (Hz)']
    zimon_filename = 'zi_monitor_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'.h5'

    zispectra_filename = 'zispectra_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'.h5'

    thyra_filename = 'thyra_measure_'+datetime.datetime.now().strftime("%y%m%d_%H%M%S")+'.h5'
    thyra_header = 'Thyra Pressure (mBar)'

    def _assert_no_override(self, folder, filename):
        assert not os.path.exists(os.path.join(folder, filename)), 'The file ' + filename + \
            ' already exists in ' + folder + '. I refuse to override existing data.'

    def __init__(self, folder, show_current_value=False, T_RPi=1, T_atto=1, T_zimon=5, T_zispectra=30, T_store=60, T_thyra=1,
                 verbose=False, que=None):
        assert os.path.isdir(folder), 'This folder: ' + \
            folder + ' does not exist. Please create it.'
        self.queue = que #The queue is used to transfer the measurement data to the user interface
        self.T_RPi = T_RPi
        self.T_atto = T_atto
        self.T_store = T_store
        self.T_zimon = T_zimon
        self.T_zispectra = T_zispectra
        self.T_thyra = T_thyra
        self.sc = None


        if T_RPi > 0:
            self._assert_no_override(folder, self.rpi_filename)
            self.rpi_daq_cl = daq_client.DAQClient(
                HOST='129.132.1.142', PORT=65432)
            self.rpi_daq_cl.set_acquisition_settings(
                channels=[0, 1, 2, 3, 4, 5, 6, 7], requested_scan_rate=10e3, samples_per_channel=100)
            self.unsaved_rpi_data = []
            rpi_columns = ['timestamp'] + \
                [x for x in self.rpi_channel_header if x is not None]
            self.rpi_saver = h5_saver.h5_saver(
                rpi_columns, os.path.join(folder, self.rpi_filename))

        if T_thyra > 0:
            #self._assert_no_override(folder, self.thyra_filename)
            self.unsaved_thyra_data = []
            thyra_colunms = ['timestamp', self.thyra_header]
            self.thyra_saver = h5_saver.h5_saver(
                thyra_colunms, os.path.join(folder, self.thyra_filename))

        self.attoDRYcl = None
        if T_atto > 0:
            #self._assert_no_override(folder, self.atto_filename)
            self.attoDRYcl = attoDry800Client.attoDRYMonitorClient(
                HOST='129.132.1.185', PORT=65433)
            self.unsaved_atto_data = []
            atto_columns = ['timestamp'] + [self.attoheaders[x]
                                            for x in self.attoheaders]
            self.atto_saver = h5_saver.h5_saver(
                atto_columns, os.path.join(folder, self.atto_filename))

        self.zimon = None
        data_server_ip = '129.132.1.185'
        if T_zimon > 0:
            self._assert_no_override(folder, self.zimon_filename)
            # 'dev3714' or 'dev4968'
            self.zimon = zhinst_monitor.zhinst_monitor(
                data_server_ip=data_server_ip)
            self.zimon.addDAQmod('Z-PLL', 'dev3714', 100,
                                 '1', ['x', 'y', 'frequency'])
            self.zimon.addDAQmod('X-PLL', 'dev3714', 100,
                                 '2', ['x', 'y', 'frequency'])
            self.zimon.addDAQmod('Y-PLL', 'dev3714', 100,
                                 '3', ['x', 'y', 'frequency'])
            self.unsaved_zimon_data = []
            zimon_headers = ['timestamp'] + self.zimon_headers
            self.zimon_saver = h5_saver.h5_saver(
                zimon_headers, os.path.join(folder, self.zimon_filename))

        if T_zispectra > 0:
            self._assert_no_override(folder, self.zispectra_filename)
            if self.zimon is None:
                self.zimon = zhinst_monitor.zhinst_monitor(
                    data_server_ip=data_server_ip)
            l, df = self.zimon.addScopeChannels(
                'BalDet', 'dev3714', 10, 10, 0)
            self.unsaved_zispectra_data = []
            zispectra_headers = ['timestamp'] + ['0 Hz',
                                                 repr(df) + ' Hz'] + ['' for x in range(l - 2)]
            self.zispectra_saver = h5_saver.h5_saver(
                zispectra_headers, os.path.join(folder, self.zispectra_filename))

        # should only be set true from a terminal -> does not work in notebooks.
        self.show_current_value = show_current_value
        self.current_values = {}

        self.verbose = verbose

    def rpi_daq_get_datapoint(self):
        '''rpi_daq_cl = daq_client.DAQClient(
            HOST='129.132.1.142', PORT=65432)
        rpi_daq_cl.set_acquisition_settings(
            channels=[0, 1, 2, 3, 4, 5, 6, 7], requested_scan_rate=10e3, samples_per_channel=100)'''
        # This takes about 60us
        if self.rpi_daq_cl is None:
            return None
        t1 = time.time()
        try:
            self.rpi_daq_cl.start_measurement()
        except AttributeError:
            print("The data cannot be decoded")

        mf = False
        for i in range(10):
            if self.rpi_daq_cl.measurement_finished:
                mf = True
                break
            time.sleep(1e-3)

        dat = None
        if mf:
            try:
                dat = self.rpi_daq_cl.get_measurement_data()
                dat = np.average(dat, axis=1)
            except AttributeError:
                print("the data cannot be decoded.")
        #del self.rpi_daq_cl
        if dat is None:
            return None
        t2 = time.time()
        delta_t = t2-t1



        dct = {}
        for i in range(len(self.rpi_channel_header)):
            header = self.rpi_channel_header[i]
            if header:
                dct[header] = dat[i]

        return dct
        #del self.rpi_daq_cl

    def thyra_get_datapoint(self):
        dct = {}
        dct[self.thyra_header] = measure.measure('COM9', verbose = False)
        return dct

    def atto_get_datapoint(self):

        dct = {}
        dat = None

        if self.attoDRYcl:
            dat = self.attoDRYcl.ask_datapoint()

        if dat is None:
            return None

        for key in self.attoheaders:
            dct[self.attoheaders[key]] = dat[key]
        return dct

    def zimon_get_datapoint(self):
        dat_z = None
        dat_x = None
        dat_y = None

        if (self.zimon is not None) and (self.T_zimon > 0):
            # return: ( timestamps, [x, y, freq])
            dat_z = self.zimon.getDemodSamples('Z-PLL')
            # return: ( timestamps, [x, y, freq])
            dat_x = self.zimon.getDemodSamples('X-PLL')
            # return: ( timestamps, [x, y, freq])
            dat_y = self.zimon.getDemodSamples('Y-PLL')
        if (dat_z is None) and (dat_x is None) and (dat_y is None):
            return None

        dct = {}
        r_z = np.average(np.sqrt(dat_z[1][0]**2 + dat_z[1][1]**2))
        r_x = np.average(np.sqrt(dat_x[1][0]**2 + dat_x[1][1]**2))
        r_y = np.average(np.sqrt(dat_y[1][0]**2 + dat_y[1][1]**2))

        demod_freq_z = np.average(dat_z[1][2])
        demod_freq_x = np.average(dat_x[1][2])
        demod_freq_y = np.average(dat_y[1][2])

        for i in range(len(self.zimon_headers)):
            dct[self.zimon_headers[i]] = (
                r_z, demod_freq_z, r_x, demod_freq_x, r_y, demod_freq_y)[i]
        return dct

    def zispectra_get_datapoint(self):
        dat = None
        if (self.zimon is not None) and (self.T_zispectra > 0):
            dat = self.zimon.getScopeWave('BalDet')[1][:, 0]
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # CAREFUL: At the moment, I simply assume the return to be a single wave. In reality it could be two!
            # Get rid of the [:,0] and handle correctly.
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return dat

    def update_console_values(self):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # At the moment, we only show the attoDRY and RPi data, work on a better monitor later!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if self.show_current_value:
            tmp = []
            for key in ['4KstageTemp (K)', 'SampleTemp (K)', 'Pressure (mbar)', 'LO power (V)', 'Trap power (V)', 'ThyraPressure (mBar)']:
                if key in self.current_values:
                    if self.current_values[key] is None:
                        tmp.append(key + ": None")
                    else:
                        tmp.append(key + ": %.3e" % self.current_values[key])
            tmp = ', '.join(tmp)
            print(tmp, flush=False, end='\r')
            if not (self.queue is None):
                self.queue.put(self.current_values)#put data in the queue

    def event_read_RPi(self):
        now = time.time()
        if self.verbose:
            print('Read from RPi ', datetime.datetime.fromtimestamp(now))
        dp = self.rpi_daq_get_datapoint()
        if dp is not None:
            for key in dp:
                self.current_values[key] = dp[key]
            self.update_console_values()
            dp['timestamp'] = now
            self.unsaved_rpi_data.append(dp)

        next = self.T_RPi * (int(now / self.T_RPi) + 1)
        self.sc.enterabs(next, 2, self.event_read_RPi)

    def event_read_atto(self):
        now = time.time()
        if self.verbose:
            print('Read from Atto ', datetime.datetime.fromtimestamp(now))
        dp = self.atto_get_datapoint()
        if dp is not None:
            for key in dp:
                self.current_values[key] = dp[key]
            self.update_console_values()
            dp['timestamp'] = now
            self.unsaved_atto_data.append(dp)

        next = self.T_atto * (int(now / self.T_atto) + 1)
        self.sc.enterabs(next, 2, self.event_read_atto)

    def event_read_thyra(self):
        now = time.time()
        if self.verbose:
            print('Read from Thyra ', datetime.datetime.fromtimestamp(now))
        dp = self.thyra_get_datapoint()
        if dp is not None:
            for key in dp:
                self.current_values[key] = dp[key]
            self.update_console_values()
            dp['timestamp'] = now
            self.unsaved_thyra_data.append(dp)
        next = self.T_thyra * (int(now / self.T_thyra) + 1)
        self.sc.enterabs(next, 2, self.event_read_thyra)

    def event_read_ZImon(self):
        now = time.time()
        if self.verbose:
            print('Read from ZI monitor', datetime.datetime.fromtimestamp(now))
        dp = self.zimon_get_datapoint()
        if dp is not None:
            for key in dp:
                self.current_values[key] = dp[key]
            self.update_console_values()
            dp['timestamp'] = now
            self.unsaved_zimon_data.append(dp)
        next = self.T_zimon * (int(now / self.T_zimon) + 1)
        self.sc.enterabs(next, 2, self.event_read_ZImon)

    def event_read_ZIspectra(self):
        now = time.time()
        if self.verbose:
            print('Read Spectrum from MFLI',
                  datetime.datetime.fromtimestamp(now))
        dp = self.zispectra_get_datapoint()
        if dp is not None:
            self.unsaved_zispectra_data.append(np.concatenate(([now], dp)))
        next = self.T_zispectra * (int(now / self.T_zispectra) + 1)
        self.sc.enterabs(next, 2, self.event_read_ZIspectra)

    def save(self, saver, unsaved_data):
        if len(unsaved_data) > 0:
            dp_as_matrix = np.array(
                [[unsaved_data[row][col] for col in saver.columns] for row in range(len(unsaved_data))])
            saver.save_datapoints(dp_as_matrix)

    def event_store_new_data(self):
        now = time.time()
        print('Saving data at', datetime.datetime.fromtimestamp(now))

        if self.T_RPi > 0:
            self.save(self.rpi_saver, self.unsaved_rpi_data)
            self.unsaved_rpi_data = []
        if self.T_thyra > 0:
            self.save(self.thyra_saver, self.unsaved_thyra_data)
            self.unsaved_thyra_data = []
        if self.T_atto > 0:
            self.save(self.atto_saver, self.unsaved_atto_data)
            self.unsaved_atto_data = []
        if self.T_zimon > 0:
            self.save(self.zimon_saver, self.unsaved_zimon_data)
            self.unsaved_zimon_data = []
        if self.T_zispectra > 0:
            if len(self.unsaved_zispectra_data) > 0:
                dp_as_matrix = np.array(self.unsaved_zispectra_data)
                self.zispectra_saver.save_datapoints(dp_as_matrix)
                self.unsaved_zispectra_data = []

        next = self.T_store * (int(now / self.T_store) + 1)
        self.sc.enterabs(next, 1, self.event_store_new_data)

    def start_monitoring(self):
        self.sc = sched.scheduler(time.time, time.sleep)
        now = time.time()

        if self.T_RPi > 0:
            # start RPi monitor scheduler
            next = self.T_RPi * (int(now / self.T_RPi) + 1)
            self.sc.enterabs(next, 2, self.event_read_RPi)

        if self.T_thyra > 0:
            next = self.T_thyra * (int(now / self.T_thyra) + 1)
            self.sc.enterabs(next, 2, self.event_read_thyra)

        if self.T_atto > 0:
            # start atto scheduler
            next = self.T_atto * (int(now / self.T_atto) + 1)
            self.sc.enterabs(next, 2, self.event_read_atto)

        if self.T_zimon > 0:
            # start ZI monitor scheduler
            next = self.T_zimon * (int(now / self.T_zimon) + 1)
            self.sc.enterabs(next, 2, self.event_read_ZImon)

        if self.T_zispectra > 0:
            # start ZI spectra scheduler
            next = self.T_zispectra * (int(now / self.T_zispectra) + 1)
            self.sc.enterabs(next, 2, self.event_read_ZIspectra)

        if self.T_thyra > 0:
            next = self.T_thyra * (int(now / self.T_thyra) + 1)
            self.sc.enterabs(next, 2, self.event_read_thyra)

        # start store scheduler
        next = self.T_store * (int(now / self.T_store) + 1)
        self.sc.enterabs(next, 1, self.event_store_new_data)

        self.sc.run()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("You need to tell me a file location")
    else:
        T_RPi = -1
        T_atto = -1
        T_zimon = -1
        T_zispectra = -1
        T_thyra = -1
        print('location:', sys.argv[1])
        for arg in sys.argv[2:]:
            if arg == '-rpi':
                T_RPi = 1
                print('Connect Raspberry Pi')
            elif arg == '-atto':
                T_atto = 1
                print('Connect attoDRY')
            elif arg == '-zi':
                T_zimon = 5
                print('Connect ZH instruments MFLI')
            elif arg == '-zispectra':
                T_zispectra = 30
                print('Also take spectra from ZH instruments MFLI')
            elif arg == '-thyra':
                T_thyra = 1
                print('Connect Thyracont pressure gaudge')

            else:
                print('I don\'t understand the command', arg)
                assert False, ('I don\'t understand the command %s' % arg)

        exp_mon = ExperimentMonitor(sys.argv[1], show_current_value=True,
                                    T_RPi=T_RPi, T_atto=T_atto, T_zimon=T_zimon, T_zispectra=T_zispectra, T_thyra=T_thyra)
        exp_mon.start_monitoring()
