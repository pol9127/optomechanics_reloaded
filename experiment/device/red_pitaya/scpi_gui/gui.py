from PyQt5 import uic, QtCore, QtWidgets, QtGui
import os
import sys
import numpy as np
import pyqtgraph as pg
from time import sleep, time
import pandas as pd
import datetime as dt
from collections import Iterable

from experiment.device.red_pitaya.scpi_gui import redpitaya_scpi as scpi

class BackgroundThread(QtCore.QThread):
    updated_data = QtCore.pyqtSignal(object)
    mutex = QtCore.QMutex()
    _stop = False
    acquiring = True

    def __init__(self, process):
        self.process = process
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        try:
            self.mutex.lock()
            if self.acquiring:
                self.mutex.unlock()
                data = self.process()
                self.updated_data.emit(data)
            else:
                sleep(1)
                self.mutex.unlock()
                data = None
                self.updated_data.emit(data)
        except:
            data = None
            self.updated_data.emit(data)

    def stop(self):
        self.mutex.lock()
        self._stop = True
        self.mutex.unlock()


class RedPitayaConfig:
    acquisition = {}
    digital_IO = {}
    analog_IO = {}
    signal_generation = {}
    rp = None
    acq_started = False
    max_sample_freq = 125. # MHz
    sample_depth = 16384

    def __init__(self, rp):
        self.rp = rp

    def set_standard_config(self):
        self.rp.acq_rst()
        self.rp.acq_gain_set(1, 'HV')
        self.acquisition['Gain_CH1'] = 'HV'
        self.rp.acq_gain_set(2, 'HV')
        self.acquisition['Gain_CH2'] = 'HV'
        self.rp.acq_dec = 1024
        self.acquisition['Decimation'] = self.rp.acq_dec
        self.rp.acq_avg = True
        self.acquisition['Averaging'] = self.rp.acq_avg
        self.rp.acq_trig = {'Level': 2.5, 'Delay': 8192}
        for key in self.rp.acq_trig:
            self.acquisition[key] = self.rp.acq_trig[key]

    def set_acquisition_config(self, **kwargs):
        for kwarg, val in kwargs.items():
            if kwarg in self.acquisition:
                if kwarg == 'Gain_CH1':
                    retval = self.rp.acq_gain_set(1, val)
                    if retval == 1:
                        self.acquisition['Gain_CH1'] = val
                if kwarg == 'Gain_CH2':
                    retval = self.rp.acq_gain_set(2, val)
                    if retval == 2:
                        self.acquisition['Gain_CH2'] = val
                if kwarg == 'Decimation':
                    self.rp.acq_dec = val
                    self.acquisition['Decimation'] = self.rp.acq_dec
                if kwarg == 'Averaging':
                    self.rp.acq_avg = val
                    self.acquisition['Averaging'] = self.rp.acq_avg
                if kwarg == 'Level':
                    self.rp.acq_trig = {'Level': 2.5}
                    self.acquisition['Level'] = self.rp.acq_trig['Level']
                if kwarg == 'Delay':
                    self.rp.acq_trig = {'Delay': 2.5}
                    self.acquisition['Delay'] = self.rp.acq_trig['Delay']
                if kwarg == 'Delay in ns':
                    self.rp.acq_trig = {'Delay in ns': 2.5}
                    self.acquisition['Delay in ns'] = self.rp.acq_trig['Delay in ns']
    @property
    def runtime(self):
        sample_freq = self.max_sample_freq / self.acquisition['Decimation']
        return np.arange(self.sample_depth, dtype=float) / (sample_freq * 1e6)

class MainWidget(QtWidgets.QMainWindow):
    connection = {'hosts':[],
                  'ports':[],
                  'timeouts':[]}
    red_pitayas = {}
    red_pitaya_configurations = {}
    dockWidgets = {}
    plotWidgets = {}
    channels = []
    channel_names = []
    daq_threads = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'hosts' in self.kwargs:
            if isinstance(self.kwargs['hosts'], Iterable) and not isinstance(self.kwargs['hosts'], str):
                self.connection['hosts'] = self.kwargs['hosts']
            elif isinstance(self.kwargs['hosts'], str):
                self.connection['hosts'] = [self.kwargs['hosts']]
            else:
                sys.exit('hosts must be a list of hostnames or a string of one hostname not: {0}'.format(self.kwargs['hosts']))

        if 'ports' in self.kwargs:
            if isinstance(self.kwargs['ports'], Iterable):
                if len(self.kwargs['hosts']) == len(self.kwargs['ports']):
                    self.connection['ports'] = self.kwargs['ports']
                else:
                    sys.exit('If a list of ports is specified the length must match the number of hosts')
            elif isinstance(self.kwargs['ports'], int):
                self.connection['ports'] = [self.kwargs['ports']]
            else:
                sys.exit('ports must be a list of ports or an integer of one port not: {0}'.format(self.kwargs['ports']))
        else:
            self.connection['ports'] = [5000] * len(self.connection['hosts'])

        if 'timeouts' in self.kwargs:
            if isinstance(self.kwargs['timeouts'], Iterable):
                if len(self.kwargs['hosts']) == len(self.kwargs['timeouts']):
                    self.connection['timeouts'] = self.kwargs['timeouts']
                else:
                    sys.exit('If a list of timeouts is specified the length must match the number of hosts')
            elif isinstance(self.kwargs['timeouts'], int) or isinstance(self.kwargs['timeouts'], float):
                self.connection['timeouts'] = [self.kwargs['timeouts']]
            else:
                sys.exit('ports must be a list of timeouts or an integer/float of one timeout not: {0}'.format(self.kwargs['timeouts']))
        else:
            self.connection['timeouts'] = [None] * len(self.connection['hosts'])

        for host in self.connection['hosts']:
            self.channels.append('{0}:CH1'.format(host))
            self.channels.append('{0}:CH2'.format(host))

        if 'channel_names' in self.kwargs:
            if len(self.kwargs['channel_names']) == len(self.channels):
                self.channel_names = self.kwargs['channel_names']
            else:
                self.channel_names = self.channels.copy()
        else:
            self.channel_names = self.channels.copy()

        self.connect_red_pitayas()
        self.initialize_red_pitayas()
        QtWidgets.QDialog.__init__(self)

        # Set up the user interface from Designer.
        self.ui = uic.loadUi(os.path.join('ui_files', 'mainWindow.ui'))
        self.ui.show()
        self.create_plot_widgets()
        self.init_daq_threads()


    def connect_red_pitayas(self):
        for host, port, timeout in zip(self.connection['hosts'], self.connection['ports'], self.connection['timeouts']):
            rp_tmp = scpi.scpi(host=host, port=port, timeout=timeout)
            self.red_pitayas[host] = rp_tmp

    def initialize_red_pitayas(self):
        for host in self.red_pitayas:
            cfg_tmp = RedPitayaConfig(self.red_pitayas[host])
            cfg_tmp.set_standard_config()
            self.red_pitaya_configurations[host] = cfg_tmp

    def init_daq_threads(self):
        for host, rp in self.red_pitayas.items():
            self.daq_threads[host] = BackgroundThread(lambda host_=host: self.get_data(host_))
            self.daq_threads[host].updated_data.connect(lambda data, host_=host: self.plot_data(data, host_))
            self.daq_threads[host].mutex.lock()
            self.daq_threads[host].acquiring = True
            self.daq_threads[host].mutex.unlock()
            self.daq_threads[host].start()


    def create_plot_widgets(self):
        for channel, name in zip(self.channels, self.channel_names):
            self.dockWidgets[channel] = QtWidgets.QDockWidget()
            self.plotWidgets[channel] = pg.PlotWidget(title=name)
            self.dockWidgets[channel].setWidget(self.plotWidgets[channel])
            self.ui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dockWidgets[channel])
            self.plotWidgets[channel].setLabel('bottom', text='runtime', units='s')
            self.plotWidgets[channel].setLabel('left', text='voltage', units='V')

    def manipulate_data(self):
        return
    def manipulate_plot(self):
        return

    def plot_data(self, data, host):
        if data is not None:
            self.data = data
            self.manipulate_data()
            channels = [ch for ch in self.channels if host in ch]
            for channel, d in zip(channels, self.data):
                self.plotWidgets[channel].plot(self.red_pitaya_configurations[host].runtime, d, clear=True)
                if self.plotWidgets[channel].getPlotItem().ctrl.fftCheck.isChecked():
                    self.plotWidgets[channel].setLabel('bottom', text='frequency', units='Hz')
                    self.plotWidgets[channel].setLabel('left', text='power spectrum', units='V<sup>2</sup>')
                else:
                    self.plotWidgets[channel].setLabel('bottom', text='runtime', units='s')
                    self.plotWidgets[channel].setLabel('left', text='voltage', units='V')
            self.manipulate_plot()

        self.daq_threads[host].start()

    def get_data(self, host):
        rp = self.red_pitayas[host]
        if not self.red_pitaya_configurations[host].acq_started:
            rp.acq_start()
            rp.acq_trig_src_set('EXT_NE')
            self.red_pitaya_configurations[host].acq_started = True

        while 1:
            answer = rp.txrx_txt('ACQ:TRIG:STAT?')
            if answer == 'TD':
                break

        buff_string1 = rp.txrx_txt('ACQ:SOUR1:DATA?')
        buff_string2 = rp.txrx_txt('ACQ:SOUR2:DATA?')

        if self.red_pitaya_configurations[host].acq_started:
            rp.acq_start()
            rp.acq_trig_src_set('EXT_NE')

        buff_string1 = buff_string1.strip('{}\n\r').replace("  ", "").split(',')
        buff_string2 = buff_string2.strip('{}\n\r').replace("  ", "").split(',')
        buff1 = list(map(float, buff_string1))
        buff2 = list(map(float, buff_string2))

        return [buff1, buff2]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWidget(hosts=['red-pitaya-06.ee.ethz.ch'])
    sys.exit(app.exec_())
