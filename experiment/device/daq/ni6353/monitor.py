# from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5 import uic, QtCore, QtWidgets, QtGui
import os
import sys
import numpy as np
import pyqtgraph as pg
from PyDAQmx import *
from PyDAQmx.DAQmxCallBack import *
from numpy import zeros, array
from ctypes import byref
import threading
from time import sleep, time
import pandas as pd
import datetime as dt

class CallbackTaskSynchronous(Task):
    configuration = {'dev_name': 'Dev1',
                     'data_len': 2000,
                     'sample_rate': 100000.0,
                     'voltage_range': [-10., 10.],
                     'channels': ['ai0', 'ai1'],
                     'channel_labels': None,
                     'timeout': 10.,
                     'trigger_slope': None,
                     'trigger_level': 0.,
                     'ttl_trigger': False}

    def __init__(self, **kwargs):
        Task.__init__(self)

        for kwarg in kwargs:
            if kwarg in self.configuration:
                self.configuration[kwarg] = kwargs[kwarg]
        if self.configuration['channel_labels'] is None or len(self.configuration['channel_labels']) != len(self.configuration['channels']):
            self.configuration['channel_labels'] = self.configuration['channels']
        self.runtime = np.linspace(0, self.configuration['data_len'] / self.configuration['sample_rate'], self.configuration['data_len'])
        self._data = zeros(self.configuration['data_len'] * len(self.configuration['channels']))
        self.read = int32()
        for channel in self.configuration['channels']:
            self.CreateAIVoltageChan(self.configuration['dev_name'] + '/' + channel, "", DAQmx_Val_RSE,
                                     self.configuration['voltage_range'][0], self.configuration['voltage_range'][1],
                                     DAQmx_Val_Volts, None)

        self.CfgSampClkTiming("", self.configuration['sample_rate'], DAQmx_Val_Rising, DAQmx_Val_ContSamps,
                              self.configuration['data_len'])
        if self.configuration['trigger_slope'] == 'rising':
            self.CfgAnlgEdgeStartTrig(self.configuration['dev_name'] + '/' + self.configuration['channels'][0], DAQmx_Val_RisingSlope, self.configuration['trigger_level'])
        elif self.configuration['trigger_slope'] == 'falling':
            self.CfgAnlgEdgeStartTrig(self.configuration['dev_name'] + '/' + self.configuration['channels'][0], DAQmx_Val_FallingSlope, self.configuration['trigger_level'])


        self.AutoRegisterEveryNSamplesEvent(DAQmx_Val_Acquired_Into_Buffer, self.configuration['data_len'], 0)
        self.AutoRegisterDoneEvent(0)
        self._data_lock = threading.Lock()
        self._newdata_event = threading.Event()

    def EveryNCallback(self):
        with self._data_lock:
            self.ReadAnalogF64(self.configuration['data_len'], self.configuration['timeout'], DAQmx_Val_GroupByChannel,
                               self._data, self.configuration['data_len'] * len(self.configuration['channels']),
                               byref(self.read), None)
            self._newdata_event.set()
        return 0

    def DoneCallback(self, status):
        print("Status",status.value)
        return 0

    def get_data(self, blocking=True):
        if blocking:
            if not self._newdata_event.wait(self.configuration['timeout']):
                raise ValueError("timeout waiting for data from device")
        with self._data_lock:
            self._newdata_event.clear()
            return self._data.copy().reshape((len(self.configuration['channels']), self.configuration['data_len']))

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

class MonitorWidget(QtWidgets.QMainWindow):
    dockWidgets = {}
    plotWidgets = {}
    buttonTexts = ['Start Acquisition', 'Stop Acquisition']
    data = None
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.task = CallbackTaskSynchronous(**self.kwargs)
        QtWidgets.QDialog.__init__(self)

        # Set up the user interface from Designer.
        self.ui = uic.loadUi(os.path.join('gui', 'mainwindow.ui'))
        self.ui.show()
        # self.ui.centralWidget.hide()

        for channel in self.task.configuration['channel_labels']:
            self.dockWidgets[channel] = QtWidgets.QDockWidget()
            self.plotWidgets[channel] = pg.PlotWidget(title=channel)
            self.dockWidgets[channel].setWidget(self.plotWidgets[channel])
            self.ui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dockWidgets[channel])
            self.plotWidgets[channel].setLabel('bottom', text='runtime', units='s')
            self.plotWidgets[channel].setLabel('left', text='voltage', units='V')
        self.task.ClearTask()

        self.ui.toggleDAQ.clicked.connect(self.toggleDAQ)
        self.ui.saveData.clicked.connect(self.saveData)
        self.backgroundDAQ = BackgroundThread(self.get_data)
        self.backgroundDAQ.updated_data.connect(lambda data: self.plot_data(data))
        self.backgroundDAQ.mutex.lock()
        self.backgroundDAQ.acquiring = True
        self.backgroundDAQ.mutex.unlock()
        self.backgroundDAQ.start()

    def __del__(self):
        self.task.StopTask()
        self.task.ClearTask()

    def toggleDAQ(self):
        button_txt = self.ui.toggleDAQ.text()
        if button_txt == self.buttonTexts[0]:
            self.ui.toggleDAQ.setText(self.buttonTexts[1])
            self.backgroundDAQ.mutex.lock()
            self.backgroundDAQ.acquiring = True
            self.backgroundDAQ.mutex.unlock()
        else:
            self.ui.toggleDAQ.setText(self.buttonTexts[0])
            self.backgroundDAQ.mutex.lock()
            self.backgroundDAQ.acquiring = False
            self.backgroundDAQ.mutex.unlock()

    def saveData(self):
        out_panda = pd.DataFrame(data=self.data.T, columns=self.task.configuration['channel_labels'])
        timestamp = dt.datetime.fromtimestamp(time()).strftime('%Y-%m-%d_%H-%M-%S')
        out_panda.to_hdf(timestamp + '.h5', 'measurement')

        with pd.HDFStore(timestamp + '.h5') as store:
            store.put('measurement', out_panda)
            for key in self.task.configuration:
                store.get_storer('measurement').attrs.__setattr__(key, self.task.configuration[key])

    def get_data(self):
        self.task = CallbackTaskSynchronous(**self.kwargs)
        self.task.StartTask()
        data = self.task.get_data()
        self.task.StopTask()
        self.task.ClearTask()
        return data

    def manipulate_data(self):
        return
    def manipulate_plot(self):
        return

    def plot_data(self, data):
        if data is not None:
            self.data = data
            self.manipulate_data()
            for channel, d in zip(self.task.configuration['channel_labels'], self.data):
                self.plotWidgets[channel].plot(self.task.runtime, d, clear=True)
                if self.plotWidgets[channel].getPlotItem().ctrl.fftCheck.isChecked():
                    self.plotWidgets[channel].setLabel('bottom', text='frequency', units='Hz')
                    self.plotWidgets[channel].setLabel('left', text='power spectrum', units='V<sup>2</sup>')

                else:
                    self.plotWidgets[channel].setLabel('bottom', text='runtime', units='s')
                    self.plotWidgets[channel].setLabel('left', text='voltage', units='V')
            self.manipulate_plot()

        self.backgroundDAQ.start()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MonitorWidget(data_len=4600,
                           sample_rate=200e3,
                           channels=['ai0', 'ai1', 'ai2', 'ai3', 'ai8'],
                           channel_labels=['trigger', 'cavity sweep', 'cavity transmission', 'cavity reflection', 'PDH signal'],
                           voltage_range=[-10, 10],
                           trigger_slope=None,
                           trigger_level=-1.0)
    sys.exit(app.exec_())
