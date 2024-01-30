from __future__ import division
import os
from PyQt5 import QtWidgets, QtCore, Qt, uic, QtGui
from optomechanics.experiment.lab_control.device import Device
from optomechanics.post_processing.spectrum import derive_psd
import pyqtgraph as pg
import numpy as np
from time import sleep
from copy import copy
from timeit import default_timer
from memory_profiler import profile
import optomechanics.experiment.lab_control.hdf5 as h5
from datetime import datetime as dt
import ast

def fft_func(data, sampling_rate):
    return derive_psd(data, sampling_rate, method='rfft')

class Scope(Device, object):
    hdf5_file = None
    operation = None
    isDevice = False
    name = None
    kind = None
    time = None
    connection_settings = None
    channel_plots = {}
    fft_plots = {}
    channel_tables = {}
    trigger_tables = {}
    set_range = True
    # counter = 0

    @property
    def information(self):
        info_dict = {'name' : self.name, 'kind' : self.kind}
        return info_dict

    @staticmethod
    def find_devices():
        devices = []
        return devices

    def export(self, kind='individual'):
        if self.operation is not None:
            channels = ast.literal_eval(self.configuration['Acquisition']['Channels'])
            channels = ['Channel ' + str(ch).zfill(2) for ch in channels]
            self.operation.data = self.data
            self.operation.attributes['timestamp'] = str(dt.now())
            self.operation.attributes['label'] = channels
            self.operation.attributes['unit'] = ['mV'] * len(channels)
            if kind == 'individual' and self.hdf5_file is not None:
                self.hdf5_file.open()
                self.hdf5_file.modified = dt.now()
                self.hdf5_file.add_operation(self.operation)
                self.hdf5_file.close()
            else:
                self.operation.name = self.operation.name + '_' + self.name
                return self.operation

    class _ControlWidget(QtWidgets.QMainWindow):
        data = None
        graphicsLayoutWidget_ffts = None
        def __init__(self, outer, parent=None):
            self.outer = outer
            self.outer.inner = self
            self.background_data_acquisition = self.outer.BackgroundThread(self.outer.timetrace)

            super(self.outer._ControlWidget, self).__init__(parent)

            uic.loadUi(os.path.join('ui_files', 'control_scope.ui'), self)

            self.channel_plot_widgets()
            self.populate_configuration()

            self.pushButton_single_shot.clicked.connect(self.single_shot)
            self.pushButton_acquire.clicked.connect(self.acquire)
            self.pushButton_stop.clicked.connect(self.stop_acquire)
            self.pushButton_export_settings.clicked.connect(self.export_settings)
            self.pushButton_export.clicked.connect(self.outer.export)
            self.pushButton_save_config.clicked.connect(self.save_config)
            self.pushButton_load_config.clicked.connect(self.load_config)

            self.checkBox_display_data.setChecked(True)
            self.checkBox_fft.toggled.connect(lambda state_ : self.fft(state_))

            self.splitter.setSizes([1, 1, 0])

        def export_settings(self):
            export_dialog = h5.Frontend(self, self.outer)
            export_dialog.show()

        def channel_plot_widgets(self):
            configuration = self.outer.configuration
            if configuration is not None and 'Acquisition' in configuration:
                if 'Channels' in configuration['Acquisition']:
                    channels = ast.literal_eval(configuration['Acquisition']['Channels'])
                    for ch, i in zip(channels, range(len(channels))):
                        self.outer.channel_plots[i] = self.graphicsLayoutWidget_channels.addPlot(title='Channel ' + str(ch).zfill(2), row=i, col=0)
                        self.outer.channel_plots[i].setLabel('bottom', text='Runtime', units='s')
                        self.outer.channel_plots[i].disableAutoRange()
                        # self.outer.channel_plots[i].setDownsampling(auto=True, mode='peak', ds=1)

        def populate_configuration(self):
            configuration = self.outer.configuration
            configuration_caps = self.outer.configuration_caps
            if configuration is not None:
                if 'Acquisition' in configuration:
                    cfg_acq = configuration['Acquisition']
                    n_configuration = len(cfg_acq)

                    self.tableWidget_settings.setColumnCount(1)
                    self.tableWidget_settings.setRowCount(n_configuration)
                    settings = list(cfg_acq.keys())
                    for row, setting in zip(range(n_configuration), settings):
                        self.tableWidget_settings.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
                        if setting not in configuration_caps['Acquisition']:
                            self.tableWidget_settings.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg_acq[setting])))
                        else:
                            combo = QtWidgets.QComboBox()
                            for choice in configuration_caps['Acquisition'][setting]:
                                combo.addItem(str(choice))
                            self.tableWidget_settings.setCellWidget(row, 0, combo)
                            combo.setCurrentText(str(cfg_acq[setting]))
                            combo.currentTextChanged.connect(lambda text_, setting_=setting: self.commit_configuration(
                                text=text_, setting=setting_, config='Acquisition'))


                    self.tableWidget_settings.cellChanged.connect(lambda row, col, settings_=settings: self.commit_configuration(
                        table=self.tableWidget_settings, row=row, setting=settings_[row], config='Acquisition'))

                if 'Channel' in configuration and 'Channels' in configuration['Acquisition']:
                    channels = ast.literal_eval(configuration['Acquisition']['Channels'])
                    cfg_ch = configuration['Channel']
                    n_channels = len(channels)
                    for i, cfg, ch in zip(range(n_channels), cfg_ch, channels):
                        tab = QtWidgets.QWidget()
                        layout = QtWidgets.QGridLayout()
                        tab.setLayout(layout)
                        table = QtWidgets.QTableWidget()
                        layout.addWidget(table)
                        self.tabWidget_channel_settings.addTab(tab, 'Channel ' + str(ch).zfill(2))
                        self.outer.channel_tables[i] = table

                        n_configuration = len(cfg)
                        table.setColumnCount(1)
                        table.setRowCount(n_configuration)
                        table.horizontalHeader().setVisible(False)
                        table.horizontalHeader().setStretchLastSection(True)
                        settings = list(cfg.keys())
                        for row, setting in zip(range(n_configuration), settings):
                            table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
                            if setting not in configuration_caps['Channel'][i]:
                                table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg[setting])))
                            else:
                                combo = QtWidgets.QComboBox()
                                for choice in configuration_caps['Channel'][i][setting]:
                                    combo.addItem(str(choice))
                                table.setCellWidget(row, 0, combo)
                                combo.setCurrentText(str(cfg[setting]))
                                combo.currentTextChanged.connect(lambda text_, setting_=setting, i_=i: self.commit_configuration(
                                    text=text_, setting=setting_, config='Channel', idx=i_))

                        table.cellChanged.connect(lambda row, col, table_=table, i_=i, settings_=settings: self.commit_configuration(
                            table=table_, row=row, setting=settings_[row], config='Channel', idx=i_))

                if 'Trigger' in configuration:
                    cfg_tr = configuration['Trigger']
                    n_triggers = len(cfg_tr)
                    for i, cfg in zip(range(n_triggers), cfg_tr):
                        tab = QtWidgets.QWidget()
                        layout = QtWidgets.QGridLayout()
                        tab.setLayout(layout)
                        table = QtWidgets.QTableWidget()
                        layout.addWidget(table)
                        self.tabWidget_trigger_settings.addTab(tab, 'Trigger ' + str(i + 1).zfill(2))
                        self.outer.trigger_tables[i] = table

                        n_configuration = len(cfg)
                        table.setColumnCount(1)
                        table.setRowCount(n_configuration)
                        table.horizontalHeader().setVisible(False)
                        table.horizontalHeader().setStretchLastSection(True)
                        settings = list(cfg.keys())
                        for row, setting in zip(range(n_configuration), settings):
                            table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
                            if setting not in configuration_caps['Trigger'][i]:
                                table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg[setting])))
                            else:
                                combo = QtWidgets.QComboBox()
                                for choice in configuration_caps['Trigger'][i][setting]:
                                    combo.addItem(str(choice))
                                table.setCellWidget(row, 0, combo)
                                combo.setCurrentText(str(cfg[setting]))
                                combo.currentTextChanged.connect(lambda text_, setting_=setting, i_=i: self.commit_configuration(
                                    text=text_, setting=setting_, config='Trigger', idx=i_))

                        table.cellChanged.connect(lambda row, col, table_=table, i_=i, settings_=settings: self.commit_configuration(
                            table=table_, row=row, setting=settings_[row], config='Trigger', idx=i_))


        def update_configuration(self, additional_update=None):
            if additional_update is not None:
                additional_update()
            configuration = self.outer.configuration
            if configuration is not None:
                if 'Acquisition' in configuration:
                    cfg_acq = configuration['Acquisition']
                    n_configuration = len(cfg_acq)
                    self.tableWidget_settings.cellChanged.disconnect()
                    settings = list(cfg_acq.keys())
                    for row, setting in zip(range(n_configuration), settings):
                        cell_widget = self.tableWidget_settings.cellWidget(row, 0)
                        if isinstance(cell_widget, QtWidgets.QComboBox):
                            cell_widget.disconnect()
                            cell_widget.setCurrentText(str(cfg_acq[setting]))
                            cell_widget.currentTextChanged.connect(lambda text_, setting_=setting: self.commit_configuration(
                                text=text_, setting=setting_, config='Acquisition'))
                        else:
                            self.tableWidget_settings.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg_acq[setting])))


                    self.tableWidget_settings.cellChanged.connect(lambda row, col, settings_=settings: self.commit_configuration(
                        table=self.tableWidget_settings, row=row, setting=settings_[row], config='Acquisition'))


                if 'Channel' in configuration and 'Channels' in configuration['Acquisition']:
                    cfg_ch = configuration['Channel']
                    channels = ast.literal_eval(configuration['Acquisition']['Channels'])
                    n_channels = len(channels)
                    for i, cfg, ch in zip(range(n_channels), cfg_ch, channels):
                        table = self.outer.channel_tables[i]
                        table.cellChanged.disconnect()
                        n_configuration = len(cfg)
                        settings = list(cfg.keys())
                        for row, setting in zip(range(n_configuration), settings):
                            cell_widget = table.cellWidget(row, 0)
                            if isinstance(cell_widget, QtWidgets.QComboBox):
                                cell_widget.disconnect()
                                cell_widget.setCurrentText(str(cfg[setting]))
                                cell_widget.currentTextChanged.connect(lambda text_, setting_=setting, i_=i: self.commit_configuration(
                                    text=text_, setting=setting_, config='Channel', idx=i_))
                            else:
                                table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg[setting])))

                        table.cellChanged.connect(lambda row, col, table_=table, i_=i, settings_=settings: self.commit_configuration(
                            table=table_, row=row, setting=settings_[row], config='Channel', idx=i_))

                if 'Trigger' in configuration:
                    cfg_tr = configuration['Trigger']
                    n_triggers = len(cfg_tr)
                    for i, cfg in zip(range(n_triggers), cfg_tr):
                        table = self.outer.trigger_tables[i]
                        table.cellChanged.disconnect()
                        n_configuration = len(cfg)
                        settings = list(cfg.keys())
                        for row, setting in zip(range(n_configuration), settings):
                            cell_widget = table.cellWidget(row, 0)
                            if isinstance(cell_widget, QtWidgets.QComboBox):
                                cell_widget.disconnect()
                                cell_widget.setCurrentText(str(cfg[setting]))
                                cell_widget.currentTextChanged.connect(lambda text_, setting_=setting, i_=i: self.commit_configuration(
                                    text=text_, setting=setting_, config='Trigger', idx=i_))
                            else:
                                table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg[setting])))

                        table.cellChanged.connect(lambda row, col, table_=table, i_=i, settings_=settings: self.commit_configuration(
                            table=table_, row=row, setting=settings_[row], config='Trigger', idx=i_))

                self.outer.set_range = True


        def commit_configuration(self, setting, config, row=None, table=None, text=None, idx=None):
            configuration = self.outer.configuration
            if table is not None:
                item = table.item(row, 0).text().strip()
                try:
                    if idx is None:
                        if isinstance(configuration[config][setting], long):
                            item_converted = float(item)
                        else:
                            item_converted = type(configuration[config][setting])(item)
                        configuration[config][setting] = item_converted
                    else:
                        item_converted = type(configuration[config][idx][setting])(item)
                        configuration[config][idx][setting] = item_converted
                except:
                    pass
            else:
                if idx is None:
                    item_converted = type(configuration[config][setting])(text)
                    configuration[config][setting] = item_converted
                else:
                    item_converted = type(configuration[config][idx][setting])(text)
                    configuration[config][idx][setting] = item_converted

            self.background_data_acquisition.mutex.lock()
            if self.background_data_acquisition._stop:
                self.outer.configuration = configuration
                if config == 'Acquisition' and setting == 'Channels':
                    self.update_configuration(additional_update=self.update_channels)
                else:
                    self.update_configuration()

            else:
                try:
                    self.background_data_acquisition.updated_configuration.disconnect()
                except:
                    pass
                self.background_data_acquisition.new_configuration = configuration
                if config == 'Acquisition' and setting == 'Channels':
                    self.background_data_acquisition.updated_configuration.connect(lambda update_channels_=self.update_channels : self.update_configuration(update_channels_))
                else:
                    self.background_data_acquisition.updated_configuration.connect(self.update_configuration)
            self.background_data_acquisition.mutex.unlock()

        def update_channels(self):
            configuration = self.outer.configuration
            configuration_caps = self.outer.configuration_caps
            self.graphicsLayoutWidget_channels.clear()
            self.outer.channel_plots = {}
            if self.graphicsLayoutWidget_ffts is not None:
                self.graphicsLayoutWidget_ffts.clear()
                self.outer.fft_plots = {}

            channels = ast.literal_eval(configuration['Acquisition']['Channels'])
            for ch, i in zip(channels, range(len(channels))):
                self.outer.channel_plots[i] = self.graphicsLayoutWidget_channels.addPlot(title='Channel ' + str(ch).zfill(2), row=i, col=0)
                self.outer.channel_plots[i].setLabel('bottom', text='Runtime', units='s')
                self.outer.channel_plots[i].disableAutoRange()
                # self.outer.channel_plots[i].setDownsampling(auto=True)

                if self.graphicsLayoutWidget_ffts is not None:
                    self.outer.fft_plots[i] = self.graphicsLayoutWidget_ffts.addPlot(title='Channel ' + str(ch).zfill(2), row=i, col=0)
                    self.outer.fft_plots[i].setLabel('bottom', text='Frequency', units='Hz')
                    self.outer.fft_plots[i].disableAutoRange()
                    self.outer.fft_plots[i].setLogMode(y=True, x=False)
                    # self.outer.fft_plots[i].setDownsampling(auto=True)

            for i in self.outer.channel_tables:
                self.outer.channel_tables[i].disconnect()
                self.tabWidget_channel_settings.removeTab(0)
            self.outer.channel_tables = {}

            cfg_ch = configuration['Channel']
            n_channels = len(channels)
            for i, cfg, ch in zip(range(n_channels), cfg_ch, channels):
                tab = QtWidgets.QWidget()
                layout = QtWidgets.QGridLayout()
                tab.setLayout(layout)
                table = QtWidgets.QTableWidget()
                layout.addWidget(table)
                self.tabWidget_channel_settings.addTab(tab, 'Channel ' + str(ch).zfill(2))
                self.outer.channel_tables[i] = table

                n_configuration = len(cfg)
                table.setColumnCount(1)
                table.setRowCount(n_configuration)
                table.horizontalHeader().setVisible(False)
                table.horizontalHeader().setStretchLastSection(True)
                settings = list(cfg.keys())
                for row, setting in zip(range(n_configuration), settings):
                    table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
                    if setting not in configuration_caps['Channel'][i]:
                        table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(cfg[setting])))
                    else:
                        combo = QtWidgets.QComboBox()
                        for choice in configuration_caps['Channel'][i][setting]:
                            combo.addItem(str(choice))
                        table.setCellWidget(row, 0, combo)
                        combo.setCurrentText(str(cfg[setting]))
                        combo.currentTextChanged.connect(
                            lambda text_, setting_=setting, i_=i: self.commit_configuration(
                                text=text_, setting=setting_, config='Channel', idx=i_))

                table.cellChanged.connect(
                    lambda row, col, table_=table, i_=i, settings_=settings: self.commit_configuration(
                        table=table_, row=row, setting=settings_[row], config='Channel', idx=i_))

        def single_shot(self):
            # self.outer.lab_control.log_file.write('doing a single shot\n')
            self.background_data_acquisition.mutex.lock()
            if self.background_data_acquisition._stop:
                self.background_data_acquisition._stop = False
                self.background_data_acquisition.updated_data.connect(lambda time_data: self.process_data(time_data, thread=False))
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()



        def acquire(self):
            self.background_data_acquisition.mutex.lock()
            if self.background_data_acquisition._stop:
                self.background_data_acquisition._stop = False
                self.background_data_acquisition.updated_data.connect(lambda time_data: self.process_data(time_data, thread=True))
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()

        def stop_acquire(self):
            # print('calling stop')
            try:
                self.background_data_acquisition.updated_data.disconnect()
            except:
                pass
            self.outer.abort()
            self.background_data_acquisition.stop()

        def process_data(self, time_data, thread=False):
            # self.outer.lab_control.log_file.write('processing data')

            if time_data is not None:
                self.outer.time = time_data[0]
                self.outer.data = time_data[1]
                if self.graphicsLayoutWidget_ffts is not None:
                    sample_rate = self.outer.configuration['Acquisition']['SampleRate']

                if thread:
                    self.checkBox_display_data.setChecked(True)

                if self.checkBox_auto_export.isChecked() and not thread:
                    self.export()

                if self.checkBox_display_data.isChecked():
                    if len(self.outer.data.shape) == 2:
                        for i in range(self.outer.data.shape[0]):
                            self.outer.channel_plots[i].plot(self.outer.time, self.outer.data[i], clear=True)
                    elif len(self.outer.data.shape) == 3:
                        for i in range(self.outer.data.shape[1]):
                            self.outer.channel_plots[i].clear()
                            if self.graphicsLayoutWidget_ffts is not None:
                                self.outer.fft_plots[i].clear()
                            for j in range(self.outer.data.shape[2]):
                                self.outer.channel_plots[i].plot(self.outer.time, self.outer.data[:,i,j],
                                                                 pen=pg.mkPen(width=1, color=pg.intColor(j, hues=self.outer.data.shape[2])))
                                if self.graphicsLayoutWidget_ffts is not None:
                                    self.outer.fft_plots[i].plot(*(fft_func(self.outer.data[:, i, j], sample_rate)[::-1]),
                                                                     pen=pg.mkPen(width=1, color=pg.intColor(j, hues=
                                                                     self.outer.data.shape[2])))
                        if self.outer.set_range:
                            for i in range(self.outer.data.shape[1]):
                                self.outer.channel_plots[i].setDownsampling(ds=False)
                                self.outer.channel_plots[i].autoRange()
                                self.outer.channel_plots[i].setDownsampling(auto=True, mode='peak', ds=True)
                                if self.graphicsLayoutWidget_ffts is not None:
                                    self.outer.fft_plots[i].setDownsampling(ds=False)
                                    self.outer.fft_plots[i].autoRange()
                                    self.outer.fft_plots[i].setDownsampling(auto=True, mode='peak', ds=True)
                            self.outer.set_range = False

                else:
                    for ch_pl in self.outer.channel_plots.values():
                        ch_pl.clear()
                    if self.graphicsLayoutWidget_ffts is not None:
                        for fft_pl in self.outer.fft_plots.values():
                            fft_pl.clear()

            self.background_data_acquisition.mutex.lock()
            if thread and not self.background_data_acquisition._stop:
                # print('continuing measurement')
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()
                self.stop_acquire()
            # self.outer.lab_control.log_file.write('processed data')
            # if self.outer.lab_control.sync_lock[1] is not None:
            #     self.outer.lab_control.sync_lock[1].mutex.lock()
            self.outer.lab_control.sync_lock[0] = False
                # self.outer.lab_control.sync_lock[1].mutex.unlock()

        def save_config(self):
            fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', os.path.split(__file__)[0], '*.ini')
            if fname[0] != '':
                if not fname[0].endswith('ini'):
                    fname = fname[0] + '.ini'
                else:
                    fname = fname[0]
                self.outer.write_ini(fname)

        def load_config(self):
            fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Save File', os.path.split(__file__)[0], '*.ini')
            if fname[0] != '':
                if not fname[0].endswith('ini'):
                    fname = fname[0] + '.ini'
                else:
                    fname = fname[0]

                self.background_data_acquisition.mutex.lock()
                if self.background_data_acquisition._stop:
                    self.outer.read_ini(fname)
                    self.update_configuration(additional_update=self.update_channels)
                else:
                    try:
                        self.background_data_acquisition.updated_configuration.disconnect()
                    except:
                        pass
                    self.background_data_acquisition.new_configuration = self.outer.read_ini(fname, return_config=True)
                    self.background_data_acquisition.updated_configuration.connect(lambda update_channels_=self.update_channels: self.update_configuration(update_channels_))
                self.background_data_acquisition.mutex.unlock()

        def closeEvent(self, *args, **kwargs):
            super(QtWidgets.QMainWindow, self).closeEvent(*args, **kwargs)
            self.stop_acquire()
            print('Waiting for Acquisition in Background to finish. Please be patient.')
            while True:
                self.background_data_acquisition.mutex.lock()
                if self.background_data_acquisition.finished_last:
                    self.outer.release()
                    self.background_data_acquisition.mutex.unlock()
                    print('Closed Connection to ', self.outer)
                    return

                self.background_data_acquisition.mutex.unlock()
                sleep(0.5)


        def fft(self, state):
            if state:
                self.fft_widget = QtWidgets.QWidget()
                layout = QtWidgets.QGridLayout(self.fft_widget)
                self.graphicsLayoutWidget_ffts = pg.GraphicsLayoutWidget(self.fft_widget)
                layout.addWidget(self.graphicsLayoutWidget_ffts)
                self.splitter.addWidget(self.fft_widget)

                configuration = self.outer.configuration
                if configuration is not None and 'Acquisition' in configuration:
                    if 'Channels' in configuration['Acquisition']:
                        channels = ast.literal_eval(configuration['Acquisition']['Channels'])
                        for ch, i in zip(channels, range(len(channels))):
                            self.outer.fft_plots[i] = self.graphicsLayoutWidget_ffts.addPlot(
                                title='Channel ' + str(ch).zfill(2), row=i, col=0)
                            self.outer.fft_plots[i].setLabel('bottom', text='Frequency', units='Hz')
                            self.outer.fft_plots[i].disableAutoRange()
                            self.outer.fft_plots[i].setLogMode(y=True, x=False)
                            # self.outer.fft_plots[i].setDownsampling(auto=True)

                self.outer.set_range = True

            else:
                self.fft_widget.close()
                self.graphicsLayoutWidget_ffts = None

    def timetrace(self):
        print('WARNING: This function was not implemented yet.')
        return None



class GhostScope(Scope):
    isDevice = True
    kind = 'dummy Oscilloscope'

    def __init__(self, name, settings=None, parent=None):
        self.name = name
        self.sync_lock = parent.sync_lock
        if settings is not None:
            self.settings = settings

    def release(self):
        return

    @staticmethod
    def find_devices():
        devices = [GhostScope('default')]
        return devices

    @property
    def configuration(self):
        _settings = {'Channel': [{'name' : 1}, {'name' : 2}, {'name' : 3}, {'name' : 4}],
                     'Acquisition' : {'sampling_rate': 100, 'sampling_time': 10}}
        return _settings

    class ManualInit(QtWidgets.QMainWindow):
        def __init__(self, parent=None):
            super(GhostScope.ManualInit, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'manual_ghostscope.ui'), self)

        def update_connection_settings(self):
            return None


    def timetrace(self):
        sleep(1)
        time = np.linspace(0, self.settings['sampling_time'], self.settings['sampling_time'] * self.settings['sampling_rate'])
        data = np.array([np.random.rand(self.settings['sampling_time'] * self.settings['sampling_rate']) for ch in self.settings['channel']])
        return [time, data]

try:
    from optomechanics.experiment.lab_control.devices.oscilloscope.gage.gage import GageCard
    class GaGe(Scope, GageCard):
        isDevice = True
        kind = 'Oscilloscope'
        connection_settings = {'board_type' : 0,
                               'channels' : 0,
                               'sample_bits' : 0,
                               'index' : 0}

        def __init__(self, name, connection_settings=None, lab_control=None):
            self.name = name
            self.lab_control = lab_control
            if connection_settings is not None:
                self.connection_settings = connection_settings
            self.initialize(self.connection_settings['board_type'], self.connection_settings['channels'], self.connection_settings['sample_bits'], self.connection_settings['index'])
        class ManualInit(QtWidgets.QMainWindow):
            def __init__(self, parent=None):
                super(GaGe.ManualInit, self).__init__(parent)
                uic.loadUi(os.path.join('ui_files', 'manual_gage.ui'), self)

            def update_connection_settings(self):
                board_type = self.lineEdit_board_type.text()
                channels = self.lineEdit_channels.text().strip()
                sample_bits = self.lineEdit_sample_bits.text().strip()
                index = self.lineEdit_index.text().strip()
                connection_settings = {}
                if board_type == '':
                    connection_settings['board_type'] = 0
                else:
                    connection_settings['board_type'] = board_type
                if channels == '':
                    connection_settings['channels'] = 0
                else:
                    connection_settings['channels'] = channels
                if sample_bits == '':
                    connection_settings['sample_bits'] = 0
                else:
                    connection_settings['sample_bits'] = sample_bits
                if index == '':
                    connection_settings['index'] = 0
                else:
                    connection_settings['index'] = index
                return connection_settings

        # @profile
        def timetrace(self):
            # self.lab_control.log_file.write('getting a timetrace\n')
            configuration =  copy(self.configuration)
            sample_rate = configuration['Acquisition']['SampleRate']
            segment_size = configuration['Acquisition']['DepthPostTrigger'] + configuration['Acquisition']['DepthPreTrigger']

            time = np.linspace(0, (segment_size - 1) / sample_rate, segment_size)
            self.acquire()
            data = self.get()
            # self.lab_control.log_file.write('there is data\n')
            if time.shape[0] == data.shape[0]:
                # self.lab_control.log_file.write('got a timetrace\n')
                return [time, data]
            else:
                print('time and measured data don\'t fit together', time.shape, data.shape)
                return None

except:
    pass


