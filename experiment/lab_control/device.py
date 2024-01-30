import os
from PyQt5 import QtWidgets, QtCore, Qt, uic, QtGui
from copy import copy
from memory_profiler import profile
from optomechanics.experiment.lab_control.hdf5 import *
import ast
from datetime import datetime as dt
from optomechanics.experiment.device.gauges.thyracont import Thyracont
from optomechanics.experiment.lab_control.devices.piezo_stage.smaract import _SmarAct
from time import sleep
import optomechanics.experiment.lab_control.hdf5 as h5
import pandas as pd
import json

class Device(object):
    isDevice = False
    name = None
    kind = None
    dock = None
    inner = None

    @property
    def information(self):
        info_dict = {'name' : self.name, 'kind' : self.kind}
        return info_dict

    @staticmethod
    def find_devices():
        devices = []
        return devices


    class ManualInit(QtWidgets.QMainWindow):
        def __init__(self, parent=None):
            super(Device.ManualInit, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'not_implemented.ui'), self)


    def ControlWidget(self, parent=None):
        return self._ControlWidget(self, parent)

    class _ControlWidget(QtWidgets.QMainWindow):
        def __init__(self, outer, parent):
            self.outer = outer
            self.outer.inner = self
            super(self.outer._ControlWidget, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'not_implemented.ui'), self)

    def BackgroundThread(self, process):
        return self._BackgroundThread(self, process)

    class _BackgroundThread(QtCore.QThread):
        # counter = 0
        updated_data = QtCore.pyqtSignal(object)
        updated_configuration = QtCore.pyqtSignal()
        mutex = QtCore.QMutex()
        _stop = True
        finished_last = True
        new_configuration = None
        do_something = None
        def __init__(self, outer, process):
            self.process = process
            self.outer = outer
            QtCore.QThread.__init__(self)

        def __del__(self):
            self.wait()


        def run(self):
            self.mutex.lock()
            if self._stop:
                # print('got stop command')
                self.finished_last = True
                self.mutex.unlock()
                return
            self.mutex.unlock()
            # print('runnning measurement in background')
            self.mutex.lock()
            self.finished_last = False
            if self.new_configuration is not None:
                # print('updateing new config')
                self.outer.configuration = self.new_configuration
                self.updated_configuration.emit()
                self.new_configuration = None

            if self.do_something is not None:
                self.do_something()
                self.do_something = None

            self.mutex.unlock()
            try:
                data = self.process()
                self.updated_data.emit(data)
            except:
                pass
            self.mutex.lock()
            self.finished_last = True
            self.mutex.unlock()

            # self.counter+=1

        def stop(self):
            self.mutex.lock()
            self._stop = True
            self.mutex.unlock()


class Gauge(Device, Thyracont):
    hdf5_file = None
    operation = None
    isDevice = True
    configuration = {'Acquisition' : {'Addresses' : '[1]',
                                      'SampleRate' : 1.,
                                      'MaxSampleSize' : 10000}}
    configuration_caps = {'Acquisition' : {}}

    datetimes = {}
    pressures = {}
    reference_time = None
    clear_data = False

    def __init__(self, name, connection_settings=None, lab_control=None):
        self.name = name
        self.lab_control = lab_control
        self.address_plots = {}
        if connection_settings is not None:
            self.connection_settings = connection_settings
        super(Gauge, self).__init__(**self.connection_settings)

    def release(self):
        self.__del__()

    def read_ini(self, filename, return_config=False):
        with open(filename) as f:
            if return_config:
                return json.load(f)
            else:
                self.configuration = json.load(f)

    def write_ini(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.configuration, f, indent=4, sort_keys=True)


    class ManualInit(QtWidgets.QMainWindow):
        connection_settings = {'port': '',
                               'baudrate': 9600,
                               'timeout': 0.1}

        def __init__(self, parent=None):
            super(Gauge.ManualInit, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'manual_gauge.ui'), self)
            self.lineEdit_port.setText(str(self.connection_settings['port']))
            self.lineEdit_baudrate.setText(str(self.connection_settings['baudrate']))
            self.lineEdit_timeout.setText(str(self.connection_settings['timeout']))

        def update_connection_settings(self):
            port = int(self.lineEdit_port.text().strip()) - 1
            baudrate = int(self.lineEdit_baudrate.text().strip())
            timeout = float(self.lineEdit_timeout.text().strip())
            connection_settings = {}
            if port == '':
                connection_settings['port'] = 0
            else:
                connection_settings['port'] = port
            if baudrate == '':
                connection_settings['baudrate'] = 9600
            else:
                connection_settings['baudrate'] = baudrate
            if timeout == '':
                connection_settings['timeout'] = 0.1
            else:
                connection_settings['timeout'] = timeout
            return connection_settings

    def export(self, kind='individual'):
        if self.operation is not None:
            keys = self.pressures.keys()
            data_tmp = [pd.Series(self.pressures[key], index=self.datetimes[key]) for key in keys]
            data_tmp = pd.concat(data_tmp, axis=1)
            data_tmp['runtime'] = data_tmp.index
            self.operation.data = data_tmp
            keys = ['Address ' + str(key).zfill(2) for key in keys]
            units = ['bar'] * len(keys)
            units.append('s')
            keys.append('runtime')
            self.operation.attributes['timestamp'] = str(self.reference_time)
            self.operation.attributes['label'] = keys
            self.operation.attributes['unit'] = units
            if kind == 'individual' and self.hdf5_file is not None:
                self.hdf5_file.open()
                self.hdf5_file.modified = dt.now()
                self.hdf5_file.add_operation(self.operation)
                self.hdf5_file.close()
            else:
                self.operation.name = self.operation.name + '_' + self.name
                return self.operation


    class _ControlWidget(QtWidgets.QMainWindow):
        def __init__(self, outer, parent):
            self.outer = outer
            self.outer.inner = self
            super(self.outer._ControlWidget, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'control_gauge.ui'), self)

            self.background_data_acquisition = self.outer.BackgroundThread(self.outer.data)

            self.channel_plot_widgets()
            self.populate_configuration()

            self.pushButton_acquire.clicked.connect(self.acquire)
            self.pushButton_pause.clicked.connect(self.stop_acquire)
            self.pushButton_clear.clicked.connect(self.clear_acquire)
            self.pushButton_export_settings.clicked.connect(self.export_settings)
            self.pushButton_export.clicked.connect(self.outer.export)
            self.pushButton_save_config.clicked.connect(self.save_config)
            self.pushButton_load_config.clicked.connect(self.load_config)

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
                    self.update_configuration(additional_update=self.update_addresses)
                else:
                    try:
                        self.background_data_acquisition.updated_configuration.disconnect()
                    except:
                        pass
                    self.background_data_acquisition.new_configuration = self.outer.read_ini(fname, return_config=True)
                    self.background_data_acquisition.updated_configuration.connect(lambda update_addresses_=self.update_addresses: self.update_configuration(update_addresses_))
                self.background_data_acquisition.mutex.unlock()


        def export_settings(self):
            export_dialog = h5.Frontend(self, self.outer)
            export_dialog.show()

        def channel_plot_widgets(self):
            configuration = self.outer.configuration
            if configuration is not None and 'Acquisition' in configuration:
                if 'Addresses' in configuration['Acquisition']:
                    addresses = ast.literal_eval(configuration['Acquisition']['Addresses'])
                    self.tableWidget_current_pressure.setColumnCount(1)
                    self.tableWidget_current_pressure.setRowCount(len(addresses))
                    for ad, i in zip(addresses, range(len(addresses))):
                        self.outer.address_plots[i] = self.graphicsLayoutWidget_channels.addPlot(title='Address ' + str(ad).zfill(2), row=i, col=0)
                        self.outer.address_plots[i].setLabel('bottom', text='Runtime', units='s')
                        self.outer.address_plots[i].setLabel('left', text='Pressure', units='bar')

                        self.tableWidget_current_pressure.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem('Address ' + str(ad).zfill(2)))

                    heights = sum([self.tableWidget_current_pressure.rowHeight(i) for i in range(len(addresses))])
                    self.tableWidget_current_pressure.setFixedHeight(heights)

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

        def acquire(self):
            self.background_data_acquisition.mutex.lock()
            if self.background_data_acquisition._stop:
                self.background_data_acquisition._stop = False
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.updated_data.connect(lambda time_data: self.process_data(time_data, thread=True))
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()

        def process_data(self, time_data, thread=False):
            if time_data is not None:
                if self.outer.clear_data:
                    self.outer.datetimes = {}
                    self.outer.pressures = {}
                    self.outer.reference_time = None
                    self.outer.clear_data = False

                time = time_data[0]
                data = time_data[1]
                maxSamples = self.outer.configuration['Acquisition']['MaxSampleSize']
                addresses = ast.literal_eval(self.outer.configuration['Acquisition']['Addresses'])
                if self.outer.reference_time is None:
                    self.outer.reference_time = time
                    self.lineEdit_reference_timestamp.setText(str(self.outer.reference_time))
                for i in range(len(addresses)):
                    if addresses[i] not in self.outer.datetimes:
                        self.outer.datetimes[addresses[i]] = [(time - self.outer.reference_time).total_seconds()]
                        if len(data) > i:
                            self.outer.pressures[addresses[i]] = [data[i]]
                        else:
                            self.outer.pressures[addresses[i]] = [np.nan]

                    else:
                        self.outer.datetimes[addresses[i]].append((time - self.outer.reference_time).total_seconds())
                        if len(data) > i:
                            self.outer.pressures[addresses[i]].append(data[i])
                        else:
                            self.outer.pressures[addresses[i]].append(np.nan)


                    if len(self.outer.datetimes[addresses[i]]) > maxSamples:
                        self.outer.datetimes[addresses[i]] = self.outer.datetimes[addresses[i]][-1 * maxSamples:]
                        self.outer.pressures[addresses[i]] = self.outer.pressures[addresses[i]][-1 * maxSamples:]

                # if self.checkBox_auto_export.isChecked() and not thread:
                #     self.export()

                for i in range(len(addresses)):
                    # print(self.outer.address_plots)
                    # print(self.outer.datetimes[addresses[i]], self.outer.pressures[addresses[i]])
                    if i in self.outer.address_plots:
                        self.outer.address_plots[i].plot(self.outer.datetimes[addresses[i]], self.outer.pressures[addresses[i]], clear=True)
                        self.tableWidget_current_pressure.setItem(i, 0, QtWidgets.QTableWidgetItem('%.2E mbar' % (1000 * data[i])))
                # else:
                #     for ch_pl in self.outer.address_plots.values():
                #         ch_pl.clear()

                # self.lineEdit_current_pressure.setText(str())

            self.background_data_acquisition.mutex.lock()
            if thread and not self.background_data_acquisition._stop:
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()
                self.stop_acquire()

        def stop_acquire(self):
            try:
                self.background_data_acquisition.updated_data.disconnect()
            except:
                pass
            self.background_data_acquisition.stop()

        def clear_acquire(self):
            self.outer.clear_data = True

        def update_configuration(self, additional_update=None):
            if additional_update is not None:
                additional_update()
            configuration = self.outer.configuration
            if configuration is not None:
                if 'Acquisition' in configuration:
                    cfg_acq = configuration['Acquisition']
                    n_configuration = len(cfg_acq)
                    self.tableWidget_settings.cellChanged.disconnect()
                    self.tableWidget_settings.clear()
                    settings = list(cfg_acq.keys())
                    for row, setting in zip(range(n_configuration), settings):
                        self.tableWidget_settings.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
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

                        if setting == 'Addresses':
                            addresses = ast.literal_eval(item_converted)
                            if isinstance(addresses, list):
                                configuration[config][setting] = item_converted
                        else:
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
                if config == 'Acquisition' and setting == 'Addresses':
                    self.update_configuration(additional_update=self.update_addresses)
                else:
                    self.update_configuration()

            else:
                try:
                    self.background_data_acquisition.updated_configuration.disconnect()
                except:
                    pass
                self.background_data_acquisition.new_configuration = configuration
                if config == 'Acquisition' and setting == 'Addresses':
                    self.background_data_acquisition.updated_configuration.connect(lambda update_addresses_=self.update_addresses : self.update_configuration(update_addresses_))
                else:
                    self.background_data_acquisition.updated_configuration.connect(self.update_configuration)
            self.background_data_acquisition.mutex.unlock()

        def update_addresses(self):
            configuration = self.outer.configuration
            configuration_caps = self.outer.configuration_caps
            self.graphicsLayoutWidget_channels.clear()
            self.outer.address_plots = {}

            addresses = ast.literal_eval(configuration['Acquisition']['Addresses'])
            self.tableWidget_current_pressure.clear()
            self.tableWidget_current_pressure.setColumnCount(1)
            self.tableWidget_current_pressure.setRowCount(len(addresses))

            for ad, i in zip(addresses, range(len(addresses))):
                self.outer.address_plots[i] = self.graphicsLayoutWidget_channels.addPlot(title='Address ' + str(ad).zfill(2), row=i, col=0)
                self.outer.address_plots[i].setLabel('bottom', text='Runtime', units='s')
                self.outer.address_plots[i].setLabel('left', text='Pressure', units='bar')

                self.tableWidget_current_pressure.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem('Address ' + str(ad).zfill(2)))

            heights = sum([self.tableWidget_current_pressure.rowHeight(i) for i in range(len(addresses))])
            self.tableWidget_current_pressure.setFixedHeight(heights)

    def data(self):
        sleep(1 / self.configuration['Acquisition']['SampleRate'])
        addresses = ast.literal_eval(self.configuration['Acquisition']['Addresses'])
        data = []
        for addy in addresses:
            try:
                self.address = addy
                data.append(self.pressure / 1000.)
            except:
                data.append(np.nan)

        time = dt.now()
        return time, np.array(data)


class SmarAct(Device, _SmarAct):
    hdf5_file = None
    operation = None
    isDevice = True
    configuration_caps = {'Acquisition' : {}}

    datetimes = {}
    positions = {}
    reference_time = None
    clear_data = False

    def __init__(self, name, connection_settings=None, lab_control=None):
        self.name = name
        self.lab_control = lab_control
        self.position_plots = {}
        if connection_settings is not None:
            self.connection_settings = connection_settings
        super(SmarAct, self).__init__(**self.connection_settings)

    def release(self):
        self.__del__()

    def read_ini(self, filename, return_config=False):
        with open(filename) as f:
            if return_config:
                return json.load(f)
            else:
                self.configuration = json.load(f)

    def write_ini(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.configuration, f, indent=4, sort_keys=True)


    class ManualInit(QtWidgets.QMainWindow):
        connection_settings = {'location': 'usb:id:459394397'}

        def __init__(self, parent=None):
            super(SmarAct.ManualInit, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'manual_smaract.ui'), self)
            self.lineEdit_location.setText(str(self.connection_settings['location']))

        def update_connection_settings(self):
            location = self.lineEdit_location.text().strip()
            connection_settings = {}
            if location == '':
                connection_settings['location'] = 'usb:ix:0'
            else:
                connection_settings['location'] = location
            return connection_settings

    def export(self, kind='individual'):
        if self.operation is not None:
            keys = self.positions.keys()
            data_tmp = [pd.Series(self.positions[key], index=self.datetimes[key]) for key in keys]
            data_tmp = pd.concat(data_tmp, axis=1)
            data_tmp['runtime'] = data_tmp.index
            self.operation.data = data_tmp
            keys = ['Positioner ' + str(key).zfill(2) for key in keys]
            units = ['nm'] * len(keys)
            units.append('s')
            keys.append('runtime')
            self.operation.attributes['timestamp'] = str(self.reference_time)
            self.operation.attributes['label'] = keys
            self.operation.attributes['unit'] = units
            if kind == 'individual' and self.hdf5_file is not None:
                self.hdf5_file.open()
                self.hdf5_file.modified = dt.now()
                self.hdf5_file.add_operation(self.operation)
                self.hdf5_file.close()
            else:
                self.operation.name = self.operation.name + '_' + self.name
                return self.operation


    class _ControlWidget(QtWidgets.QMainWindow):
        def __init__(self, outer, parent):
            self.outer = outer
            self.outer.inner = self
            super(self.outer._ControlWidget, self).__init__(parent)
            uic.loadUi(os.path.join('ui_files', 'control_smaract.ui'), self)

            self.background_data_acquisition = self.outer.BackgroundThread(self.outer.data)

            self.channel_plot_widgets()
            self.populate_configuration()

            self.pushButton_acquire.clicked.connect(self.acquire)
            self.pushButton_pause.clicked.connect(self.stop_acquire)
            self.pushButton_clear.clicked.connect(self.clear_acquire)
            self.pushButton_export_settings.clicked.connect(self.export_settings)
            self.pushButton_export.clicked.connect(self.outer.export)
            self.pushButton_save_config.clicked.connect(self.save_config)
            self.pushButton_load_config.clicked.connect(self.load_config)
            self.pushButton_xp.clicked.connect(lambda x: self.control(0, 1))
            self.pushButton_xm.clicked.connect(lambda x: self.control(0, -1))
            self.pushButton_yp.clicked.connect(lambda x: self.control(1, 1))
            self.pushButton_ym.clicked.connect(lambda x: self.control(1, -1))
            self.pushButton_zp.clicked.connect(lambda x: self.control(2, 1))
            self.pushButton_zm.clicked.connect(lambda x: self.control(2, -1))

            self.label_pic.setPixmap(QtGui.QPixmap(os.path.join(os.path.split(__file__)[0], 'ui_files', 'picture', 'smaract.png')))
            self.label_pic.setScaledContents(True)

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
                    self.update_configuration(additional_update=self.update_addresses)
                else:
                    try:
                        self.background_data_acquisition.updated_configuration.disconnect()
                    except:
                        pass
                    self.background_data_acquisition.new_configuration = self.outer.read_ini(fname, return_config=True)
                    self.background_data_acquisition.updated_configuration.connect(lambda update_addresses_=self.update_addresses: self.update_configuration(update_addresses_))
                self.background_data_acquisition.mutex.unlock()


        def export_settings(self):
            export_dialog = h5.Frontend(self, self.outer)
            export_dialog.show()

        def channel_plot_widgets(self):
            configuration = self.outer.configuration
            if configuration is not None and 'Acquisition' in configuration:
                if 'NumOfChannels' in configuration['Acquisition']:
                    channels = np.arange(configuration['Acquisition']['NumOfChannels'])
                    self.tableWidget_current_position.setColumnCount(1)
                    self.tableWidget_current_position.setRowCount(len(channels))
                    for ad, i in zip(channels, range(len(channels))):
                        self.outer.position_plots[i] = self.graphicsLayoutWidget_channels.addPlot(title='Positioner ' + str(ad).zfill(2), row=i, col=0)
                        self.outer.position_plots[i].setLabel('bottom', text='Runtime', units='s')
                        self.outer.position_plots[i].setLabel('left', text='Position', units='m')

                        self.tableWidget_current_position.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem('Positioner ' + str(ad).zfill(2)))

                    heights = sum([self.tableWidget_current_position.rowHeight(i) for i in range(len(channels))])
                    self.tableWidget_current_position.setFixedHeight(heights)

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

        def acquire(self):
            self.background_data_acquisition.mutex.lock()
            if self.background_data_acquisition._stop:
                self.background_data_acquisition._stop = False
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.updated_data.connect(lambda time_data: self.process_data(time_data, thread=True))
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()

        def process_data(self, time_data, thread=False):
            if time_data is not None:
                if self.outer.clear_data:
                    self.outer.datetimes = {}
                    self.outer.positions = {}
                    self.outer.reference_time = None
                    self.outer.clear_data = False

                time = time_data[0]
                data = time_data[1]
                maxSamples = self.outer.configuration['Acquisition']['MaxSampleSize']
                channels = np.arange(self.outer.configuration['Acquisition']['NumOfChannels'])
                if self.outer.reference_time is None:
                    self.outer.reference_time = time
                    self.lineEdit_reference_timestamp.setText(str(self.outer.reference_time))
                for i in range(len(channels)):
                    if channels[i] not in self.outer.datetimes:
                        self.outer.datetimes[channels[i]] = [(time - self.outer.reference_time).total_seconds()]
                        if len(data) > i:
                            self.outer.positions[channels[i]] = [data[i]]
                        else:
                            self.outer.positions[channels[i]] = [np.nan]

                    else:
                        self.outer.datetimes[channels[i]].append((time - self.outer.reference_time).total_seconds())
                        if len(data) > i:
                            self.outer.positions[channels[i]].append(data[i])
                        else:
                            self.outer.positions[channels[i]].append(np.nan)


                    if len(self.outer.datetimes[channels[i]]) > maxSamples:
                        self.outer.datetimes[channels[i]] = self.outer.datetimes[channels[i]][-1 * maxSamples:]
                        self.outer.positions[channels[i]] = self.outer.positions[channels[i]][-1 * maxSamples:]

                # if self.checkBox_auto_export.isChecked() and not thread:
                #     self.export()

                for i in range(len(channels)):
                    # print(self.outer.position_plots)
                    # print(self.outer.datetimes[addresses[i]], self.outer.pressures[addresses[i]])
                    if i in self.outer.position_plots:
                        self.outer.position_plots[i].plot(self.outer.datetimes[channels[i]], np.array(self.outer.positions[channels[i]]) * 1e-9, clear=True)
                        self.tableWidget_current_position.setItem(i, 0, QtWidgets.QTableWidgetItem('%.2E um' % (1e-4 * data[i])))
                # else:
                #     for ch_pl in self.outer.position_plots.values():
                #         ch_pl.clear()

                # self.lineEdit_current_pressure.setText(str())

            self.background_data_acquisition.mutex.lock()
            if thread and not self.background_data_acquisition._stop:
                self.background_data_acquisition.mutex.unlock()
                self.background_data_acquisition.start()
            else:
                self.background_data_acquisition.mutex.unlock()
                self.stop_acquire()

        def stop_acquire(self):
            try:
                self.background_data_acquisition.updated_data.disconnect()
            except:
                pass
            self.background_data_acquisition.stop()

        def clear_acquire(self):
            self.outer.clear_data = True

        def update_configuration(self, additional_update=None):
            if additional_update is not None:
                additional_update()
            configuration = self.outer.configuration
            if configuration is not None:
                if 'Acquisition' in configuration:
                    cfg_acq = configuration['Acquisition']
                    n_configuration = len(cfg_acq)
                    self.tableWidget_settings.cellChanged.disconnect()
                    self.tableWidget_settings.clear()
                    settings = list(cfg_acq.keys())
                    for row, setting in zip(range(n_configuration), settings):
                        self.tableWidget_settings.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(setting))
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
                self.update_configuration()

            else:
                try:
                    self.background_data_acquisition.updated_configuration.disconnect()
                except:
                    pass
                self.background_data_acquisition.new_configuration = configuration
                self.background_data_acquisition.updated_configuration.connect(self.update_configuration)
            self.background_data_acquisition.mutex.unlock()

        def control(self, channel, dir):
            backgroundMovement = self.outer.BackgroundMovementThread(lambda channel_=channel, dir_=dir: self._control(channel_, dir_))
            backgroundMovement.start()

        def _control(self, channel, dir):
            step = self.lineEdit_step.text().strip()
            if step != '':
                step = dir * int(float(step))
                self.outer.move_relative(channel, step)

    def data(self):
        sleep(1 / self.configuration['Acquisition']['SampleRate'])
        data = self.position
        time = dt.now()
        return time, data

    # def move_relative_sequence3d(self, seq, func=None):
    #     self.backgroundMovement = self.BackgroundMovementThread(lambda seq_=seq, func_=func : self._move_relative_sequence3d(seq_, func_))
    #     self.backgroundMovement.start()

    def move_relative_sequence3d(self, seq, func=None):
        # self.lab_control.log_file.write('##########################################################\n')
        for pos in zip(*seq):
            for ch in range(len(pos)):
                self.move_relative(ch, pos[ch])
            while any(self.status == 4):
                sleep(0.01)
            if func is not None:
                func()
                self.lab_control.sync_lock = [True, None]
                # self.backgroundMovement.mutex.lock()
                counter = 0
                while self.lab_control.sync_lock[0]:
                    # self.backgroundMovement.mutex.unlock()
                    sleep(0.01)
                    QtWidgets.QApplication.processEvents()
                    # self.backgroundMovement.mutex.lock()
                    # self.lab_control.log_file.write('am waiting\n')
                    counter += 1
                    if counter >100:
                        return
                # self.backgroundMovement.mutex.unlock()

    def BackgroundMovementThread(self, process):
        return self._BackgroundMovementThread(self, process)

    class _BackgroundMovementThread(QtCore.QThread):
        mutex = QtCore.QMutex()
        _stop = False
        def __init__(self, outer, process):
            self.process = process
            self.outer = outer
            QtCore.QThread.__init__(self)

        def __del__(self):
            self.wait()

        def run(self):
            self.mutex.lock()
            if self._stop:
                # print('got stop command')
                self.mutex.unlock()
                return
            self.mutex.unlock()
            # print('runnning measurement in background')
            self.process()

        def stop(self):
            self.mutex.lock()
            self._stop = True
            self.mutex.unlock()





