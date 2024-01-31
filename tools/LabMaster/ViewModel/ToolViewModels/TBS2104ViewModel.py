from QtModularUiPack.ViewModels import BaseViewModel
from TBS2104.TBS2104 import TBS2104, CH1, CH2, CH3, CH4, REF1, REF2, MATH, COUPLING_DC, COUPLING_AC, COUPLING_GND, TRIGGER_MODES
from time import sleep
from threading import Thread
from PyQt5.QtCore import QObject, QMetaObject, Q_ARG, pyqtSlot
import numpy as np
import sys
import json
import os
import h5py


CHANNELS = [CH1, CH2, CH3, CH4, REF1, REF2, MATH]
COUPLING_MODES = [COUPLING_DC, COUPLING_AC, COUPLING_GND]


class TBS2104ViewModel(QObject, BaseViewModel):
    """
    Data context for the TBS2104 Oscilloscope
    """

    name = 'tbs'

    _TBS2104_config_file = 'TBS2104FrameConfig.json'

    @property
    def host(self):
        """
        Gets the hostname of the device
        """
        return self._host

    @host.setter
    def host(self, value):
        """
        Sets the hostname of the device
        """
        self._host = value
        self.notify_change('host')

    @property
    def connection_button_text(self):
        """
        Gets the text of the connect / disconnect button
        """
        return 'disconnect' if self.connected else 'connect'

    @property
    def connected(self):
        """
        True if the device is connected
        """
        return self._connected

    @connected.setter
    def connected(self, value):
        """
        Sets if the device is connected
        """
        self._connected = value
        self.status = 'connected' if value else 'disconnected'
        self.notify_change('connected')
        self.notify_change('connection_button_text')

    @property
    def status(self):
        """
        Gets the status text of the device
        """
        return self._status

    @status.setter
    def status(self, value):
        """
        Sets the status text of the device
        """
        self._status = value
        self.notify_change('status')

    @property
    def auto_connect(self):
        """
        Gets if the device should be connected to automatically on startup
        :return:
        """
        return self._auto_connect

    @auto_connect.setter
    def auto_connect(self, value):
        """
        Sets if the device should be connected to automatically on startup
        """
        self._auto_connect = value
        self.notify_change('auto_connect')

    @property
    def get_data_allowed(self):
        """
        Gets if data retrieval is currently possible
        """
        return self._get_data_allowed

    @get_data_allowed.setter
    def get_data_allowed(self, value):
        """
        Sets if data retrieval is currently possible
        """
        self._get_data_allowed = value
        self.notify_change('get_data_allowed')

    @property
    def x(self):
        """
        Gets the currently displayed x data
        """
        return self._x

    @x.setter
    def x(self, value):
        """
        Sets the currently displayed x data
        """
        self._x = value
        self.notify_change('x')

    @property
    def y(self):
        """
        Gets the currently displayed y data
        """
        return self._y

    @y.setter
    def y(self, value):
        """
        Sets the currently displayed y data
        """
        self._y = value
        self.notify_change('y')

    @property
    def x_unit(self):
        """
        Gets the units of the x axis
        """
        return self._tbs.x_unit if self.connected else 's'

    @property
    def y_unit(self):
        """
        Gets the units of the y axis
        """
        return self._tbs.y_unit if self.connected else 'V'

    @property
    def start(self):
        """
        Gets the start point of the current channel
        :return:
        """
        return self._tbs.start if self.connected else 0

    @start.setter
    def start(self, value):
        """
        Sets the start point of the current channel
        """
        if self.connected:
            self._tbs.start = value
            self.notify_change('start')

    @property
    def stop(self):
        """
        Gets the endpoint of the current channel
        """
        return self._tbs.stop if self.connected else 0

    @stop.setter
    def stop(self, value):
        """
        Sets the endpoint of the current channel
        """
        if self.connected:
            self._tbs.stop = value
            self.notify_change('stop')

    @property
    def length(self):
        """
        Gets the available points of the current channel
        """
        return self._tbs.length if self.connected else 0

    @length.setter
    def length(self, value):
        """
        Sets the available points of the current channel
        """
        if self.connected:
            self._tbs.length = value
            self.notify_change('length')

    @property
    def vertical_scale(self):
        """
        Gets the vertical scale of the current channel
        """
        if self.connected:
            return self._tbs.vertical_scale
        else:
            return 1

    @vertical_scale.setter
    def vertical_scale(self, value):
        """
        Sets the vertical scale of the current channel
        """
        if self.connected:
            self._tbs.vertical_scale = value
            self.notify_change('vertical_scale')

    @property
    def vertical_position(self):
        """
        Gets the vertical position of the current channel
        """
        if self.connected:
            return self._tbs.vertical_position
        else:
            return 0

    @vertical_position.setter
    def vertical_position(self, value):
        """
        Sets the vertical position of the current channel
        """
        if self.connected:
            self._tbs.vertical_position = value
            self.notify_change('vertical_position')

    @property
    def vertical_units(self):
        """
        Gets the vertical units of the current channel
        """
        if self.connected:
            return self._tbs.channel_unity_y
        else:
            return 'V'

    @property
    def selected_channel(self):
        """
        Gets the index of the selected channel
        """
        return self._selected_channel

    @selected_channel.setter
    def selected_channel(self, value):
        """
        Sets the index of the selected channel
        """
        if self.connected:
            self._selected_channel = value
            self._tbs.source = CHANNELS[value]
            self.notify_change('selected_channel')
            self._notify_all()

    @property
    def selected_coupling(self):
        """
        Gets the coupling of the current channel
        """
        if self.connected:
            return COUPLING_MODES.index(self._tbs.coupling)
        else:
            return -1

    @selected_coupling.setter
    def selected_coupling(self, value):
        """
        Sets the coupling of the selected channel (use index)
        """
        if self.connected:
            self._tbs.coupling = COUPLING_MODES[value]
            self.notify_change('selected_coupling')

    @property
    def selected_trigger_mode(self):
        """
        Gets the trigger mode of the scope
        """
        if self.connected:
            return TRIGGER_MODES.index(self._tbs.trigger_mode)
        else:
            return -1

    @selected_trigger_mode.setter
    def selected_trigger_mode(self, value):
        """
        Sets the trigger mode of the scope
        """
        if self.connected:
            self._tbs.trigger_mode = TRIGGER_MODES[value]
            self.notify_change('selected_trigger_mode')

    @property
    def measurement_in_progress(self):
        """
        Gets if a measurement is currently in progress
        """
        if self.connected:
            state = True
            try:
                state = self._tbs.acquisition_state != '0'
            except:
                pass
            return state
        else:
            return False

    @property
    def get_data_after_measurement(self):
        """
        Gets if the measured data should be queried after the current measurement has been completed
        """
        return self._get_data_after_measurement

    @get_data_after_measurement.setter
    def get_data_after_measurement(self, value):
        """
        Sets if the measured data should be queried after the current measurement has been completed
        """
        self._get_data_after_measurement = value
        self.notify_change('get_data_after_measurement')

    @property
    def is_saving(self):
        return self._is_saving

    @is_saving.setter
    def is_saving(self, value):
        self._is_saving = value
        self.notify_change('is_saving')

    @property
    def run_button_text(self):
        """
        Text of the run/stop button
        """
        if self.measurement_in_progress:
            return 'stop'
        else:
            return 'start'

    def __init__(self):
        super(TBS2104ViewModel, self).__init__()
        self._host = '129.132.1.157'
        self._connected = False
        self._auto_connect = False
        self._get_data_allowed = False
        self._status = 'disconnected'
        self._tbs = None
        self._x = list()
        self._y = list()
        self._selected_channel = -1
        self._get_data_after_measurement = True
        self._query_thread = None
        self.load_configuration()
        self.current_save_path = None
        self._is_saving = False
        self.widget = None

        if self.auto_connect:
            self.connect_to_device()

    def __del__(self):
        self.disconnect()

    def select_all(self):
        """
        Sets the end point to the last available point on the selected channel
        """
        self.stop = self.length

    def run_stop(self):
        """
        Start or stop a measurement
        """
        if self.measurement_in_progress:
            self._tbs.stop_acquisition()
        else:
            self._tbs.run()
            self._run_measurement_()
        self.notify_change('run_button_text')

    def single(self):
        """
        Start single measurement
        """
        self._tbs.single()
        self._run_measurement_()

    def single_and_measure(self):
        """
        Run single measurement and return data (used by experiments)
        """
        if not self.connected:
            return
        self._tbs.single()
        print('measurement in progress...')
        while self.measurement_in_progress:
            sleep(0.01)
        self.notify_change('run_button_text')
        self._get_data_worker_()
        return self.x, self.y

    def _run_measurement_(self):
        """
        Run measurement in separate thread
        """
        self._notify_all()
        measurement_thread = Thread(target=self._measurement_worker_)
        measurement_thread.start()

    def save_configuration(self):
        data = {'host': self.host, 'auto_connect': self.auto_connect}
        with open(self._TBS2104_config_file, 'w') as file:
            file.write(json.dumps(data))

    def load_configuration(self):
        if os.path.isfile(self._TBS2104_config_file):
            with open(self._TBS2104_config_file, 'r') as file:
                data = json.loads(file.read())
                self.host = data['host']
                self.auto_connect = data['auto_connect']

    def _measurement_worker_(self, *args):
        if not self.connected:
            return
        print('measurement in progress...')
        while self.measurement_in_progress:
            sleep(0.01)
        self.notify_change('run_button_text')

        print('measurement stopped.')
        if self.get_data_after_measurement:
            self.get_data()

    def _get_data_worker_(self, *args):
        print('retrieving data from TBS2104...')
        self.get_data_allowed = False
        try:
            self.x, self.y = self._tbs.get_data()
        except Exception as e:
            print(e)
        self.get_data_allowed = True
        print('data retrieved.')

    def get_data(self):
        """
        Get data asynchronously from device
        """
        self._query_thread = Thread(target=self._get_data_worker_)
        self._query_thread.start()

    def get_data_and_wait(self):
        self._get_data_worker_()
        return self.x, self.y

    def clear_data(self):
        """
        Clear plot
        """
        self.x = list()
        self.y = list()

    @pyqtSlot(float)
    def set_progress(self, value):
        self.widget.set_progress(value)

    @pyqtSlot(str, str)
    def message(self, message, title):
        self.widget.message(message, title)

    def save_data(self, path):
        """
        Save data to file
        :param path: path
        """
        self.current_save_path = path
        save_thread = Thread(target=self._save_worker_)
        save_thread.start()
        progress_thread = Thread(target=self._progress_worker_)
        progress_thread.start()

    def _save_worker_(self):
        self.is_saving = True
        data = np.vstack([self.x, self.y]).transpose()
        #np.savetxt(self.current_save_path, data, delimiter=';')

        if os.path.exists(self.current_save_path):
            os.remove(self.current_save_path)

        hf = h5py.File(self.current_save_path, 'w')
        hf.create_dataset('data', data=data)
        hf.close()

        """message = 'Saved file: {}'.format(self.current_save_path)
        if self.widget is not None:
            QMetaObject.invokeMethod(self, 'message', Q_ARG(str, message), Q_ARG(str, 'Operation completed'))
        else:
            print(message)"""
        self.is_saving = False

    compression_factor = 0.8125
    h5_conversion = 0.31374556862745095

    def _progress_worker_(self):
        if len(self.y) > 0:
            estimated_size = 2 * len(self.y) * sys.getsizeof(self.y[0]) * self.compression_factor * self.h5_conversion

            while self.is_saving:
                if os.path.exists(self.current_save_path):
                    try:
                        progress = os.path.getsize(self.current_save_path) / estimated_size * 100
                        QMetaObject.invokeMethod(self, 'set_progress', Q_ARG(float, progress))
                    except:
                        pass
            QMetaObject.invokeMethod(self, 'set_progress', Q_ARG(float, 100))

    def _notify_all(self):
        for name in ['start', 'stop', 'length', 'vertical_scale', 'vertical_units', 'selected_coupling', 'vertical_position', 'run_button_text', 'selected_trigger_mode']:
            try:
                self.notify_change(name)
            except:
                pass
            sleep(0.05)

    def connect_to_device(self):
        """
        Connect or disconnect device
        :return:
        """
        if not self.connected:
            self.connect()
        else:
            self.disconnect()

    def connect(self):
        """
        Connect to device
        """
        print('connecting...')
        self._tbs = TBS2104(self.host)

        try:
            self._tbs.open()
            self.connected = True
            self.get_data_allowed = True
            print(self._tbs.idn)
            self._notify_all()
        except Exception as e:
            self.status = 'connection failed'
            print(e)

    def disconnect(self):
        """
        Disconnect device
        """
        print('disconnecting...')
        if self._tbs is not None:
            self._tbs.close()
        self.get_data_allowed = False
        self.connected = False
