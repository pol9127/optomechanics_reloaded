from __future__ import division, print_function, unicode_literals

from optomechanics.data import TimeSeries
from threading import Thread
from time import time, sleep

import numpy


class Observable(object):

    def __init__(self):
        self.observers = []

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def deregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def deregister_all(self):
        if self.observers:
            del self.observers[:]

    def update_observers(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)


class Observer(object):

    def update(self, *args, **kwargs):
        raise NotImplementedError


class Quantity(Observable):

    def __init__(self, data=None, **kwargs):
        super(Quantity, self).__init__()

        if data is None:
            data = []
        self.storage = TimeSeries(data, **kwargs)


class MeasuredQuantity(Quantity):
    _acquisition_running = False

    thread = None

    storage = None
    sources_list = []

    window = None

    def __init__(self, name, sources, acquisition_period=None, data=None,
                 **kwargs):
        self.name = name
        self.sources = sources
        self.sources_list = list(self.sources.keys())
        self.acquisition_period = acquisition_period

        if data is None:
            data = numpy.ndarray((0, len(self.sources_list)))

        super(MeasuredQuantity, self).__init__(data=data,
                                               time_step=acquisition_period,
                                               name=name, **kwargs)

        self.storage.labels = self.sources_list

    def start_acquisition(self):
        if not self._acquisition_running:
            self._acquisition_running = True
            self.thread = Thread(target=self.run)
            self.thread.start()
            self.storage.time_stamp = time()

    def stop_acquisition(self):
        if self._acquisition_running:
            self._acquisition_running = False
            self.thread.join()

    def restart_acquisition(self, delay=None):
        self.stop_acquisition()

        if delay is not None:
            sleep(delay)

        self.start_acquisition()

    def run(self):
        while self._acquisition_running:
            start_time = time()

            self.measure_value()

            time_passed = time() - start_time

            if time_passed < self.acquisition_period:
                sleep(self.acquisition_period - time_passed)

    def measure_value(self):
        result = dict()

        for name in self.sources_list:
            if self.sources[name]['active']:
                meas = getattr(self.sources[name]['source-object'],
                               self.sources[name]['source-attribute'])

                if meas > 0:
                    result[name] = meas
                else:
                    result[name] = numpy.nan
            else:
                result[name] = numpy.nan

        timestamp = numpy.datetime64(int(time() * 1e6), 'us')

        self._store_value(result, timestamp)

        self.update_observers(self.name, {'type': 'measurement',
                                          'value': result,
                                          'timestamp': timestamp})

        return result, timestamp

    def plot_window(self):
        """
        If used in interactive QT console, it might be necessary to run
        "%gui qt" to unhang the window.

        :return:
        """
        import pyqtgraph as pg

        self.window = pg.GraphicsWindow()
        self.window.setWindowTitle(self.name)

        self.plot = self.window.addPlot()
        # Use automatic downsampling and clipping to reduce the drawing load
        self.plot.setDownsampling(mode='peak')
        self.plot.setClipToView(True)
        self.plot.setRange(xRange=[-100, 0])
        self.plot.setLimits(xMax=0)
        self.plot.showGrid(x=True, y=True)
        self.plot.addLegend()
        self.plot.setLabel('left', self.name, units=self.storage.unit)

        self.curve = dict()
        for name in self.sources_list:
            self.curve[name] = self.plot.plot(
                    name=name, pen=(self.storage.get_column(name),
                                    self.storage.number_of_columns))

        self._update_plot()

    def _update_plot(self):
        for name in self.sources_list:
            self.curve[name].setData(self.storage.data_column(
                    self.storage.get_column(name)))
            self.curve[name].setPos(-self.storage.shape[0], 0)

    def _store_value(self, value, timestamp):
        data = dict()
        for name in self.sources_list:
            if hasattr(value[name], '__len__'):
                data[name] = value[name]
            else:
                data[name] = [value[name]]

        if not hasattr(timestamp, '__len__'):
            timestamp = [timestamp]

        self.storage.append_data_by_label(data)

        if self.window is not None and self.window.isVisible():
            self._update_plot()


class ProcessedQuantity(Quantity, Observer):

    def update(self, *args, **kwargs):
        self.post_processing()

    def post_processing(self, value, timestamp):
        raise NotImplementedError
    pass
