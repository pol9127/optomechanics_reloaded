# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 17:46:08 2014

@author: Erik Hebestreit
"""

from __future__ import division, unicode_literals

import threading
import time
from ctypes import c_uint32, c_int32

try:
    import queue
except ImportError:
    import Queue as queue

import numpy

from experiment.device.daq.ni7852r_support import fpga_binder


class NI7852R(object):
    session = c_uint32()
    status = c_int32()

    def __init__(self):
        pass

    def start(self):
        fpga_binder.start_fpga(self.session, self.status)
        return self.status

    def stop(self):
        fpga_binder.stop_fpga(self.session, self.status)

    @property
    def device_temperature(self):
        return fpga_binder.read_DeviceTemperature(self.session, self.status)


class FIFOAnalogInput(object):
    _fpga = None
    _block_size = None
    _acquisition_running = False

    thread = None

    def __init__(self, fpga, data_queue=None):
        self._fpga = fpga
        self.block_size = 2 ** 14

        self.time_reference = numpy.datetime64(int(time.time() * 1e6), 'us')

        if data_queue is None:
            self.data_queue = queue.Queue()
        else:
            self.data_queue = data_queue

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, value):
        self._block_size = value

        self.configure_fifo(value * 10)

    @property
    def block_time(self):
        return self.block_size * self.loop_ticks * 25e-9

    @property
    def loop_ticks(self):
        return fpga_binder.read_LoopTicks(self._fpga.session,
                                          self._fpga.status)

    @loop_ticks.setter
    def loop_ticks(self, value):
        fpga_binder.set_LoopTicks(value, self._fpga.session, self._fpga.status)

    @property
    def sampling_rate(self):
        return 1/(self.loop_ticks * 25e-9)

    @property
    def acquisition_running(self):
        # return self._acquisition_running
        return fpga_binder.AI_acquisition_active(self._fpga.session,
                                                 self._fpga.status)

    def start_fifo(self):
        fpga_binder.start_FIFO_AI(self._fpga.session, self._fpga.status)

        # start point of acquisition
        self.time_reference = numpy.datetime64(int(time.time() * 1e6), 'us')

    def stop_fifo(self):
        fpga_binder.stop_FIFO_AI(self._fpga.session, self._fpga.status)

    def configure_fifo(self, fifo_size):
        return fpga_binder.configure_FIFO_AI(fifo_size, self._fpga.session,
                                             self._fpga.status)

    def start_acquisition(self):
        if not self._acquisition_running:
            self._acquisition_running = True
            self.thread = threading.Thread(target=self.run)

            self.start_fifo()
            fpga_binder.toggle_AI_acquisition(True, self._fpga.session,
                                              self._fpga.status)
            self.thread.start()

    def stop_acquisition(self):
        if self._acquisition_running:
            self._acquisition_running = False
            fpga_binder.toggle_AI_acquisition(False, self._fpga.session,
                                              self._fpga.status)
            self.thread.join()
            self.stop_fifo()

    def read_fifo_block(self):
        ai0, ai1, ai2, times = fpga_binder.read_FIFO_conv(
            self.block_size, self._fpga.session, self._fpga.status)

        times = numpy.array(times * 1e9, dtype='timedelta64[ns]')

        total_block_time = times[-1]
        times = times + self.time_reference
        self.time_reference = self.time_reference + total_block_time

        return ai0, ai1, ai2, times

    def relaxation_cycle(self, ms_msrmt, ms_before_fb, ms_fb, co2_switch,
                         ms_before_co2, ms_co2, co2_on, co2_off):
        # type: (int, int, int, bool, int, int, float, float)

        co2_on = fpga_binder.voltage_to_int(co2_on)
        co2_off = fpga_binder.voltage_to_int(co2_off)

        ms_after_msrmt = int(self.block_size * 0.000025 * self.loop_ticks * 2)

        ai0, ai1, ai2, ticks = [], [], [], []

        fpga_binder.start_relaxation_measurement(
            ms_msrmt, ms_after_msrmt, ms_before_fb, ms_fb, co2_switch,
            ms_before_co2, ms_co2, co2_on, co2_off, self._fpga.session,
            self._fpga.status)

        # set remaining, such that while loop is executed at least once
        remaining = 1

        # read while acquisition is running or there is still data in FIFO
        while self.acquisition_running or remaining > 0:
            time.sleep(self.block_size * 25e-9 * self.loop_ticks / 2)

            [_ai0, _ai1, _ai2, _ticks, remaining] = \
                fpga_binder.read_FIFO_AI(
                    remaining, self._fpga.session, self._fpga.status)

            ai0.append(_ai0)
            ai1.append(_ai1)
            ai2.append(_ai2)
            ticks.append(_ticks)

        ai0 = numpy.concatenate(ai0)[2:]
        ai1 = numpy.concatenate(ai1)[2:]
        ai2 = numpy.concatenate(ai2)[2:]
        ticks = numpy.concatenate(ticks)[2:]

        times = numpy.cumsum(ticks) * 25e-9

        return ai0, ai1, ai2, times

    def timed_acquisition(self, ms_msrmt):
        # type: (int)
        ms_after_msrmt = int(self.block_size * 0.000025 * self.loop_ticks * 2)

        ai0, ai1, ai2, ticks = [], [], [], []

        fpga_binder.start_timed_measurement(
            ms_msrmt, ms_after_msrmt, self._fpga.session, self._fpga.status)

        # set remaining, such that while loop is executed at least once
        remaining = 1

        # read while acquisition is running or there is still data in FIFO
        while self.acquisition_running or remaining > 0:
            time.sleep(self.block_size * 25e-9 * self.loop_ticks / 2)

            [_ai0, _ai1, _ai2, _ticks, remaining] = \
                fpga_binder.read_FIFO_AI(
                    remaining, self._fpga.session, self._fpga.status)

            ai0.append(_ai0)
            ai1.append(_ai1)
            ai2.append(_ai2)
            ticks.append(_ticks)

        ai0 = numpy.concatenate(ai0)[2:]
        ai1 = numpy.concatenate(ai1)[2:]
        ai2 = numpy.concatenate(ai2)[2:]
        ticks = numpy.concatenate(ticks)[2:]

        times = numpy.cumsum(ticks) * 25e-9

        return ai0, ai1, ai2, times

    def run(self):
        # the first block is weird sometimes, drop it
        self.read_fifo_block()

        while self._acquisition_running:
            data = self.read_fifo_block()

            self.data_queue.put(data)


# TODO: FIFOAnalogOutput

class AnalogInput(object):
    _channel_number = None
    _fpga = None

    def __init__(self, channel, fpga):
        self._channel_number = channel
        self._fpga = fpga

    def read(self):
        return fpga_binder.int_to_voltage(
            getattr(fpga_binder, 'read_AI%0d' % self._channel_number)
            (self._fpga.session, self._fpga.status))


class AnalogOutput(object):
    _channel_number = None
    _fpga = None

    def __init__(self, channel, fpga):
        self._channel_number = channel
        self._fpga = fpga

    def write(self, value):
        return getattr(fpga_binder, 'set_AO%0d' % self._channel_number) \
            (fpga_binder.voltage_to_int(value), self._fpga.session,
             self._fpga.status)


class DigitalInput(object):
    _channel_number = None
    _fpga = None

    def __init__(self, channel, fpga):
        self._channel_number = channel
        self._fpga = fpga

    def read(self):
        return getattr(fpga_binder, 'read_DIO%0d' % self._channel_number) \
            (self._fpga.session, self._fpga.status)


class DigitalOutput(object):
    _channel_number = None
    _fpga = None

    def __init__(self, channel, fpga):
        self._channel_number = channel
        self._fpga = fpga

    def write(self, state):
        return getattr(fpga_binder, 'set_DIO%0d' % self._channel_number) \
            (state, self._fpga.session, self._fpga.status)