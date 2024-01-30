'''
Created on 08.10.2014

@author: Erik Hebestreit
'''

from __future__ import division, unicode_literals

from ctypes import *
from numpy import array, cumsum
from os.path import dirname, sep

_libfpga = CDLL(dirname(__file__)+sep+'libfpga_bundle.dll')

_libfpga.start_fpga.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.start_fpga.restype = None

_libfpga.stop_fpga.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.stop_fpga.restype = None

_libfpga.read_DeviceTemperature.argtypes = [POINTER(c_uint32),
                                            POINTER(c_int32)]
_libfpga.read_DeviceTemperature.restype = c_int16

_libfpga.read_LoopTicks.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_LoopTicks.restype = c_uint16

_libfpga.set_LoopTicks.argtypes = [c_uint16, POINTER(c_uint32),
                                   POINTER(c_int32)]
_libfpga.set_LoopTicks.restype = None

_libfpga.set_FifoTimeout.argtypes = [c_int32, POINTER(c_uint32),
                                     POINTER(c_int32)]
_libfpga.set_FifoTimeout.restype = None

_libfpga.configure_FIFO_AI.argtypes = [c_uint32, POINTER(c_uint32),
                                       POINTER(c_int32)]
_libfpga.configure_FIFO_AI.restype = c_uint32

_libfpga.start_FIFO_AI.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.start_FIFO_AI.restype = None

_libfpga.stop_FIFO_AI.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.stop_FIFO_AI.restype = None

_libfpga.toggle_AI_acquisition.argtypes = [c_bool, POINTER(c_uint32),
                                           POINTER(c_int32)]
_libfpga.toggle_AI_acquisition.restype = None

_libfpga.AI_acquisition_active.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.AI_acquisition_active.restype = c_bool

_libfpga.start_relaxation_measurement.argtypes = \
    [c_uint16, c_uint16, c_uint16, c_uint16, c_bool, c_uint16, c_uint16,
     c_int16, c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.start_relaxation_measurement.restype = None

_libfpga.start_timed_measurement.argtypes = \
    [c_uint16, c_uint16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.start_timed_measurement.restype = None

_libfpga.read_FIFO_AI.argtypes = [POINTER(c_uint64), c_int32,
                                  POINTER(c_uint32), POINTER(c_int32),
                                  POINTER(c_int32)]
_libfpga.read_FIFO_AI.restype = None

_libfpga.read_FIFO_AI_unpack.argtypes = [POINTER(c_int16), POINTER(c_int16),
                                         POINTER(c_int16), POINTER(c_uint16),
                                         c_int32, POINTER(c_uint32),
                                         POINTER(c_int32), POINTER(c_int32)]
_libfpga.read_FIFO_AI_unpack.restype = None

_libfpga.read_FIFO_AI_unpack_timeout.argtypes = \
    [POINTER(c_int16), POINTER(c_int16), POINTER(c_int16), POINTER(c_uint16),
     c_int32, c_uint32, POINTER(c_uint32), POINTER(c_int32), POINTER(c_int32)]
_libfpga.read_FIFO_AI_unpack_timeout.restype = None

_libfpga.read_AI3.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_AI3.restype = c_int16

_libfpga.read_AI4.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_AI4.restype = c_int16

_libfpga.read_AI5.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_AI5.restype = c_int16

_libfpga.read_AI6.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_AI6.restype = c_int16

_libfpga.read_AI7.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_AI7.restype = c_int16

_libfpga.set_AO0.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO0.restype = None

_libfpga.set_AO1.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO1.restype = None

_libfpga.set_AO2.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO2.restype = None

_libfpga.set_AO3.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO3.restype = None

_libfpga.set_AO4.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO4.restype = None

_libfpga.set_AO5.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO5.restype = None

_libfpga.set_AO6.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO6.restype = None

_libfpga.set_AO7.argtypes = [c_int16, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_AO7.restype = None

_libfpga.set_DIO0.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO0.restype = None

_libfpga.set_DIO1.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO1.restype = None

_libfpga.set_DIO2.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO2.restype = None

_libfpga.set_DIO3.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO3.restype = None

_libfpga.set_DIO4.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO4.restype = None

_libfpga.set_DIO5.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO5.restype = None

_libfpga.set_DIO6.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO6.restype = None

_libfpga.set_DIO7.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO7.restype = None

_libfpga.set_DIO8.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO8.restype = None

_libfpga.set_DIO9.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO9.restype = None

_libfpga.set_DIO10.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO10.restype = None

_libfpga.set_DIO11.argtypes = [c_bool, POINTER(c_uint32), POINTER(c_int32)]
_libfpga.set_DIO11.restype = None

_libfpga.read_DIO12.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_DIO12.restype = c_bool

_libfpga.read_DIO13.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_DIO13.restype = c_bool

_libfpga.read_DIO14.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_DIO14.restype = c_bool

_libfpga.read_DIO15.argtypes = [POINTER(c_uint32), POINTER(c_int32)]
_libfpga.read_DIO15.restype = c_bool

def start_fpga(session, status):
    return _libfpga.start_fpga(byref(session), byref(status))

def stop_fpga(session, status):
    return _libfpga.stop_fpga(byref(session), byref(status))

def read_DeviceTemperature(session, status):
    return _libfpga.read_DeviceTemperature(byref(session), byref(status))/4.0

def read_LoopTicks(session, status):
    return _libfpga.read_LoopTicks(byref(session), byref(status))

def set_LoopTicks(ticks, session, status):
    return _libfpga.set_LoopTicks(ticks, byref(session), byref(status))

def set_FifoTimeout(ticks, session, status):
    return _libfpga.set_FifoTimeout(ticks, byref(session), byref(status))

def configure_FIFO_AI(requestedDepth, session, status):
    return _libfpga.configure_FIFO_AI(requestedDepth, byref(session),
                                      byref(status))

def start_FIFO_AI(session, status):
    return _libfpga.start_FIFO_AI(byref(session), byref(status))

def stop_FIFO_AI(session, status):
    return _libfpga.stop_FIFO_AI(byref(session), byref(status))

def toggle_AI_acquisition(state, session, status):
    return _libfpga.toggle_AI_acquisition(state, byref(session), byref(status))

def AI_acquisition_active(session, status):
    return _libfpga.AI_acquisition_active(byref(session), byref(status))

def start_relaxation_measurement(ms_msrmt, ms_after_msrmt, ms_before_fb, ms_fb,
                                 co2_switch, ms_before_co2, ms_co2, co2_on,
                                 co2_off, session, status):
    return _libfpga.start_relaxation_measurement(
        ms_msrmt, ms_after_msrmt, ms_before_fb, ms_fb, co2_switch,
        ms_before_co2, ms_co2, co2_on, co2_off, byref(session), byref(status))

def start_timed_measurement(ms_msrmt, ms_after_msrmt, session, status):
    return _libfpga.start_timed_measurement(ms_msrmt, ms_after_msrmt,
                                            byref(session), byref(status))

def read_FIFO_AI(size, session, status):
    AI0 = (c_int16*size)()
    AI1 = (c_int16*size)()
    AI2 = (c_int16*size)()
    ticks = (c_uint16*size)() # ticks elapsed between (j-1)-st and j-th element
    elements_remaining = c_int32()
    _libfpga.read_FIFO_AI_unpack(AI0, AI1, AI2, ticks, size, byref(session),
                                 byref(status), byref(elements_remaining))
    return [AI0, AI1, AI2, ticks, elements_remaining.value]

def read_FIFO_AI_timeout(size, timeout, session, status):
    AI0 = (c_int16*size)()
    AI1 = (c_int16*size)()
    AI2 = (c_int16*size)()
    ticks = (c_uint16*size)() # ticks elapsed between (j-1)-st and j-th element
    elements_remaining = c_int32()
    _libfpga.read_FIFO_AI_unpack_timeout(
        AI0, AI1, AI2, ticks, size, timeout, byref(session), byref(status),
        byref(elements_remaining))
    return [AI0, AI1, AI2, ticks, elements_remaining.value]

def read_AI3(session, status):
    return _libfpga.read_AI3(byref(session), byref(status))

def read_AI4(session, status):
    return _libfpga.read_AI4(byref(session), byref(status))

def read_AI5(session, status):
    return _libfpga.read_AI5(byref(session), byref(status))

def read_AI6(session, status):
    return _libfpga.read_AI6(byref(session), byref(status))

def read_AI7(session,status):
    return _libfpga.read_AI7(byref(session), byref(status))

def set_AO0(value, session, status):
    return _libfpga.set_AO0(value, byref(session), byref(status))

def set_AO1(value, session, status):
    return _libfpga.set_AO1(value, byref(session), byref(status))

def set_AO2(value, session, status):
    return _libfpga.set_AO2(value, byref(session), byref(status))

def set_AO3(value, session, status):
    return _libfpga.set_AO3(value, byref(session), byref(status))

def set_AO4(value, session, status):
    return _libfpga.set_AO4(value, byref(session), byref(status))

def set_AO5(value, session, status):
    return _libfpga.set_AO5(value, byref(session), byref(status))

def set_AO6(value, session, status):
    return _libfpga.set_AO6(value, byref(session), byref(status))

def set_AO7(value, session, status):
    return _libfpga.set_AO7(value, byref(session), byref(status))

def set_DIO0(state, session, status):
    return _libfpga.set_DIO0(state, byref(session), byref(status))

def set_DIO1(state, session, status):
    return _libfpga.set_DIO1(state, byref(session), byref(status))

def set_DIO2(state, session, status):
    return _libfpga.set_DIO2(state, byref(session), byref(status))

def set_DIO3(state, session, status):
    return _libfpga.set_DIO3(state, byref(session), byref(status))

def set_DIO4(state, session, status):
    return _libfpga.set_DIO4(state, byref(session), byref(status))

def set_DIO5(state, session, status):
    return _libfpga.set_DIO5(state, byref(session), byref(status))

def set_DIO6(state, session, status):
    return _libfpga.set_DIO6(state, byref(session), byref(status))

def set_DIO7(state, session, status):
    return _libfpga.set_DIO7(state, byref(session), byref(status))

def set_DIO8(state, session, status):
    return _libfpga.set_DIO8(state, byref(session), byref(status))

def set_DIO9(state, session, status):
    return _libfpga.set_DIO9(state, byref(session), byref(status))

def set_DIO10(state, session, status):
    return _libfpga.set_DIO10(state, byref(session), byref(status))

def set_DIO11(state, session, status):
    return _libfpga.set_DIO11(state, byref(session), byref(status))

def read_DIO12(session, status):
    return _libfpga.read_DIO12(byref(session), byref(status))

def read_DIO13(session, status):
    return _libfpga.read_DIO13(byref(session), byref(status))

def read_DIO14(session, status):
    return _libfpga.read_DIO14(byref(session), byref(status))

def read_DIO15(session, status):
    return _libfpga.read_DIO15(byref(session), byref(status))
    
def read_FIFO_conv(size, session, status, ticks=56):
    """Reads a block of elements from the FPGA FIFO and determines the time
    array corresponding to them.
    """
    set_LoopTicks(ticks, session, status)
    
    [ai0, ai1, ai2, ticks, elements_remaining] = read_FIFO_AI(size, session,
                                                              status)
    
    if elements_remaining == size:
        print("Warning: FIFO full and elements might get lost.")

    ai0 = int_to_voltage(array(list(ai0)))
    ai1 = int_to_voltage(array(list(ai1)))
    ai2 = int_to_voltage(array(list(ai2)))

    times = cumsum(array(list(ticks))) * 25e-9
    
    return ai0, ai1, ai2, times

def int_to_voltage(integer):
    return (10*integer)/32767.

def voltage_to_int(voltage):
    # TODO: make it work for arrays and lists
    return int((voltage * 32767)/10)

def time_to_buffersize(time, ticks=56):
    return int(time / (ticks*0.000000025))

def buffersize_to_time(size, ticks=56):
    return size * (ticks*0.000000025)