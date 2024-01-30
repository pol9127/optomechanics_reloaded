# author: Andrei Militaru
# date: 28th February 2022
"""description: API for the AttoDRY800 cryostat used in the cryotrapping experiment. I started with a
rudimentary API already used by the experiment_monitor on optomechanics and added additional functionalities."""

#!/usr/bin/env python3
import os
from ctypes import *
import time
import socket
import json
import sys
sys.path.append('C:\\')
import optomechanics 
from optomechanics.experiment.cryolev.experiment_monitor.drivers.Thyra import measure

def pressure():
    return measure.measure('COM9', verbose = False)


class AttoDryAPI(object):

    AttoDRY_Interface_Device_attoDRY800 = 2
    AttoDRY800_COMPORT = b"COM10"
    attoDLL = None
    
    def __init__(self):
        optomechanics_path = os.path.dirname(optomechanics.__file__)
        dll_path = os.path.join(optomechanics_path, 'experiment', 'cryolev', 'experiment_monitor', 'drivers', 'attoDRY800', 'DLL')
        self.attoDLL = CDLL(r'{:s}'.format(os.path.join(dll_path, 'attoDRYLib.dll')))

        ## First, try to connect
        res = self.attoDLL.AttoDRY_Interface_begin(c_uint16(self.AttoDRY_Interface_Device_attoDRY800))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        print("Interface Began.")
        res = self.attoDLL.AttoDRY_Interface_Connect(c_char_p(self.AttoDRY800_COMPORT))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        time.sleep(2) # for some reason we have to wait for 2 seconds here.
        print("USB connection set up.")
        
    def __del__(self):
        ## Let's disconnect
        res = self.attoDLL.AttoDRY_Interface_Disconnect()
        if res:
            print("Could not disconnect. Errorcode:", res)
            return
        print("Disconnected.")
        res = self.attoDLL.AttoDRY_Interface_end()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        print("Interface Ended.")
    
    def get4KStageTemperature(self):
        ## Let's ask for the temperature
        _4KStageTemperatureK = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_get4KStageTemperature(byref(_4KStageTemperatureK))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return _4KStageTemperatureK.value
    
    def getSampleTemperature(self):
        ## Let's ask for the temperature of the sample
        sampleTemperatureK = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_getSampleTemperature(byref(sampleTemperatureK))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return sampleTemperatureK.value
        
    def getPressure(self):
        ## Let's ask for the pressure
        CryostatInPressureMbar = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_getPressure800(byref(CryostatInPressureMbar))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return CryostatInPressureMbar.value
        
    def is_reaching_base_temperature(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isGoingToBaseTemperature(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_pumping(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isPumping(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_exchanging_sample(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isSampleExchangeInProgress(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_heater_on(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isSampleHeaterOn(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_sample_exchange_ready(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isSampleReadyToExchange(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def start_sample_exchange(self):
        res = self.attoDLL.AttoDRY_Interface_startSampleExchange()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Sample exchange procedure started.')
    
    def toggle_pump(self):
        res = self.attoDLL.AttoDRY_Interface_togglePump()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Pump toggled.')
            
    def outlet_pressure(self):
        answer = c_float()
        res = self.attoDLL.AttoDRY_Interface_getCryostatOutPressure(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return answer.value
        
    def heater_power(self):
        answer = c_float()
        res = self.attoDLL.AttoDRY_Interface_getReservoirHeaterPower(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return answer.value
        
    def helium_temperature(self):
        answer = c_float()
        res = self.attoDLL.AttoDRY_Interface_getReservoirTemperature(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return answer.value
        
    def is_exchange_heater_on(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_isExchangeHeaterOn(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_break_vacuum_valve_open(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_getBreakVac800Valve(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def toggle_sample_space_valve(self):
        res = self.attoDLL.AttoDRY_Interface_toggleSampleSpace800Valve()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Sample space valve toggled.')
            
    def is_pump_valve_open(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_getPump800Valve(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def is_sample_space_valve_open(self):
        answer = c_int()
        res = self.attoDLL.AttoDRY_Interface_getSampleSpace800Valve(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return False if answer.value == 0 else True
        
    def toggle_pump_valve(self):
        res = self.attoDLL.AttoDRY_Interface_togglePump800Valve()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Pump valve toggled.')
            
    def toggle_break_vacuum_valve(self):
        res = self.attoDLL.AttoDRY_Interface_toggleBreakVac800Valve()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Break vacuum valve toggled.')
            
    def turbo_pump_frequency(self):
        answer = c_uint16()
        res = self.attoDLL.AttoDRY_Interface_GetTurbopumpFrequ800(byref(answer))
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return answer.value
        
    def confirm(self):
        res = self.attoDLL.AttoDRY_Interface_Confirm()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Confirmed.')
            
    def cancel(self):
        res = self.attoDLL.AttoDRY_Interface_Cancel()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Canceled.')
            
    def action_message(self):
        answer = c_char_p()
        length = c_int(1000)
        res = self.attoDLL.AttoDRY_Interface_getActionMessage(byref(answer), length)
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        return answer.value

    def go_to_base_temperature(self):
        res = self.attoDLL.AttoDRY_Interface_goToBaseTemperature()
        if res:
            raise Exception("Could not get a reply. Errorcode:", res)
        else:
            print('Base temperature procedure started.')
            
    def open_break_vacuum_valve(self):
        if self.is_break_vacuum_valve_open():
            pass
        else:
            self.toggle_break_vacuum_valve()
        return self
        
    def open_pump_valve(self):
        if self.is_pump_valve_open():
            pass
        else:
            self.toggle_pump_valve()
        return self
        
    def close_break_vacuum_valve(self):
        if not self.is_break_vacuum_valve_open():
            pass
        else:
            self.toggle_break_vacuum_valve()
        return self
        
    def close_pump_valve(self):
        if not self.is_pump_valve_open():
            pass
        else:
            self.toggle_pump_valve()
        return self
        
    def start_pumping(self):
        if not self.is_pumping():
            self.toggle_pump()
        else:
            pass
        return self
        
    def stop_pumping(self):
        if self.is_pumping():
            self.toggle_pump()
        else:
            pass
        return self
    
    def start_cooling_down(self):
        if self.is_reaching_base_temperature():
            pass 
        else: 
            self.go_to_base_temperature()
    
    @property 
    def pump_valve_open(self):
        return self.is_pump_valve_open()
        
    @pump_valve_open.setter
    def pump_valve_open(self, new_state):
        if new_state:
            self.open_pump_valve()
        else:
            self.close_pump_valve()
    
    @property 
    def break_vacuum_valve_open(self):
        return self.is_break_vacuum_valve_open()
        
    @break_vacuum_valve_open.setter
    def break_vacuum_valve_open(self, new_state):
        if new_state:
            self.open_break_vacuum_valve()
        else:
            self.close_break_vacuum_valve()
            
    @property
    def pumping(self):
        return self.is_pumping()
        
    @pumping.setter
    def pumping(self, new_state):
        if new_state:
            self.start_pumping()
        else:
            self.stop_pumping()
        
if __name__ == '__main__':

    help_string = """Script options are:
    1) 'vent': turns off pump and starts venting
    2) 'characterization': stops venting, turns on pump, opens pump valve until pressure reaches 4.5mbar and then closes it 
    3) 'cooldown': starts reaching base temperature 
    4) 'exchange_sample': starts sample exchange
    5) 'cancel': equivalent to pressing cancel on the touch screen.
    6) 'confirm': equivalent to pressing confirm on the touch screen.
    7) 'action': supposed to get the action message shown on the touch screen. Unfortunately, it does not work so far."""
    
    script = sys.argv[1]
    
    if script == 'vent':
        cryostat = AttoDryAPI()
        cryostat.pump_valve_open = False
        cryostat.pumping = False 
        cryostat.break_vacuum_valve_open = True
        del cryostat
    
    elif script == 'help':
        print(help_string)
    
    elif script == 'cooldown':
        cryostat = AttoDryAPI()
        cryostat.start_cooling_down()
        del cryostat
        
    elif script == 'cancel':
        cryostat = AttoDryAPI()
        cryostat.cancel()
        del cryostat
        
    elif script == 'confirm':
        cryostat = AttoDryAPI()
        cryostat.confirm()
        del cryostat
        
    elif script == 'action':
        cryostat = AttoDryAPI()
        msg = cryostat.action_message()
        print(msg)
        del cryostat
        
    elif script == 'exchange_sample':
        cryostat = AttoDryAPI()
        cryostat.start_sample_exchange()
        del cryostat
        
    elif script == 'characterization':
        cryostat = AttoDryAPI()
        threshold = 4.5
        print('Getting down to {:.2f}mbar'.format(threshold))
        cryostat.break_vacuum_valve_open = False
        cryostat.pumping = True
        time.sleep(10)
        cryostat.pump_valve_open = True
        while pressure() >threshold:
            time.sleep(0.1)
        cryostat.pump_valve_open = False
        print('Pressure at {:.2f}'.format(pressure()))
        del cryostat
        
    else:
        pass
    
    print('Peace out.')
