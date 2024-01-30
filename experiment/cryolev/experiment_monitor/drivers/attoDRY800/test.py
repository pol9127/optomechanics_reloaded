from ctypes import *
#import numpy as np
#import os
import time

AttoDRY_Interface_Device_attoDRY800 = 2
AttoDRY800_COMPORT = b"COM10"

def main():
    attoDLL = CDLL(r'DLL/attoDRYLib.dll')

    ## First, try to connect
    res = attoDLL.AttoDRY_Interface_begin(c_uint16(AttoDRY_Interface_Device_attoDRY800))
    if res:
        print("Could not begin interface. Errorcode:", res)
        return
    print("Interface Began.")
    res = attoDLL.AttoDRY_Interface_Connect(c_char_p(AttoDRY800_COMPORT))
    if res:
        print("Could not connect to USB. Errorcode:", res)
        return
    print("USB connection set up.")
    time.sleep(2) # for some reason we have to wait for 2 seconds here.
    
    ## Let's ask for the temperature
    _4KStageTemperatureK = c_float(0.0)
    res = attoDLL.AttoDRY_Interface_get4KStageTemperature(byref(_4KStageTemperatureK))
    if res:
        print("Could not get temperature. Errorcode:", res)
        return
    print("Temp (K):", _4KStageTemperatureK.value)
        
    ## Let's ask for the pressure
    CryostatInPressureMbar = c_float(0.0)
    res = attoDLL.AttoDRY_Interface_getPressure800(byref(CryostatInPressureMbar))
    if res:
        print("Could not get Pressure. Errorcode:", res)
        return
    print("Pressure (mBar):", CryostatInPressureMbar.value)
    
    
    
    
    
    ## Let's disconnect
    res = attoDLL.AttoDRY_Interface_Disconnect()
    if res:
        print("Could not disconnect. Errorcode:", res)
        return
    print("Disconnected.")
    res = attoDLL.AttoDRY_Interface_end()
    if res:
        print("Could not end interface. Errorcode:", res)
        return
    print("Interface Ended.")
    

main()