from ctypes import *
import time

class fakeAttoDLLwrapper(object):    
    def __init__(self):
        print("Fake Interface Began.")
        time.sleep(2) # for some reason we have to wait for 2 seconds here.
        print("Fake USB connection set up.")
        
    def __del__(self):
        print("Disconnected.")
        print("Fake Interface Ended.")
    
    def get4KStageTemperature(self):
        time.sleep(0.01)
        return -96.7896
        
    def getSampleTemperature(self):
        time.sleep(0.01)
        return -11.2345
        
    def getPressure(self):
        time.sleep(0.01)
        return 1.3456e-19

class attoDLLwrapper(object):
    AttoDRY_Interface_Device_attoDRY800 = 2
    AttoDRY800_COMPORT = b"COM10"
    attoDLL = None
    
    def __init__(self):
        self.attoDLL = CDLL(r'DLL/attoDRYLib.dll')

        ## First, try to connect
        res = self.attoDLL.AttoDRY_Interface_begin(c_uint16(self.AttoDRY_Interface_Device_attoDRY800))
        if res:
            print("Could not begin interface. Errorcode:", res)
            return
        print("Interface Began.")
        res = self.attoDLL.AttoDRY_Interface_Connect(c_char_p(self.AttoDRY800_COMPORT))
        if res:
            print("Could not connect to USB. Errorcode:", res)
            return
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
            print("Could not end interface. Errorcode:", res)
            return
        print("Interface Ended.")
    
    def get4KStageTemperature(self):
        ## Let's ask for the temperature
        _4KStageTemperatureK = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_get4KStageTemperature(byref(_4KStageTemperatureK))
        if res:
            print("Could not get 4K stage temperature. Errorcode:", res)
            return None
        return _4KStageTemperatureK.value
    
    def getSampleTemperature(self):
        ## Let's ask for the temperature of the sample
        sampleTemperatureK = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_getSampleTemperature(byref(sampleTemperatureK))
        if res:
            print("Could not get Sample temperature. Errorcode:", res)
            return None
        return sampleTemperatureK.value
        
    def getPressure(self):
        ## Let's ask for the pressure
        CryostatInPressureMbar = c_float(0.0)
        res = self.attoDLL.AttoDRY_Interface_getPressure800(byref(CryostatInPressureMbar))
        if res:
            print("Could not get pressure. Errorcode:", res)
            return None
        return CryostatInPressureMbar.value