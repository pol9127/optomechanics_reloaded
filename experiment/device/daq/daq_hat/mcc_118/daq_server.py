'''This is the server module for the raspberry pi daq hat server-client system. It should be enabled as a service on
the raspberry pi. The daqhats module is provided by the manufacturer. It can be downloaded from github as mentioned
in the readme.

You should almost be able to use this file as it is on your raspberry pi. Only in the class "DAQServer" you have to
provide the server ip address (ip address of the raspberry pi).'''

from daqhats import mcc118, OptionFlags, HatIDs, HatError, hat_list
from time import sleep
import threading
import numpy as np
import socket
import json

class Hat:
    channels = None
    samples_per_channel = None
    requested_scan_rate = None
    address = None
    actual_scan_rate = None
    hat = None
    options = None
    channel_mask = None
    num_channels = None
    status = None
    timeout = 5.0
    maximum_scan_rate = 100000
    options_dict = {'DEFAULT': OptionFlags.DEFAULT,
                    'NOSCALEDATA': OptionFlags.NOSCALEDATA,
                    'NOCALIBRATEDATA': OptionFlags.NOCALIBRATEDATA,
                    'EXTCLOCK': OptionFlags.EXTCLOCK,
                    'EXTTRIGGER': OptionFlags.EXTTRIGGER,
                    'CONTINUOUS': OptionFlags.CONTINUOUS}
    
    def __init__(self, channels=[0], samples_per_channel=1000, requested_scan_rate=100000, options=['DEFAULT']):
        self.address = self.select_hat_device(HatIDs.MCC_118)
        self.mcc118 = mcc118(self.address)
        self.set_acquisition_parameters(channels, samples_per_channel, requested_scan_rate, options)


    def set_acquisition_parameters(self, channels=None, samples_per_channel=None, requested_scan_rate=None, options=None):
        if channels is not None:
            if not isinstance(channels, np.ndarray):
                channels = np.array(channels)
                channels = np.sort(channels[(channels >= 0) & (channels < 8)])
            self.channels = list(channels)
            self.channel_mask = self.chan_list_to_mask(self.channels)
            self.num_channels = len(self.channels)
            if requested_scan_rate is None:
                self.set_acquisition_parameters(requested_scan_rate=self.requested_scan_rate)
        if samples_per_channel is not None:
            self.samples_per_channel = samples_per_channel
        if requested_scan_rate is not None:
            n_channels = len(self.channels)
            max_scan_rate_per_channel = self.maximum_scan_rate // n_channels
            if requested_scan_rate > max_scan_rate_per_channel:                
                self.requested_scan_rate = max_scan_rate_per_channel
            else:
                self.requested_scan_rate = requested_scan_rate                
            self.actual_scan_rate = self.mcc118.a_in_scan_actual_rate(self.num_channels, self.requested_scan_rate)
        if options is not None:
            self.options = 0
            for opt in options:
                if opt in self.options_dict:
                    self.options |= self.options_dict[opt.strip()]

    def chan_list_to_mask(self, chan_list):
        chan_mask = 0

        for chan in chan_list:
            chan_mask |= 0x01 << chan

        return chan_mask
    
    def enum_mask_to_string(self, enum_type, bit_mask):
        item_names = []
        if bit_mask == 0:
            item_names.append('DEFAULT')
        for item in enum_type:
            if item & bit_mask:
                item_names.append(item.name)
        return ', '.join(item_names)
    
    def select_hat_device(self, filter_by_id):
        
        selected_hat_address = None

        # Get descriptors for all of the available HAT devices.
        hats = hat_list(filter_by_id=filter_by_id)
        number_of_hats = len(hats)

        # Verify at least one HAT device is detected.
        if number_of_hats < 1:
            raise HatError(0, 'Error: No HAT devices found')
        else:
            selected_hat_address = hats[0].address

        if selected_hat_address is None:
            raise ValueError('Error: Invalid HAT selection')

        return selected_hat_address

    def trigger(self, on_off=False, mode=0):
        '''This functions enables or disables triggering. If triggering is turned on the trigger mode should be provided. Possible choices are:
        RISING_EDGE = 0
        FALLING_EDGE = 1
        ACTIVE_HIGH = 2
        ACTIVE_LOW = 3'''
        options = self.enum_mask_to_string(OptionFlags, self.options)
        if on_off:
            self.mcc118.trigger_mode(mode)
            if not 'EXTTRIGGER' in options:
                self.options |= self.options_dict['EXTTRIGGER']
        else:
            if 'EXTTRIGGER' in options:
                self.options = 0
                for opt in options.split(','):
                    opt = opt.strip()
                    if 'EXTTRIGGER' not in opt:
                        self.options |= self.options_dict[opt]
                           
    def start_measurement(self):
        self.mcc118.a_in_scan_cleanup()
        self.mcc118.a_in_scan_start(self.channel_mask, self.samples_per_channel, self.requested_scan_rate, self.options)
    
    def read_data(self):
        data = self.mcc118.a_in_scan_read_numpy(-1, self.timeout).data
        self.mcc118.a_in_scan_cleanup()
        return data
    

class DAQServer:
    HOST = '129.132.1.131'
    PORT = 65432
    sock = None
    hat = None
    def __init__(self):
        self.hat = Hat()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen()
        while True:
            self.conn, addr = self.sock.accept()
            with self.conn:
                print('Connected by', addr)
                while True:
                    data_full = self.conn.recv(1024)
                    if not data_full:
                        break
                    else:
                        data_full = data_full.split(b'*|*')
                        if not data_full:
                            break
                        else:
                            for data in data_full[1:]:
                                data = data.split(b'|*|', 1)
                                if data[0] == b'get_acquisition_settings':
                                    self.conn.sendall(b'*|*' + b'get_acquisition_settings' + b'|*|' + self.get_acquisition_settings() + b'*|*')
                                elif data[0] == b'set_acquisition_settings':
                                    self.set_acquisition_settings(data[1])
                                elif data[0] == b'set_trigger':
                                    self.set_trigger(data[1])
                                elif data[0] == b'start_measurement':
                                    self.start_measurement()
                                elif data[0] == b'get_acquisition_status':
                                    self.conn.sendall(b'*|*' + b'get_acquisition_status' + b'|*|' + json.dumps(self.get_acquisition_status()).encode('utf-8') + b'*|*')
                                elif data[0] == b'get_measurement_data':
                                    self.conn.sendall(b'*|*' + b'get_measurement_data' + b'|*|' + self.get_measurement_data() + b'*|*')
    def __del__(self):
        self.sock.close()
                
    def get_acquisition_settings(self):
        settings = {'requested_scan_rate': self.hat.requested_scan_rate,
                    'actual_scan_rate': self.hat.actual_scan_rate,
                    'samples_per_channel': self.hat.samples_per_channel,
                    'options': self.hat.enum_mask_to_string(OptionFlags, self.hat.options).split(','),
                    'channels': self.hat.channels}
        return json.dumps(settings).encode('utf-8')
    
    def set_acquisition_settings(self, new_settings):
        new_settings = json.loads(new_settings.decode('utf-8'))
        self.hat.set_acquisition_parameters(**new_settings)
        
    
    def set_trigger(self, new_trigger):
        new_trigger = json.loads(new_trigger.decode('utf-8'))
        self.hat.trigger(**new_trigger)
        
    def start_measurement(self):
        self.hat.start_measurement()

    def get_acquisition_status(self):
        self.status = self.hat.mcc118.a_in_scan_status()
        status_dict = {'running': self.status.running,
                       'hardware_overrun': self.status.hardware_overrun,
                       'buffer_orverrun': self.status.buffer_overrun,
                       'triggered': self.status.triggered,
                       'samples_available': self.status.samples_available}
        return status_dict
    
    def get_measurement_data(self):
        data = self.hat.read_data().tobytes()
        return data
            
if __name__ == '__main__':
    daqserver = DAQServer()
