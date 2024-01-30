'''This is the client module for the raspberry pi daq hat server-client system.

You should almost be able to use this file as it is on your raspberry pi. Only in the class "DAQClient" you have to
provide the server ip address (ip address of the raspberry pi). You can run this script in an interactive ipython
terminal and run measurements from the command line.'''


import socket
import json
from time import sleep
import threading
import numpy as np


class DAQClient:
    '''This class represents a socket client that communicates with a socket server running on as raspberry pi. The
    server has access to a DAQ Hat and transfers data to this client.'''
    HOST = '129.132.1.131'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server
    acquisition_status = {}
    _measurement_finished = False
    def __init__(self, HOST=None, PORT=None):
        if HOST is not None:
            self.HOST = HOST
        if PORT is not None:
            self.PORT = PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))
        self.lock = threading.Lock()

    def get_acquisition_settings(self):
        '''The function queries the acquisition settings on the server. It should be pointed out that due to the
        internal clock the actual_scan_rate can deviate from the requested_scan_rate.'''
        self.sock.sendall(b'*|*' + b'get_acquisition_settings')
        data = self.recv_data()
        return json.loads(data.decode('utf-8'))

    def set_acquisition_settings(self,
                             channels=None,
                             requested_scan_rate=None,
                             options=None,
                             samples_per_channel=None):
        '''The function is used to change acquisition paramters on the server.
        The following input format is expected.
        channels: [0, 1, 2, ...]            List of integers
        requested_scan_rate: 1000           Integer
        options: ['EXTTRIGGER', 'EXTCLOCK']     List of options.
                                            Possible values are: DEFAULT, NOSCALEDATA, NOCALIBRATEDATE,
                                            EXTCLOCK, EXTTRIGGER, CONTINUOUS
        samples_per_channel: 1000           Integer'''

        new_settings = {'channels': channels,
                        'requested_scan_rate': requested_scan_rate,
                        'options': options,
                        'samples_per_channel': samples_per_channel}
        self.sock.sendall(b'*|*' + b'set_acquisition_settings' + b'|*|' + json.dumps(new_settings).encode('utf-8'))

    def set_trigger(self,
                    on_off=False,
                    mode=0):
        '''The function is used to enable/disable triggering. The mode sets the way the trigger works.
        The following modes are available:
        on_off: False                       Boolean
        mode: 0                             Rising Edge (0), Falling Edge (1), Active High (2), Active Low (3)'''
        new_trigger = {'on_off': on_off,
                       'mode': mode}

        self.sock.sendall(b'*|*' + b'set_trigger' + b'|*|' + json.dumps(new_trigger).encode('utf-8'))

    def start_measurement(self):
        '''The function starts a measurement on the server and starts a daemon that monitors the progress
        on the server.'''
        self.sock.sendall(b'*|*' + b'start_measurement')
        self.acquisition_status_daemon = threading.Thread(target=self.update_acquisition_status)
        self.acquisition_status_daemon.daemon = True
        self.acquisition_status_daemon.start()

    def recv_data(self):
        '''The function decomposes the input stream into separate messages, separates the command from the transmitted
        information and returns the information for further processing.'''
        data_full = self.recvall()
        if not data_full:
            return
        else:
            data_full = data_full.split(b'*|**|*')
            if not data_full:
                return
            else:
                for data in data_full:
#                    print("data_full: ", data_full)
#                    exchanged the order of replace and split
                    data = data.split(b'|*|')
                    data[-1] = data[-1].replace(b'*|*', b'')

                    if len(data) != 2:
                        print("something happend....")
#                        print("data: ", data)
#                        print("full: ", data_full)
                    return data[1]

    def get_acquisition_status(self):
        '''The function asks the server for the current acquisition status.'''
        self.sock.sendall(b'*|*' + b'get_acquisition_status')
        data = self.recv_data()
        return json.loads(data.decode('utf-8')) 

    def update_acquisition_status(self):
        '''After starting a measurement a daemon constantly updates the acquisition status to see when the measurement
        is finished.'''
        while True:
            with self.lock:
                self.acquisition_status = self.get_acquisition_status()
                if not self.acquisition_status['running'] and self.acquisition_status['samples_available'] > 0:
                    self._measurement_finished = True
                    break
            sleep(0.001)

    def get_measurement_data(self):
        '''This function transfers measurement data if a completed dataset is available'''
        with self.lock:
            if self._measurement_finished:
                self.sock.sendall(b'*|*' + b'get_measurement_data')
                data = self.recv_data()
#                print("datf: ", data)
                data = np.frombuffer(data, dtype=np.float64)
                acquisition_settings = self.get_acquisition_settings()
                n_channels = len(acquisition_settings['channels'])
                n_samples = acquisition_settings['samples_per_channel']
                self._measurement_finished = False
                return data.reshape(n_samples, n_channels).T
            else:
                print('Measurement did not complete yet.')

    @property
    def measurement_finished(self):
        '''This function returns the measurment status while protecting the variable from the background thread'''
        with self.lock:
            return self._measurement_finished

    def __del__(self):
        '''Close the socket upon closing the program.'''
        self.sock.close()

    def recvall(self):
        '''In general sent messages can have an arbitrary size. Messages are designed to start with *|* and end with *|*.
        This methods therefore streams data until the delimiters are found.'''
        BUFF_SIZE = 1024 # 4 KiB
        data = self.sock.recv(BUFF_SIZE)
        fix_counter = data.count(b'|*|*')
        delimiter_count = data.count(b'*|*')
        while (delimiter_count-fix_counter) % 2 != 0:
            part = self.sock.recv(BUFF_SIZE)
            data += part
            delimiter_count = data.count(b'*|*')
        return data

if __name__ == '__main__':
    client = DAQClient()
    #%%
    print(client.get_acquisition_settings())
    for n in range(200):
        client.start_measurement()
        while client.get_acquisition_status()['running']:
            sleep(0.0001)
        print(client.get_acquisition_settings())
        print(client.get_measurement_data())
        sleep(0.0001)
