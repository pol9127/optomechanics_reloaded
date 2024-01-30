#!/usr/bin/env python3

import socket
import attoDLLwrapper
import json
import sys

class attoDRYMonitorServer(object): 
    HOST = '129.132.1.185'
    PORT = 65433
    sock = None

    def __del__(self):
        self.sock.close()
        
    def __init__(self, fake = False):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.HOST, self.PORT))
        self.sock.listen()
        if fake:
            self.attoDLLwrapper = attoDLLwrapper.fakeAttoDLLwrapper()
        else:
            self.attoDLLwrapper = attoDLLwrapper.attoDLLwrapper()
        
    def start_server(self):
        while True:
            self.conn, addr = self.sock.accept()
            with self.conn:
                print('Connected by', addr)
                while True:
                    data_full = self.conn.recv(1024)
                    if not data_full:
                        break
                    if b'ASK_ATTO_DATAPOINT' in data_full:
                        _4Ktemp = self.attoDLLwrapper.get4KStageTemperature()
                        sampleTemp = self.attoDLLwrapper.getSampleTemperature()
                        pressure = self.attoDLLwrapper.getPressure()
                        tmp = {'4KstageTemp': _4Ktemp, 'pressure': pressure, 'SampleTemp': sampleTemp}
                        msg = b'DP_START' + json.dumps(tmp).encode('utf-8') + b'DP_STOP'
                        self.conn.sendall(msg)
                    else:
                        self.conn.sendall(b'Did not understand request')
                    
if __name__ == '__main__':
    if ('fakeatto' in sys.argv):
        print('OK, I will fake the attoDRY800.')
        fake = True
    else:
        fake = False
        
    s = attoDRYMonitorServer(fake = fake)
    s.start_server()