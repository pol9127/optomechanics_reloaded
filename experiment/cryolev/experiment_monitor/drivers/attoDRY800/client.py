#!/usr/bin/env python3
import time
import socket
import json

class attoDRYMonitorClient(object):
    HOST = '129.132.1.185'  # The server's hostname or IP address
    PORT = 65433        # The port used by the server
    sock = None
    
    def __del__(self):
        self.sock.close()
    def __init__(self, HOST = None, PORT = None):
        if HOST is not None:
            self.HOST = HOST
        if PORT is not None:
            self.PORT = PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.HOST, self.PORT))
    
    
    def ask_datapoint(self):
        self.sock.sendall(b'ASK_ATTO_DATAPOINT')
        msg = self.sock.recv(1024)
        if None:
            print('Connection to the attoDRY python server failed.')
            return None
            
        msg = msg.decode('utf-8')
        tmp = msg.split('DP_START')
        if len(tmp) == 2:
            tmp = tmp[1].split('DP_STOP')
            if len(tmp) == 2:
                return json.loads(tmp[0])
        print('Received message from the attoDRY pyhton server has unexpected format.')
        return None
                
        
if __name__ == '__main__':
    client = attoDRYMonitorClient()
    for i in range(10):
        time.sleep(1)
        print(client.ask_datapoint())