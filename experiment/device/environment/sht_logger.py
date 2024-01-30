import serial as sr
import re
from datetime import datetime as dt
from datetime import timedelta as td
from time import sleep
from optomechanics.database.mongodb import MongoSisyphous
import matplotlib.pyplot as plt
import numpy as np

class SHT3x(object):
    handler = None
    def __init__(self, port):
        self.handler = sr.Serial(port, 9600)

    @property
    def rht(self):
        rht_ = {'rh' : None, 't' : None, 'timestamp' : None}
        try:
            self.handler.flushInput()
        except:
            self.handler.reset_input_buffer()
        while rht_['rh'] is None or rht_['t'] is None:
            line = self.handler.readline()
            if b'RH' in line:
                vals = re.findall(b"[-+]?\d*\.\d+|\d+", line)
                if vals != []:
                    rht_['rh'] = float(vals[0])
            elif b'T' in line:
                vals = re.findall(b"[-+]?\d*\.\d+|\d+", line)
                if vals != []:
                    rht_['t'] = float(vals[0])
        rht_['timestamp'] = dt.now()
        return rht_

    def log_rht(self, interval=1, function=None):
        while(True):
            if function is not None:
                function(self.rht)
            else:
                print(self.rht)
            sleep(interval)

class SHT3xDBLogger(MongoSisyphous):
    sensor_name = None
    collection = None
    def __init__(self, sensor_name, collection,**kwargs):
        super(SHT3xDBLogger, self).__init__(**kwargs)
        self.sensor_name = sensor_name
        self.collection = collection
    def _log_function(self, rht):
        self.database.get_collection(self.collection).update({'_id': self.sensor_name},
                                                            {'$push': {'data': rht}}, upsert=True)


    def log_to_database(self, sht, interval):
        sht.log_rht(interval=interval, function=self._log_function)

    def read_from_database(self, timestamp0, timestamp1):
        data = list(self.database.get_collection(self.collection).aggregate([
            { '$match': {'_id': self.sensor_name}},
            { '$project': {
                'data': {'$filter': {
                    'input': '$data',
                    'as': 'data',
                    'cond': {
                        '$and': [
                            { '$gte': [ "$$data.timestamp", timestamp0 ] },
                            { '$lte': [ "$$data.timestamp", timestamp1 ] }
                        ]}
                }}
            }}
        ]))[0]['data']
        return data

    def plot_data(self, timestamp0, timestamp1):
        data = self.read_from_database(timestamp0, timestamp1)
        rhs = np.array([d['rh'] for d in data])
        ts = np.array([d['t'] for d in data])
        runtime = np.array([d['timestamp'] for d in data])
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(runtime, rhs, color='blue')
        ax2.plot(runtime, ts, color='red')
        ax1.set_ylabel(r'rel. humidity in [%rh]')
        ax2.set_ylabel(r'temperature in [degC]')
        ax1.set_xlabel(r'runtime')
        ax1.yaxis.label.set_color('blue')
        ax2.yaxis.label.set_color('red')
        ax1.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    sht = SHT3x('COM4')
    print(sht.rht['t'])
    print(sht.rht['t'])
    print(sht.rht['t'])
    print(sht.rht['t'])
    # sht_logger = SHT3xDBLogger('dominiksPC', 'environment', username='cavitydata', password='micro_cavity', database='cavitydata')
    # sht_logger.log_to_database(sht, 10)
    # data = sht_logger.read_from_database(dt.now() - td(minutes=10), dt.now())
    # sht_logger.plot_data(dt.now() - td(hours=10), dt.now())