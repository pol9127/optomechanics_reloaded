from datetime import datetime as dt
from datetime import timedelta as td
from time import sleep
from optomechanics.database.mongodb import MongoSisyphous
import matplotlib.pyplot as plt
import numpy as np

class ThyracontDBLogger(MongoSisyphous):
    sensor_name = None
    collection = None
    def __init__(self, sensor_name, collection,**kwargs):
        super(ThyracontDBLogger, self).__init__(**kwargs)
        self.sensor_name = sensor_name
        self.collection = collection
    def _log_function(self, pressure):
        self.database.get_collection(self.collection).update({'_id': self.sensor_name},
                                                            {'$push': {'data': pressure}}, upsert=True)


    def log_to_database(self, gauge, addresses, interval):
        while True:
            pressure = {}
            for ad in addresses:
                gauge.address = ad
                pressure[str(ad)] = gauge.pressure
            pressure['timestamp'] = dt.now()
            self._log_function(pressure)
            sleep(interval)

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
        addresses = list(data[0].keys())
        addresses.remove('timestamp')

        pressure = [np.array([d[ad] for d in data]) for ad in addresses]
        runtime = np.array([d['timestamp'] for d in data])
        fig, ax1 = plt.subplots()
        for pres, ad in zip(pressure, addresses):
            ax1.plot(runtime, pres, label=ad)
        ax1.set_ylabel(r'pressure in [mbar]')
        ax1.set_xlabel(r'runtime')
        ax1.grid(True)
        ax1.legend(loc='best')
        plt.tight_layout()
        plt.show()
