## Date 2018-03-11
## Author Felix Tebbenjohanns
## Enables to read time traces taken with the DAQ and controlled from the LabView Program of Vijay.
## Edit 2018-10-19: number of channels not fixed.
## Edit 2019-10-08: can handle zipped files.

import numpy as np
import os
import zipfile


class VjChannel:
    def __init__(self):
        self.timetraces = []

    def append_timetrace(self, timetrace):
        self.timetraces.append(timetrace)


class VjDAQData:
    def __init__(self, info_filename, N_channels=8, zip_folder = None):
        self.name = info_filename
        
        if zip_folder == None:
            file = open(info_filename, 'r').read()
        else:
            with zipfile.ZipFile(zip_folder, 'r') as myZip:
                zipped_listdir = [os.path.basename(x) for x in myZip.namelist()]
                with myZip.open(info_filename) as myfile:
                    file = myfile.read().decode('utf-8')
            
        self.Fs = int(round(float(file.split('sample rate (S/s)')[1].split('<Val>')[1].split('</Val>')[0])))
        
        if '<Name># Channels</Name>' in file:
            N_channels = int(file.split('<Name># Channels</Name>')[1].split('<Val>')[1].split('</Val>')[0])

        
        key = os.path.basename(info_filename)[:-5]

        if zip_folder == None:
            tmp = [x for x in os.listdir(os.path.dirname(info_filename)) if (key in x and x.endswith('.dat'))]
        else:
            tmp = [x for x in zipped_listdir if (key in x and x.endswith('.dat'))]
        
        file_prefix = os.path.join(os.path.dirname(info_filename), tmp[0].split('_')[0] + '_' + key + '_')

        self.N = len(tmp)

        self.channels = []
        for i in range(N_channels):
            self.channels.append(VjChannel())

        for i in range(self.N):
            filename = file_prefix + str(i).zfill(3) + '.dat'
            if zip_folder == None:
                file = open(filename, "rb")
                tmp = np.fromfile(file, np.int16)[4:]  # first 4 numbers (8 bytes) are overhead (no data)
            else:
                with zipfile.ZipFile(zip_folder, 'r') as myZip:
                    with myZip.open(filename, "r") as myfile:
                        tmp = np.frombuffer(myfile.read(), np.int16)[4:]  # first 4 numbers (8 bytes) are overhead (no data)
                
            tmp = tmp.reshape(N_channels, int(len(tmp) / N_channels))

            for ch in range(N_channels):
                self.channels[ch].append_timetrace(tmp[ch, :])

        self.L = len(self.channels[0].timetraces[0])

