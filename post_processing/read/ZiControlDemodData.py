## Date 2018-03-11
## Author Felix Tebbenjohanns
## This file creates two classes that allow for easy reading of
## demodulated data as stored by the ZiControl program from Zurich Instruments.
## 2019-10-17: Added possibility to read from zipped folder in 'loadZiBinFile'.

import numpy as np
import os
import zipfile

def loadZiBinFile(filename, zip_folder = None):
    if zip_folder == None:
        fid = open(filename, 'rb')
        raw = np.fromfile(fid, dtype='>f8')
        fid.close()
    else:
        with zipfile.ZipFile(zip_folder, 'r') as myZip:
            with myZip.open(filename, 'r') as myfile: 
                raw = np.frombuffer(myfile.read(), dtype='>f8')
                
    assert len(raw) % 7 == 0, 'Illegal file size of ziBin file detected.'

    data = {}
    data['timestamp'] = raw[::7]
    data['x'] = raw[1::7]
    data['y'] = raw[2::7]
    data['frequency'] = raw[3::7]
    data['bits'] = raw[4::7].astype(np.uint32)
    data['auxin0'] = raw[5::7]
    data['auxin1'] = raw[6::7]
    
    return data


class ZiDemodChannel:
    def __init__(self, filename, dtype='bin'):
        self.name = filename
        print(filename)

        if (dtype == 'bin'):
            dat = np.fromfile(filename, dtype=np.dtype('float64').newbyteorder('>'))
            dat = dat.reshape((int(dat.shape[0] / 7), 7))
        else:
            ## Todo if needed!
            raise NotImplementedError

        # Extract timestamps and estimate sampling rate from it
        t = dat[:, 0]
        self.fs_est = t.shape[0] / (t[-1] - t[0])
        print('Estimated sampling rate', self.fs_est, 'Hz')

        # Extract demodulated signals X and Y, and demodulation frequency
        self.X = dat[:, 1]
        self.Y = dat[:, 2]
        self.Fdemod = dat[:, 3]


# simply create an instance of this class passing a folder into its constructor.
# It will look for all *.ziBin files in that folder and read them into its "channels" array

class ZiDemodData:
    def __init__(self, foldername, dtype='bin', zip_folder = None):
        self.name = foldername
        self.channelnames = []
        self.channels = []
        self.dtype = dtype

        for ziFile in np.extract([x.endswith('.ziBin') for x in os.listdir(self.name)], os.listdir(self.name)):
            self.channelnames.append(ziFile)
            self.channels.append(ZiDemodChannel(os.path.join(self.name, ziFile), self.dtype))


# Use like this:
# folder = r'.'
# dat = ZiDemodData(folder)
