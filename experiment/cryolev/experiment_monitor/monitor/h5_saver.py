import numpy as np
import tables

class h5_saver(object):
    def __init__(self, columns, filename):
        # columns should be the captions of the columns of the datapoints as an array
        # eg: ['timestamp', Power LO (V)', 'Power trap (V)', 'Pressure attoDRY (mbar)']
        self.columns = columns
        self.filename = filename
        
        f = tables.open_file(filename, mode='w')
        f.create_earray(f.root, 'data', tables.Float64Atom(), (0, len(columns)))
        array_headers = f.create_earray(f.root, 'column_headers', tables.StringAtom(itemsize = 100), (0, len(columns)))
        array_headers.append(np.array([columns]))
        f.close()
        
    def save_datapoints(self, datapoints):
        # datapoints should be a 2D-np.array with the number of colums matching the size of self.columns
        assert len(datapoints.shape) == 2, "Dimensionality of \"datapoints\" wrong. Expected 2, received %i" % len(datapoints.shape)
        assert datapoints.shape[1] == len(self.columns), "Number of columns does not match initialization. Expected %i, received %i" % (len(self.columns), datapoints.shape[1])
        
        f = tables.open_file(self.filename, mode='a')
        f.root.data.append(datapoints)
        f.close()
        
        