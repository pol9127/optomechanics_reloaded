##
# Script used to take single measurement using NI6363 DAQ card.
# Enter parameters at top of script.
# DAQ cannot be used for any other process while this task is running.
##

import pandas as pd
import datetime as dt
import h5py
from NI6363 import CallbackTaskSynchronous

# Parameters to set by user
data_length = 3600
sample_rate = 30
channels = ['ai0']
channel_labels = ['X']
voltage_range=[0., 2.]
timeout = 1.5*data_length/sample_rate

path = 'C:\\Users\\Fons\\PycharmProjects\\optomechanics\\experiment\\device\\daq\\ni6363\\'
filename = "Test.hdf5"



# Create task for DAQ to measure
task = CallbackTaskSynchronous( data_len=2000,
                       sample_rate=20000,
                       channels=['ai0'],
                       channel_labels=['X'],
                       voltage_range=[-1., 1.])

# Start task and read out data
task.StartTask()
data = task.get_data()
task.StopTask()
task.ClearTask()

print(data)

# Save data to hdf5 file
#dataset = h5py.File(path + filename, 'w')

#Change data to pandas, to hdf for storing
#out_panda = pd.DataFrame(data = data.T, columns = channel_labels)
#out_panda.to_hdf(path + filename + '.hdf5', 'measurement')

#with pd.HDFStore(path + filename) as store:
#  store.put('measurement', out_panda)
#  store.get_storer('measurement').attrs.__setattr__('sample_rate',sample_rate)

