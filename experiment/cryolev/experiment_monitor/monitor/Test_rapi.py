import sys
import time
import sched
from optomechanics.experiment.device.daq.daq_hat.mcc_118 import daq_client
import os
import numpy as np

class ExperimentMonitor(object):
    rpi_channel_header = [  # None means nothing is connected -> those are not stored.
        None,  # Ch 0
        None,  # Ch 1
        None,  # Ch 2
        None,  # Ch 3
        None,  # Ch 4
        None,  # Ch 5
        'LO power (V)',  # Ch 6
        'Trap power (V)']  # Ch 7
    rpi_filename = 'rpi_monitor.h5'

    def _assert_no_override(self, folder, filename):
        assert not os.path.exists(os.path.join(folder, filename)), 'The file ' + filename + \
            ' already exists in ' + folder + '. I refuse to override existing data.'
