'''
@ author: Andrei Militaru
@ email: andreimi@ethz.ch
@ date: 13th of January 2020

@ description:
    Python wrapper for the bitstreamfile: "MilitaruDAQ_FPGATarget_FirmwareForPytho_+lmkLI6ZzMk.lvbitx".
    It works for the National Instruments (NI) data acquisition card model NI7852R (with minor effor, it would
    probably work on other models of the same family as well, however it has not been tested so far).
    
    The main class of interest for the user is DaqManager. DaqManager takes full care of the interface with
    the FPGA on the NI board. The DaqManager is a context manager, meaning that the suggested use is within a 
    "with" block or with the "try ... finally" construct. Using the DaqManager in the suggested way ensures that
    the FPGA session does not remain open if a fatal exception occurs (which would prevent other programs to work with it).
    The internal attributes modify automatically the corresponding registers of the FPGA,
    allowing the user to choose what to do on which channel (both digital, analog, input and output).
    Additionally, it can stream analog data to the computer directly to the RAM of directly to a HDF5 dataset (depending
    on the preference of the user). The streaming works up to five channels at the same time. In case more channels are 
    needed, please write to "andreimi@ethz.ch" or "andrei.militaru@gmail.com".
    
    Here is an example script that reads three analog channels (2,3,4) to RAM and one (0) to HDF5 dataset, 
    controls two digital outputs (0, 1), reads
    from two digital inputs (2,3) and sets three analog outputs (0, 1, 2) to the respective values of (1 V, 2 V and -2 V).
    The script assumes that the bitstreamfile belongs to the same folder.
    
    
    
from apirio import volt_to_bit, DaqManager
from time import sleep

bitfile = 'MilitaruDAQ_FPGATarget_FirmwareForPytho_+lmkLI6ZzMk.lvbitx'
DIO = [True]*16 # sets the digital input outputs to inputs.
DIO[0] = False # sets DIO0 to output
DIO[1] = False # sets DIO1 to output
DO[0] = True # sets DO0 to True
DO[1] = False # sets DO1 to False, after 1 second it will be turned to True

AO = [0]*8
AO[0] = volt_to_bit(1) # AO0 set to 1 V
AO[1] = volt_to_bit(2) # AO1 set to 2 V
AO[2] = volt_to_bit(-2)  # AO2 set to -2 V

with DaqManager(bitfile=bitfile, 
                DIO=DIO, 
                AO=AO, 
                DO=DO,
                Xchannel=2,
                Ychannel=3,
                Zchannel=4,
                auxChannel=0
                DCchannel=7) as daq:
                
    with h5py.File('myfile.h5', mode='w') as output_file:
        dset = output_file.create_dataset('mydataset')
        daq.measure_three_channels(timelength=0.2) # stream for 0.2 s to computer
        daq.measure_one_channel(channel='aux', timelength=1, h5=True, datasets=dset) # stream aux channel for 1 s to the hdf5 file
        sleep(1)        
        daq.DO1 = True # set DO1 to True        
        daq.join_threads() # wait for the ChannelManager threads to complete the data transfer
    DI2_read_value = daq.DI2    # reading DI2
    DI3_read_value = daq.DI3    # reading DI3
    x = daq.measurements['measurement000'][0] # timetrace of AI2
    y = daq.measurements['measurement000'][1] # timetrace of AI3
    z = daq.measurements['measurement000'][2] # timetrace of AI4
'''

from nifpga import Session
from threading import Thread, Lock
from warnings import warn
from math import ceil

def extract_interleaving(block):
    '''
    Given a list of elements, it returns three lists
    made of alternated elements from the original one.
    For instance: extract_interleaving([1,2,3,4,5,6]) returns
    ([1,4], [2,5], [3, 6]).
    ---------------------
    Parameters:
        block: list
    --------------------
    Returns:
        tuple of three lists.
    -------------------
    '''
    L = len(block)
    x = [block[i] for i in range(0, L, 3)]
    y = [block[i] for i in range(1, L, 3)]
    z = [block[i] for i in range(2, L, 3)]
    return (x, y, z)

def bit_to_volt(n_bits):
    '''
    Conversion between bits and volts for NI7852r.
    ----------------
    Parameters:
        n_bits: int
    ---------------
    Returns:
        float: value in volts.
    --------------
    '''
    if n_bits >= 0:
        return n_bits/(2**15 - 1)*10
    else:
        return n_bits/2**15*10
        
def volt_to_bit(n_volts):
    '''
    Conversion between volts and bits for NI7852r.
    ----------------
    Parameters:
        n_volts: float,
            value in volts.
    ---------------
    Returns:
        int: value in bits.
    --------------
    '''
    if n_volts >= 0:
        return int(n_volts/10*(2**15 - 1))
    else:
        return int(n_volts/10*2**15)
        
def ticks_to_fs(ticks):
    '''
    Conversion between clock ticks (40 Mhz) and corresponding sampling frequencies.
    -----------------
    Parameters:
        ticks: int
            number of ticks corresponding to one sampling cycle
    ----------------
    Returns:
        float: corresponding sampling frequency
    ----------------
    '''
    return 4e7/ticks
    
def ticks_to_dt(ticks):
    '''
    Conversion between ticks and sampling period (at 40 Mhz).
    -----------------
    Parameters:
        ticks: int
            number of ticks corresponding to one sampling cycle
    ----------------
    Returns:
        float: sampling period in seconds.
    ----------------
    '''
    return ticks/4e7

def check_for_conflicts(*args):
    '''
    Given a list of elements, checks if any two of them are the same.
    -----------------
    Parameters:
        args: list
    ----------------
    Returns:
        bool: True if at least two elements are equal, False otherwise
    ----------------
    '''
    L = len(args)
    conflict = False
    for i in range(L-1):
        for j in range(i+1, L):
            if args[i] == args[j]:
                conflict = True
    return conflict
    
class ChannelManager(Thread):
    
    def __init__(self,
                 threadID,
                 fifo,
                 blocksize,
                 sampling,
                 timeout_alert,
                 timelength=-1,
                 timeout=-1,
                 lock=None,
                 result_container=None,
                 xcontainer=None,
                 ycontainer=None,
                 zcontainer=None,
                 warnings=True,
                 h5=False):
        '''
        Parent class for the SingleChannelManager and for the 
        TripleChannelManager, which take care of the streaming from the 
        FPGA to the computer.
        -----------------
        Parameters:
            threadID: str
                name given to this thread.
            fifo:
                an element from nifpga.Session(...).fifos.
            blocksize: int
                number of elements taken at every DMA cycle.
            sampling: float
                sampling frequency
            timeout_alert:
                reading nifpga.Session(...).registers. Indicates whether
                the fifo had a timeout.
            timelength: float, optional
                length of the desired timetrace in seconds.
                If negative, only one block is read from the fifo.
                Defaults to -1.
            lock: an instance of threading.Lock(), optional
                if h5 is False, it is ignored.
                Used to make sure that no conflict with other programs
                arises while saving the data. Defaults to None. Please
                note that if h5 is True, a lock must be provided.
            result_container: list or h5py dataset.
                a list where the results will be saved if h5 is False.
                a dataset where the data are streamed if h5 is True.
            xcontainer: same as result_container
            ycontainer: same as result_container
            zcontainer: same as result_container
            warnings: bool, optional
                if True, warnings arise if a timeout occurs. Defaults to True.
            h5: bool, optional
                if True, the data are saved to a file.
                If Falce, the data are added to the provided list.
                Defaults to False.
        ----------------
        '''                    
                
        Thread.__init__(self)
        self.threadID = threadID
        self.fifo = fifo
        self.blocksize = blocksize
        self.sampling = sampling
        self.timelength = timelength
        self.timeout = timeout
        self.timeout_alert = timeout_alert
        self.lock = lock
        self.result_container = result_container
        self.warnings = warnings
        self.h5 = h5
        self.xcontainer = xcontainer
        self.ycontainer = ycontainer
        self.zcontainer = zcontainer
    
    
class SingleChannelManager(ChannelManager):
    
    def __init__(self, *args, **kwargs):
        '''
        Thread that takes care of the data streaming from FPGA to 
        computer. It either reads from the aux-fifo or from the dc-fifo.
        -----------------
        Parameters:
            See ChannelManager.
        ----------------
        '''       
        super().__init__(*args, **kwargs)
        
    def run(self):

        result_container = self.result_container
        warnings = self.warnings
        h5 = self.h5
        
        timed_out = False
        Nelements = self.timelength*self.sampling
        if self.timelength > 0:
            Niterations = ceil(Nelements/self.blocksize)
        else:
            Niterations = 1
        self.fifo.start()
        for iteration in range(Niterations):
            previous_timeout = timed_out
            timed_out = timed_out or self.timeout_alert.read()
            block = self.fifo.read(self.blocksize, timeout_ms=self.timeout)
            if not h5:
                result_container += block.data
            else:
                if self.lock is None:
                    complain = 'When saving the data, a Lock needs to be provided to the ChannelManager.'
                    raise Exception(complain)
                else:
                    measured_data = block.data
                    with self.lock:
                        current_length = result_container.shape[0]
                        new_length = current_length + self.blocksize
                        result_container.resize((new_length,))
                        result_container[-self.blocksize:] = measured_data                    
            if warnings:
                if timed_out and not previous_timeout:
                    warn('Thread ' + self.threadID + ' timed out!')
        self.fifo.stop()
    
    
class TripleChannelManager(ChannelManager):
    
    def __init__(self, *args, **kwargs):
        '''
        Thread that takes care of the data streaming from the FPGA to 
        the computer. It reads from the data-fifo and extracts the three
        channels from the measured data. 
        
        WARNING: if the data_ticks is too low and the data are streamed
        directly to a HDF5 file it can happen that few data points are missing.
        While this is in principle not a problem, it messes with the extraction
        of the x, y and z traces from the data, which is a problem.
        If few seconds of timetrace are needed with less than 100 ticks for sampling,
        please set H5=False.
        -----------------
        Parameters:
            See ChannelManager.
        ----------------
        '''       
        super().__init__(*args, **kwargs)
        
    def run(self):
        
        xcontainer = self.xcontainer
        ycontainer = self.ycontainer
        zcontainer = self.zcontainer
        warnings = self.warnings
        h5 = self.h5        
        
        if h5:
            self.blocksize -= (self.blocksize % 3)
        
        timed_out = False
        Nelements = self.timelength*self.sampling
        if self.timelength > 0:
            Niterations = ceil(Nelements/self.blocksize)
        else:
            Niterations = 1
        self.fifo.start()
        measured_data = []
        for iteration in range(Niterations):
            previous_timeout = timed_out
            block = self.fifo.read(self.blocksize, timeout_ms=self.timeout)
            if warnings:
                if timed_out and not previous_timeout:
                    warn('Thread ' + self.threadID + ' timed out!')
            if h5:
                if self.lock is None:
                    complain = 'When saving the data, a Lock needs to be provided to the ChannelManager.'
                    raise Exception(complain)
                else:
                    (x_data, y_data, z_data) = extract_interleaving(block.data)
                    with self.lock:
                        current_length = xcontainer.shape[0]
                        new_length = current_length + len(x_data)
                        xcontainer.resize((new_length,))
                        xcontainer[-len(x_data):] = x_data
                        
                        current_length = ycontainer.shape[0]
                        new_length = current_length + len(y_data)
                        ycontainer.resize((new_length,))
                        ycontainer[-len(y_data):] = y_data
                        
                        current_length = zcontainer.shape[0]
                        new_length = current_length + len(z_data)
                        zcontainer.resize((new_length,))
                        zcontainer[-len(z_data):] = z_data
            else:
                measured_data += block.data
                
        if not h5:
            (x_data, y_data, z_data) = extract_interleaving(measured_data)
            xcontainer += x_data
            ycontainer += y_data
            zcontainer += z_data
        self.fifo.stop()
    

class IdGenerator():
    '''
    Class that provides strings used to label measurements if no other
    measurementID is provided to the DaqManager.
    '''
    def __init__(self, zeros=3):
        '''
        ------------------
        Parameters:
            zeros: int, optional
                Number of zeros used for the measurement ID.
                Defaults to 3, which generates the following IDs:
                    measurement000
                    measurement001
                    measurement002...
                    etc.
        ------------------
        '''
        self.counter = 0
        self.zeros = zeros
    def generate(self):
        generated = 'measurement' + str(self.counter).zfill(self.zeros)
        self.counter += 1
        return generated


class DaqManager():
    
    def __init__(self,
                 bitfile,
                 data_ticks=64,
                 data_elements=32000,
                 Xchannel=3,
                 Ychannel=4,
                 Zchannel=2,
                 DC_ticks=400,
                 DC_elements=1000,
                 DCchannel=6,
                 aux_ticks=80,
                 aux_elements=16000,
                 auxChannel=7,
                 AO_ticks=80,
                 DIO_ticks=50,
                 max_data_values=32767,
                 max_DC_values=1023,
                 max_aux_values=16383,
                 DIO=[True]*16,
                 DO=[False]*16,
                 AO=[0]*8,
                 zeros=3):
        '''
        -----------------
        Parameters:
            bitfile: str
                path and file name of the bitstreamfile:
                "MilitaruDAQ_FPGATarget_FirmwareForPytho_+lmkLI6ZzMk.lvbitx".
            data_ticks: int, optional
                clock ticks that make a sampling period. Defaults to 64, which
                is the minimum for the NI7852R.
            data_elements: int, optional
                number of data points extracted in a single block from the fifo.
            Xchannel: int, optional
                analog input channer corresponding to the X mode.
                Defaults to 3.
            Ychannel: int, optional
                analog input channel corresponding to the Y mode.
                Defaults to 4.
            Zchannel: int, optional
                analog input channel corresponding to the Z mode.
                Defaults to 2.
            DC_ticks: int, optional
                Sampling ticks of the DC channel. Defaults to 400.
            DCchannel: int, optional
                analog input channel corresponding to the DC channel.
                Defaults to 6.
            aux_ticks: int, optional
                Sampling ticks of the aux channel. Defaults to 80.
            DIO_ticks: int, optional
                Sampling ticks of the digital input output channels.
                Defaults to 50.
            max_data_values: int, optional
                Maximum numbers of elements that can be stored in the data-fifo.
                Defaults to 32767.
            max_DC_values: int, optional
                Maximum number of elements that can be stored in the DC-fifo.
                Defaults to 1023.
            max_aux_values: int, optional
                Maximum number of elements that can be stored in the aux-fifo.
                Defaults to 16383.
            DIO: list of 16 bool, optional
                list describing the function of corresponding digital input output channels.
                True means "input" and False means "output". Defaults to [True]*16.
            DO: list of 16 bool, optional
                values to give to the digital input output channels in case they are set to output,
                ignored otherwise. Defaults to [False]*16.
            AO: list of 8 int, optional
                values to give to the analog output channels.
                Defaults to [0]*8.
            zeros: int, optional
                number of zeros to fill the measurementID with.
                Defaults to 3.
        ----------------
        '''       
        
        self.isActive = False
        self.bitfile = bitfile
        self.session = Session(bitfile, "RIO0")
        self.session.reset()     
        self.max_data_values = max_data_values
        self.max_DC_values = max_DC_values
        self.max_aux_values = max_aux_values
        
        self.measurements = {}
        self.threads = []
        self.ID_generator = IdGenerator(zeros=zeros)
        
        channel_conflict = check_for_conflicts(Xchannel,
                                               Ychannel,
                                               Zchannel,
                                               DCchannel,
                                               auxChannel)
        if channel_conflict:
            raise Exception('Conflict between channels. Make sure no two channel read from the same ADC.')
        
        self.data_ticks = data_ticks
        self.data_elements = data_elements
        
        self._Xchannel = Xchannel
        self._Ychannel = Ychannel
        self._Zchannel = Zchannel
        self._auxChannel = auxChannel
        self._DCchannel = DCchannel
        self.Xchannel = Xchannel
        self.Ychannel = Ychannel
        self.Zchannel = Zchannel
        self.auxChannel = auxChannel
        self.DCchannel = DCchannel
        
        self.DC_ticks = DC_ticks
        self.DC_elements = DC_elements
        self.aux_ticks = aux_ticks
        self.aux_elements = aux_elements
        self.AO_ticks = AO_ticks
        self.DIO_ticks = DIO_ticks
        
        self.DIO0 = DIO[0]  # if True, it works as input
        self.DIO1 = DIO[1]
        self.DIO2 = DIO[2]
        self.DIO3 = DIO[3]
        self.DIO4 = DIO[4]
        self.DIO5 = DIO[5]
        self.DIO6 = DIO[6]
        self.DIO7 = DIO[7]
        self.DIO8 = DIO[8]
        self.DIO9 = DIO[9]
        self.DIO10 = DIO[10]
        self.DIO11 = DIO[11]
        self.DIO12 = DIO[12]
        self.DIO13 = DIO[13]
        self.DIO14 = DIO[14]
        self.DIO15 = DIO[15]
        
        self.DO0 = DO[0]
        self.DO1 = DO[1] 
        self.DO2 = DO[2] 
        self.DO3 = DO[3] 
        self.DO4 = DO[4] 
        self.DO5 = DO[5] 
        self.DO6 = DO[6] 
        self.DO7 = DO[7] 
        self.DO8 = DO[8] 
        self.DO9 = DO[9] 
        self.DO10 = DO[10] 
        self.DO11 = DO[11] 
        self.DO12 = DO[12] 
        self.DO13 = DO[13] 
        self.DO14 = DO[14]
        self.DO15 = DO[15] 
        
        self.AO0 = AO[0]
        self.AO1 = AO[1]
        self.AO2 = AO[2]
        self.AO3 = AO[3]
        self.AO4 = AO[4]
        self.AO5 = AO[5]
        self.AO6 = AO[6]
        self.AO7 = AO[7]
        
        self.timeouts = {'data timeout' : self.session.registers["Acquisition\nTimeout"],    # when sampling
                         'Xtimeout' : self.session.registers['X-timeout'],    # when saving to buffer
                         'Ytimeout' : self.session.registers['Y-timeout'],    # when saving to buffer
                         'Ztimeout' : self.session.registers['Z-tmeout'],    # when saving to buffer
                         'fifo timeout' : self.session.registers['FIFO-acquisition'], # when interleaving
                         'data fifo' : self.session.registers['data FIFO'],   # when saving to data fifo
                         'bufferX' : self.session.registers['FIFO X'],        # when reading from buffer
                         'bufferY' : self.session.registers['FIFO Y'],        # when reading from buffer
                         'bufferZ' : self.session.registers['FIFO Z-balanced'], # when reading from buffer
                         'DC acquisition' : self.session.registers['Acquisition\nTimeout-DC'], # sampling
                         'DC fifo' : self.session.registers['DC-FIFO-Timeout'], # when saving to fifo
                         'aux acquisition' : self.session.registers['Acquisition\nTimeout-Bonus1'], #sampling
                         'aux fifo' : self.session.registers['Bonus1-FIFO-Timeout'], # saving to fifo
                         'DIO' : self.session.registers['Timeout-DIO'],        # sampling
                         'AO' : self.session.registers['Timeout-AO']}         # sampling
    '''
    -----------------------------------
    Making DaqManager a context manager.
    This ensures that exceptions do not
    leave the session open.
    -----------------------------------
    '''
    
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close_session()
        self.session.__exit__(type, value, traceback)
        
    '''
    ----------------------------------
    End of context manager implementation.
    ----------------------------------
    '''
        
    def clear_measurements(self):
        if len(self.measurements.keys()) > 0:
            warn('Measured data points have been erased.')
        self.measurements = {}
        return self
    
    @property
    def DC_timeout(self):    
        return self.timeouts['DC acquisition'].read()
        
    @property
    def aux_timeout(self):
        return self.timeouts['aux acquisition'].read()            
    
    @property
    def data_timeout(self):
        return self.timeouts['data timeout'].read()
        
    @property
    def interleaving_timeout(self):
        return self.timeouts['fifo timeout'].read()
    
    @property
    def AO0(self):
        return self._AO0
        
    @AO0.setter
    def AO0(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO0 = new_value
            reg = self.session.registers['AO0']
            reg.write(self._AO0)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)
            
    @property
    def AO1(self):
        return self._AO1
        
    @AO1.setter
    def AO1(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO1 = new_value
            reg = self.session.registers['AO1']
            reg.write(self._AO1)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)            
            
    @property
    def AO2(self):
        return self._AO2
        
    @AO2.setter
    def AO2(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO2 = new_value
            reg = self.session.registers['AO2']
            reg.write(self._AO2)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)
            
    @property
    def AO3(self):
        return self._AO3
        
    @AO3.setter
    def AO3(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO3 = new_value
            reg = self.session.registers['AO3']
            reg.write(self._AO3)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution) 
            
    @property
    def AO4(self):
        return self._AO4
        
    @AO4.setter
    def AO4(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO4 = new_value
            reg = self.session.registers['AO4']
            reg.write(self._AO4)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)    
    
    @property
    def AO5(self):
        return self._AO5
        
    @AO5.setter
    def AO5(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO5 = new_value
            reg = self.session.registers['AO5']
            reg.write(self._AO5)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)
    
    @property
    def AO6(self):
        return self._AO6
        
    @AO6.setter
    def AO6(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO6 = new_value
            reg = self.session.registers['AO6']
            reg.write(self._AO6)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)
    
    @property
    def AO7(self):
        return self._AO7
        
    @AO7.setter
    def AO7(self, new_value):
        bit_min = -2**15
        bit_max = 2**15 - 1
        if bit_min <= new_value <= bit_max:
            self._AO7 = new_value
            reg = self.session.registers['AO7']
            reg.write(self._AO7)
        else:
            complain = 'Value requested is out of range.\n'
            solution = 'Allowed solutions are between {:d} and {:d}.'.format(bit_min, bit_max)
            raise Exception(complain + solution)    
    
    @property
    def DI0(self):
        reg = self.session.registers['DI0']
        return reg.read()
        
    @property
    def DI1(self):
        reg = self.session.registers['DI1']
        return reg.read()
    
    @property
    def DI2(self):
        reg = self.session.registers['DI2']
        return reg.read()    
    
    @property
    def DI3(self):
        reg = self.session.registers['DI3']
        return reg.read()    
    
    @property
    def DI4(self):
        reg = self.session.registers['DI4']
        return reg.read()
    
    @property
    def DI5(self):
        reg = self.session.registers['DI5']
        return reg.read()    
    
    @property
    def DI6(self):
        reg = self.session.registers['DI6']
        return reg.read()    
    
    @property
    def DI7(self):
        reg = self.session.registers['DI7']
        return reg.read()    
    
    @property
    def DI8(self):
        reg = self.session.registers['DI8']
        return reg.read()    
    
    @property
    def DI9(self):
        reg = self.session.registers['DI9']
        return reg.read()    
    
    @property
    def DI10(self):
        reg = self.session.registers['DI10']
        return reg.read()
    
    @property
    def DI11(self):
        reg = self.session.registers['DI11']
        return reg.read()
    
    @property
    def DI12(self):
        reg = self.session.registers['DI12']
        return reg.read()    
    
    @property
    def DI13(self):
        reg = self.session.registers['DI13']
        return reg.read()
    
    @property
    def DI14(self):
        reg = self.session.registers['DI14']
        return reg.read()    
    
    @property
    def DI15(self):
        reg = self.session.registers['DI15']
        return reg.read()
    
    @property
    def DO0(self):
        return self._DO0
        
    @DO0.setter
    def DO0(self, new_state):
        self._DO0 = new_state
        dio_register = self.session.registers['DO0']
        dio_register.write(new_state)    
        
    @property
    def DO1(self):
        return self._DO1
        
    @DO1.setter
    def DO1(self, new_state):
        self._DO1 = new_state
        dio_register = self.session.registers['DO1']
        dio_register.write(new_state)            
        
    @property
    def DO2(self):
        return self._DO2
        
    @DO2.setter
    def DO2(self, new_state):
        self._DO2 = new_state
        dio_register = self.session.registers['DO2']
        dio_register.write(new_state)        
    
    @property
    def DO3(self):
        return self._DO3
        
    @DO3.setter
    def DO3(self, new_state):
        self._DO3 = new_state
        dio_register = self.session.registers['DO3']
        dio_register.write(new_state)        
    
    @property
    def DO4(self):
        return self._DO4
        
    @DO4.setter
    def DO4(self, new_state):
        self._DO4 = new_state
        dio_register = self.session.registers['DO4']
        dio_register.write(new_state)        
    
    @property
    def DO5(self):
        return self._DO5
        
    @DO5.setter
    def DO5(self, new_state):
        self._DO5 = new_state
        dio_register = self.session.registers['DO5']
        dio_register.write(new_state)        
    
    @property
    def DO6(self):
        return self._DO6
        
    @DO6.setter
    def DO6(self, new_state):
        self._DO6 = new_state
        dio_register = self.session.registers['DO6']
        dio_register.write(new_state)        
    
    @property
    def DO7(self):
        return self._DO7
        
    @DO7.setter
    def DO7(self, new_state):
        self._DO7 = new_state
        dio_register = self.session.registers['DO7']
        dio_register.write(new_state)        
    
    @property
    def DO8(self):
        return self._DO8
        
    @DO8.setter
    def DO8(self, new_state):
        self._DO8 = new_state
        dio_register = self.session.registers['DO8']
        dio_register.write(new_state)    
    
    @property
    def DO9(self):
        return self._DO9
        
    @DO9.setter
    def DO9(self, new_state):
        self._DO9 = new_state
        dio_register = self.session.registers['DO9']
        dio_register.write(new_state)        
    
    @property
    def DO10(self):
        return self._DO10
        
    @DO10.setter
    def DO10(self, new_state):
        self._DO10 = new_state
        dio_register = self.session.registers['DO10']
        dio_register.write(new_state)        
    
    @property
    def DO11(self):
        return self._DO11
        
    @DO11.setter
    def DO11(self, new_state):
        self._DO11 = new_state
        dio_register = self.session.registers['DO11']
        dio_register.write(new_state)        
    
    @property
    def DO12(self):
        return self._DO12
        
    @DO12.setter
    def DO12(self, new_state):
        self._DO12 = new_state
        dio_register = self.session.registers['DO12']
        dio_register.write(new_state)        
    
    @property
    def DO13(self):
        return self._DO13
        
    @DO13.setter
    def DO13(self, new_state):
        self._DO13 = new_state
        dio_register = self.session.registers['DO13']
        dio_register.write(new_state)        
    
    @property
    def DO14(self):
        return self._DO14
        
    @DO14.setter
    def DO14(self, new_state):
        self._DO14 = new_state
        dio_register = self.session.registers['DO14']
        dio_register.write(new_state)        
    
    @property
    def DO15(self):
        return self._DO15
        
    @DO15.setter
    def DO15(self, new_state):
        self._DO15 = new_state
        dio_register = self.session.registers['DO15']
        dio_register.write(new_state)        
    
    @property
    def DIO0(self):
        return self._DIO0
        
    @DIO0.setter
    def DIO0(self, new_state):
        self._DIO0 = new_state
        dio_register = self.session.registers['DIO0-IN']
        dio_register.write(new_state)
        
    @property
    def DIO1(self):
        return self._DIO1
        
    @DIO1.setter
    def DIO1(self, new_state):
        self._DIO1 = new_state
        dio_register = self.session.registers['DIO1-IN']
        dio_register.write(new_state)
        
    @property
    def DIO2(self):
        return self._DIO2
        
    @DIO2.setter
    def DIO2(self, new_state):
        self._DIO2 = new_state
        dio_register = self.session.registers['DIO2-IN']
        dio_register.write(new_state)
        
    @property
    def DIO3(self):
        return self._DIO3
        
    @DIO3.setter
    def DIO3(self, new_state):
        self._DIO3 = new_state
        dio_register = self.session.registers['DIO3-IN']
        dio_register.write(new_state)
        
    @property
    def DIO4(self):
        return self._DIO4
        
    @DIO4.setter
    def DIO4(self, new_state):
        self._DIO4 = new_state
        dio_register = self.session.registers['DIO4-IN']
        dio_register.write(new_state)
    
    @property
    def DIO5(self):
        return self._DIO5
        
    @DIO5.setter
    def DIO5(self, new_state):
        self._DIO5 = new_state
        dio_register = self.session.registers['DIO5-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO6(self):
        return self._DIO6
        
    @DIO6.setter
    def DIO6(self, new_state):
        self._DIO6 = new_state
        dio_register = self.session.registers['DIO6-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO7(self):
        return self._DIO7
        
    @DIO7.setter
    def DIO7(self, new_state):
        self._DIO7 = new_state
        dio_register = self.session.registers['DIO7-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO8(self):
        return self._DIO8
        
    @DIO8.setter
    def DIO8(self, new_state):
        self._DIO8 = new_state
        dio_register = self.session.registers['DIO8-IN']
        dio_register.write(new_state)    
        
    @property
    def DIO9(self):
        return self._DIO9
        
    @DIO9.setter
    def DIO9(self, new_state):
        self._DIO9 = new_state
        dio_register = self.session.registers['DIO9-IN']
        dio_register.write(new_state)        
        
    @property
    def DIO10(self):
        return self._DIO10
        
    @DIO10.setter
    def DIO10(self, new_state):
        self._DIO10 = new_state
        dio_register = self.session.registers['DIO10-IN']
        dio_register.write(new_state)
        
    @property
    def DIO11(self):
        return self._DIO11
        
    @DIO11.setter
    def DIO11(self, new_state):
        self._DIO11 = new_state
        dio_register = self.session.registers['DIO11-IN']
        dio_register.write(new_state)        
        
    @property
    def DIO12(self):
        return self._DIO12
        
    @DIO12.setter
    def DIO12(self, new_state):
        self._DIO12 = new_state
        dio_register = self.session.registers['DIO12-IN']
        dio_register.write(new_state)        
    
    @property
    def DIO13(self):
        return self._DIO13
        
    @DIO13.setter
    def DIO13(self, new_state):
        self._DIO13 = new_state
        dio_register = self.session.registers['DIO13-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO14(self):
        return self._DIO14
        
    @DIO14.setter
    def DIO14(self, new_state):
        self._DIO14 = new_state
        dio_register = self.session.registers['DIO14-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO15(self):
        return self._DIO15
        
    @DIO15.setter
    def DIO15(self, new_state):
        self._DIO15 = new_state
        dio_register = self.session.registers['DIO15-IN']
        dio_register.write(new_state)    
    
    @property
    def DIO_ticks(self):
        return self._DIO_ticks
        
    @DIO_ticks.setter
    def DIO_ticks(self, ticks):
        self._DIO_ticks = ticks
        ticks_register = self.session.registers['Sampling-DIO']
        ticks_register.write(self._DIO_ticks)
    
    @property
    def AO_ticks(self):
        return self._AO_ticks
        
    @AO_ticks.setter
    def AO_ticks(self, ticks):
        self._AO_ticks = ticks
        ticks_register = self.session.registers['Sampling-AO']
        ticks_register.write(self._AO_ticks)
    
    @property
    def aux_ticks(self):
        return self._aux_ticks
        
    @aux_ticks.setter
    def aux_ticks(self, ticks):
        self._aux_ticks = ticks
        ticks_register = self.session.registers['Sampling-Bonus1']
        ticks_register.write(self._aux_ticks)
    
    @property
    def DC_ticks(self):
        return self._DC_ticks
        
    @DC_ticks.setter
    def DC_ticks(self, ticks):
        self._DC_ticks = ticks
        DC_register = self.session.registers['Sampling-DC']
        DC_register.write(self._DC_ticks)

    @property
    def auxChannel(self):
        return self._auxChannel
    
    @auxChannel.setter
    def auxChannel(self, new_channel):
        if not 0 <= new_channel <= 7:
            raise Exception('Analog channels can only be between 0 and 7.')
        elif (self.Xchannel == new_channel or self.Ychannel == new_channel) or (
              self.Zchannel == new_channel or self.DCchannel == new_channel):
            main_problem = 'Conflict with other channels.\n'
            detail = 'Currently: X: {:d}, Y: {:d}, Z: {:d}, DC: {:d}, aux: {:d}.\n'.format(self.Xchannel,
                                                                                           self.Ychannel,
                                                                                           self.Zchannel,
                                                                                           self.DCchannel,
                                                                                           self.auxChannel)
            raise Exception(main_problem + detail)
        else:
            self._auxChannel = new_channel
            aux_register = self.session.registers['Bonus1channel']
            aux_register.write(self.auxChannel)
        
    @property
    def DCchannel(self):
        return self._DCchannel
    
    @DCchannel.setter
    def DCchannel(self, new_channel):
        if not 0 <= new_channel <= 7:
            raise Exception('Analog channels can only be between 0 and 7.')
        elif (self.Xchannel == new_channel or self.Ychannel == new_channel) or (
              self.Zchannel == new_channel or self.auxChannel == new_channel):
            main_problem = 'Conflict with other channels.\n'
            detail = 'Currently: X: {:d}, Y: {:d}, Z: {:d}, DC: {:d}, aux: {:d}.\n'.format(self.Xchannel,
                                                                                           self.Ychannel,
                                                                                           self.Zchannel,
                                                                                           self.DCchannel,
                                                                                           self.auxChannel)
            raise Exception(main_problem + detail)
        else:
            self._DCchannel = new_channel
            DC_register = self.session.registers['DCchannel']
            DC_register.write(self.DCchannel)
    
    @property
    def Xchannel(self):
        return self._Xchannel
        
    @Xchannel.setter
    def Xchannel(self, new_channel):
        if not 0 <= new_channel <= 7:
            raise Exception('Analog channels can only be between 0 and 7.')
        elif (self.Ychannel == new_channel or self.Zchannel == new_channel) or (
              self.DCchannel == new_channel or self.auxChannel == new_channel):
            main_problem = 'Conflict with other channels.\n'
            detail = 'Currently: X: {:d}, Y: {:d}, Z: {:d}, DC: {:d}, aux: {:d}.\n'.format(self.Xchannel,
                                                                                           self.Ychannel,
                                                                                           self.Zchannel,
                                                                                           self.DCchannel,
                                                                                           self.auxChannel)
            raise Exception(main_problem + detail)
        else:
            self._Xchannel = new_channel
            x_register = self.session.registers['Xchannel']
            x_register.write(self.Xchannel)
            
    @property
    def Ychannel(self):
        return self._Ychannel
        
    @Ychannel.setter
    def Ychannel(self, new_channel):
        if not 0 <= new_channel <= 7:
            raise Exception('Analog channels can only be between 0 and 7.')
        elif (self.Xchannel == new_channel or self.Zchannel == new_channel) or (
              self.DCchannel == new_channel or self.auxChannel == new_channel):
            main_problem = 'Conflict with other channels.\n'
            detail = 'Currently: X: {:d}, Y: {:d}, Z: {:d}, DC: {:d}, aux: {:d}.\n'.format(self.Xchannel,
                                                                                           self.Ychannel,
                                                                                           self.Zchannel,
                                                                                           self.DCchannel,
                                                                                           self.auxChannel)
            raise Exception(main_problem + detail)
        else:
            self._Ychannel = new_channel
            y_register = self.session.registers['Ychannel']
            y_register.write(self.Ychannel)
            
    @property
    def Zchannel(self):
        return self._Zchannel
        
    @Zchannel.setter
    def Zchannel(self, new_channel):
        if not 0 <= new_channel <= 7:
            raise Exception('Analog channels can only be between 0 and 7.')
        elif (self.Xchannel == new_channel or self.Ychannel == new_channel) or (
              self.DCchannel == new_channel or self.auxChannel == new_channel):
            main_problem = 'Conflict with other channels.\n'
            detail = 'Currently: X: {:d}, Y: {:d}, Z: {:d}, DC: {:d}, aux: {:d}.\n'.format(self.Xchannel,
                                                                                           self.Ychannel,
                                                                                           self.Zchannel,
                                                                                           self.DCchannel,
                                                                                           self.auxChannel)
            raise Exception(main_problem + detail)
        else:
            self._Zchannel = new_channel
            z_register = self.session.registers['Zchannel']
            z_register.write(self._Zchannel)
    
    @property
    def data_ticks(self):
        return self._data_ticks
        
    @data_ticks.setter
    def data_ticks(self, ticks):
        self._data_ticks = ticks
        self._interleaved_ticks = int(self._data_ticks/3)
        sampling_register = self.session.registers['Sampling']
        sampling_register.write(self.data_ticks)
        interleaving_register = self.session.registers['Interleaving \nSampling']
        interleaving_register.write(self.interleaved_ticks)
    
    @property
    def interleaved_ticks(self):
        return self._interleaved_ticks
        
    @interleaved_ticks.setter
    def interleaved_ticks(self, ticks):
        raise Exception('Parameter fixed by self.data_ticks.')
        
    @property
    def data_elements(self):
        return self._data_elements
    
    @data_elements.setter
    def data_elements(self, new_value):
        if new_value > self.max_data_values:
            complain = 'Requested value is higher than threshold.\n'
            solution = 'data_elements will be set to ' + str(self.max_data_values) + '.'
            warn(complain + solution)
            self._data_elements = self.max_data_values
        else:
            self._data_elements = new_value
        
    @property
    def DC_elements(self):
        return self._DC_elements
        
    @DC_elements.setter
    def DC_elements(self, new_value):
        if new_value > self.max_DC_values:
            complain = 'Requested value is higher than threshold.\n'
            solution = 'DC_elements will be set to ' + str(self.max_DC_values) + '.'
            warn(complain + solution)
            self._DC_elements = self.max_DC_values
        else:
            self._DC_elements = new_value
            
    @property
    def aux_elements(self):
        return self._aux_elements
        
    @aux_elements.setter
    def aux_elements(self, new_value):
        if new_value > self.max_aux_values:
            complain = 'Requested value is higher than threshold.\n'
            solution = 'aux_elements will be set to ' + str(self.max_aux_values) + '.'
            warn(complain + solution)
            self._aux_elements = self.max_aux_values
        else:
            self._aux_elements = new_value
    
    def reset_session(self):  
        self.close_session()
        self.isActive = False
        self.session.reset()
        return self
    
    def run_session(self):
        if not self.isActive:
            self.session.run()
            self.isActive = True
        return self
        
    def close_session(self):
        if self.isActive:
            self.session.close()
            self.isActive = False
        return self
        
    def measure_three_channels(self, 
                               measurementID=None, 
                               timelength=0.1,
                               h5=False, 
                               datasets=[None]*3,
                               warnings=True):
        '''
        Method that calls a ChannelManager to read the timetrace with a thread.
        It measures data from that fifo that has three interleaved modes.
        ----------------
        Parameters:
            measurementID: str, optional.
                label to assign to the measured data. 
                If None, it is provided by the IdGenerator.
            timelength: float, optional
                length of the desired timetrace in seconds.
            h5: bool, optional
                if True, the measurement is streamed directly to the hdf5 datasets
                provided in the variable datasets.
                Defaults to False.
            datasets: list of h5py datasets
                ignored if h5 is False, used to store data otherwise.
            warnings: bool, optional
                If true, warnings are thrown if time-outs occur.
        -------------
        Returns:
            No return value. The measurement is however stored as a key
            in the self.measurements dictionary under the measurementID key.
            Also, the ChannelManager thread is added to self.threads.
        ---------------
        '''
        if measurementID is None:
            measurementID = self.ID_generator.generate()
        fifo = self.session.fifos['data-FIFO']
        timeout_alert = self.timeouts['fifo timeout']
        if not self.isActive:
            self.run_session()
        if not h5:
            self.measurements[measurementID] = ([], [], [])
            measurer = TripleChannelManager(measurementID,
                                            fifo,
                                            self.data_elements,
                                            ticks_to_fs(self.interleaved_ticks),
                                            timeout_alert,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=None,
                                            xcontainer=self.measurements[measurementID][0],
                                            ycontainer=self.measurements[measurementID][1],
                                            zcontainer=self.measurements[measurementID][2],
                                            warnings=warnings,
                                            h5=h5)
            measurer.start()
        else:
            lock = Lock()
            measurer = TripleChannelManager(measurementID,
                                            fifo,
                                            self.data_elements,
                                            ticks_to_fs(self.interleaved_ticks),
                                            timeout_alert,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=lock,
                                            xcontainer=datasets[0],
                                            ycontainer=datasets[1],
                                            zcontainer=datasets[2],
                                            warnings=warnings,
                                            h5=h5)
            measurer.start()
        self.threads.append(measurer)


    def measure_one_channel(self, 
                            measurementID=None, 
                            timelength=0.1,
                            h5=False, 
                            channel='aux',
                            datasets=None,
                            warnings=True):
        '''
        Method that calls a ChannelManager to read the timetrace with a thread.
        ----------------
        Parameters:
            measurementID: str, optional.
                label to assign to the measured data. 
                If None, it is provided by the IdGenerator.
            timelength: float, optional
                length of the desired timetrace in seconds.
            h5: bool, optional
                if True, the measurement is streamed directly to the hdf5 datasets
                provided in the variable datasets.
                Defaults to False.
            channel: str, optional
                if 'aux', the aux channel is read, if 'dc' the dc channel is read.
                Defaults to 'aux'.
            datasets: list of h5py datasets
                ignored if h5 is False, used to store data otherwise.
            warnings: bool, optional
                If true, warnings are thrown if time-outs occur.
        -------------
        Returns:
            No return value. The measurement is however stored as a key
            in the self.measurements dictionary under the measurementID key.
            Also, the ChannelManager thread is added to self.threads.
        ---------------
        '''
        if measurementID is None:
            measurementID = self.ID_generator.generate()
        if channel == 'aux':
            fifoname = 'Bonus1-FIFO'
            alertname = 'aux acquisition'
            elements = self.aux_elements
            ticks = self.aux_ticks
        elif channel == 'dc':
            fifoname = 'DC-FIFO'
            alertname = 'DC acquisition'
            elements = self.DC_elements
            ticks = self.DC_ticks
        else:
            raise Exception('Channel can only be ''aux'' or ''dc''.')
        if not self.isActive:
            self.run_session()
        fifo = self.session.fifos[fifoname]
        timeout_alert = self.timeouts[alertname]
        if not h5:
            self.measurements[measurementID] = []
            measurer = SingleChannelManager(measurementID,
                                            fifo,
                                            elements,
                                            ticks_to_fs(ticks),
                                            timeout_alert,
                                            result_container=self.measurements[measurementID],
                                            warnings=warnings,
                                            h5=h5,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=None)
            measurer.start()
        else:
            lock = Lock()
            measurer = SingleChannelManager(measurementID,
                                            fifo,
                                            elements,
                                            ticks_to_fs(ticks),
                                            timeout_alert,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=lock,
                                            result_container=datasets,
                                            warnings=warnings,
                                            h5=h5)
            measurer.start()
        self.threads.append(measurer)
            
    def measure_five_channels(self, 
                               measurementID=None, 
                               timelength=0.1,
                               h5=False, 
                               datasets=[None]*5,
                               warnings=True):
        '''
        Method that calls a ChannelManager to read the timetrace with a thread.
        It reads from all three fifos simultaneously, thus having the three 
        spatial modes, the dc channel and the aux channel.
        ----------------
        Parameters:
            measurementID: str, optional.
                label to assign to the measured data. 
                If None, it is provided by the IdGenerator.
            timelength: float, optional
                length of the desired timetrace in seconds.
            h5: bool, optional
                if True, the measurement is streamed directly to the hdf5 datasets
                provided in the variable datasets.
                Defaults to False.
            datasets: list of h5py datasets
                ignored if h5 is False, used to store data otherwise.
            warnings: bool, optional
                If true, warnings are thrown if time-outs occur.
        -------------
        Returns:
            No return value. The measurement is however stored as a key
            in the self.measurements dictionary under the measurementID key.
            Also, the ChannelManager thread is added to self.threads.
        ---------------
        '''
        if measurementID is None:
            measurementID = self.ID_generator.generate()
        data_fifo = self.session.fifos['data-FIFO']
        data_timeout_alert = self.timeouts['fifo timeout']
        dc_fifo = self.session.fifos['DC-FIFO']
        dc_timeout_alert = self.timeouts['DC acquisition']
        aux_fifo = self.session.fifos['Bonus1-FIFO']
        aux_timeout_alert = self.timeouts['aux acquisition']
        if not self.isActive:
            self.run_session()
        if not h5:
            self.measurements[measurementID] = ([], [], [], [], []) # x, y, z, aux, dc
            measurer = TripleChannelManager(measurementID + '_data',
                                            data_fifo,
                                            self.data_elements,
                                            ticks_to_fs(self.interleaved_ticks),
                                            data_timeout_alert,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=None,
                                            xcontainer=self.measurements[measurementID][0],
                                            ycontainer=self.measurements[measurementID][1],
                                            zcontainer=self.measurements[measurementID][2],
                                            warnings=warnings,
                                            h5=h5)
                                            
            measurer_aux = SingleChannelManager(measurementID + '_aux',
                                                aux_fifo,
                                                self.aux_elements,
                                                ticks_to_fs(self.aux_ticks),
                                                aux_timeout_alert,
                                                timelength=timelength,
                                                timeout=-1,
                                                lock=None,
                                                result_container=self.measurements[measurementID][3],
                                                warnings=warnings,
                                                h5=h5)      
                                                
            measurer_dc = SingleChannelManager(measurementID + '_dc',
                                               dc_fifo,
                                               self.DC_elements,
                                               ticks_to_fs(self.DC_ticks),
                                               dc_timeout_alert,
                                               timelength=timelength,
                                               timeout=-1,
                                               lock=None,
                                               result_container=self.measurements[measurementID][4],
                                               warnings=warnings,
                                               h5=h5)      
                                            
            measurer.start()
            measurer_dc.start() 
            measurer_aux.start()
        else:
            lock = Lock()
            measurer = TripleChannelManager(measurementID + '_data',
                                            data_fifo,
                                            self.data_elements,
                                            ticks_to_fs(self.interleaved_ticks),
                                            data_timeout_alert,
                                            timelength=timelength,
                                            timeout=-1,
                                            lock=lock,
                                            xcontainer=datasets[0],
                                            ycontainer=datasets[1],
                                            zcontainer=datasets[2],
                                            warnings=warnings,
                                            h5=h5)
                                            
            measurer_aux = SingleChannelManager(measurementID + '_aux',
                                                aux_fifo,
                                                self.aux_elements,
                                                ticks_to_fs(self.aux_ticks),
                                                aux_timeout_alert,
                                                timelength=timelength,
                                                timeout=-1,
                                                lock=lock,
                                                result_container=datasets[3],
                                                warnings=warnings,
                                                h5=h5)      
                                                
            measurer_dc = SingleChannelManager(measurementID + '_dc',
                                               dc_fifo,
                                               self.DC_elements,
                                               ticks_to_fs(self.DC_ticks),
                                               dc_timeout_alert,
                                               timelength=timelength,
                                               timeout=-1,
                                               lock=lock,
                                               result_container=datasets[4],
                                               warnings=warnings,
                                               h5=h5)      
                                            
            measurer.start()
            measurer_dc.start()
            measurer_aux.start()
            
        self.threads.append(measurer)
        self.threads.append(measurer_aux)
        self.threads.append(measurer_dc)
        
    def join_threads(self):
        if len(self.threads) == 0:
            return self
        else:
            for thread in self.threads:
                if thread.is_alive():
                    thread.join()
            return self