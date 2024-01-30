"""SCPI access to Red Pitaya.
This Class implements the remote connection to the red pitaya device
and wraps all available SCPI Commands in a convenient way."""

import socket

class scpi (object):
    """SCPI class used to access Red Pitaya over an IP network."""
    delimiter = '\r\n'
    decimation_factor = [1, 8, 64, 1024, 8192, 65536]
    averaging_status = {True: 'ON', False: 'OFF'}
    averaging_status_inverse = {value : key for key, value in averaging_status.items()}

    trigger_props = {'Delay': None,
                     'Delay in ns': None,
                     'Level': None}
    trigger_props_cmds = {'Delay': 'DLY',
                          'Delay in ns': 'DLY:NS',
                          'Level': 'LEV'}
    trigger_props_types = {'Delay': int,
                           'Delay in ns': float,
                           'Level': float}
    trigger_sources = ['DISABLED', 'NOW', 'CH1_PE', 'CH1_NE', 'CH2_PE', 'CH2_NE',
                       'EXT_PE', 'EXT_NE', 'AWG_PE', 'AWG_NE']

    def __init__(self, host, timeout=None, port=5000):
        """Initialize object and open IP connection.
        Host IP should be a string in parentheses, like '192.168.1.100'.
        """
        self.host    = host
        self.port    = port
        self.timeout = timeout

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if timeout is not None:
                self._socket.settimeout(timeout)
            self._socket.connect((host, port))
        except socket.error as e:
            print('SCPI >> connect({:s}:{:d}) failed: {:s}'.format(host, port, e))

    def __del__(self):
        if self._socket is not None:
            self._socket.close()
        self._socket = None

    def close(self):
        """Close IP connection."""
        self.__del__()

    def rx_txt(self, chunksize = 4096):
        """Receive text string and return it after removing the delimiter."""
        msg = ''
        while 1:
            chunk = self._socket.recv(chunksize + len(self.delimiter)).decode('utf-8') # Receive chunk size of 2^n preferably
            msg += chunk
            if (len(chunk) and chunk[-2:] == self.delimiter):
                break
        return msg[:-2]

    def rx_arb(self):
        numOfBytes = 0
        """ Recieve binary data from scpi server"""
        str=''
        while (len(str) != 1):
            str = (self._socket.recv(1))
        if not (str == '#'):
            return False
        str=''
        while (len(str) != 1):
            str = (self._socket.recv(1))
        numOfNumBytes = int(str)
        if not (numOfNumBytes > 0):
            return False
        str=''
        while (len(str) != numOfNumBytes):
            str += (self._socket.recv(1))
        numOfBytes = int(str)
        str=''
        while (len(str) != numOfBytes):
            str += (self._socket.recv(1))
        return str

    def tx_txt(self, msg):
        """Send text string ending and append delimiter."""
        return self._socket.send((msg + self.delimiter).encode('utf-8'))

    def txrx_txt(self, msg):
        """Send/receive text string."""
        self.tx_txt(msg)
        return self.rx_txt()

# IEEE Mandated Commands

    def cls(self):
        """Clear Status Command"""
        return self.tx_txt('*CLS')

    @property
    def ese(self):
        """Standard Event Status Enable Query"""
        return self.txrx_txt('*ESE?')

    @ese.setter
    def ese(self, value: int):
        """Standard Event Status Enable Command"""
        return self.tx_txt('*ESE {}'.format(value))

    @property
    def esr(self):
        """Standard Event Status Register Query"""
        return self.txrx_txt('*ESR?')

    @property
    def idn(self):
        """Identification Query"""
        return self.txrx_txt('*IDN?')

    def opc(self):
        """Operation Complete Command"""
        return self.tx_txt('*OPC')

    def opc_q(self):
        """Operation Complete Query"""
        return self.txrx_txt('*OPC?')

    def rst(self):
        """Reset Command"""
        return self.tx_txt('*RST')

    def sre(self):
        """Service Request Enable Command"""
        return self.tx_txt('*SRE')

    def sre_q(self):
        """Service Request Enable Query"""
        return self.txrx_txt('*SRE?')

    @property
    def stb(self):
        """Read Status Byte Query"""
        return self.txrx_txt('*STB?')

# :SYSTem

    def err_c(self):
        """Error count."""
        return self.txrx_txt('SYST:ERR:COUN?')

    def err_c(self):
        """Error next."""
        return self.txrx_txt('SYST:ERR:NEXT?')

# : ACQuire

    def acq_start(self):
        return self.tx_txt('ACQ:START')

    def acq_stop(self):
        return self.tx_txt('ACQ:STOP')

    def acq_rst(self):
        return self.tx_txt('ACQ:RST')

    @property
    def acq_dec(self):
        return int(self.txrx_txt('ACQ:DEC?'))

    @acq_dec.setter
    def acq_dec(self, decimation_factor):
        decimation_factor = int(decimation_factor)
        if decimation_factor in self.decimation_factor:
            return self.tx_txt('ACQ:DEC {0}'.format(int(decimation_factor)))
        else:
            print('Choose from one of the following decimation factors {0}'.format(self.decimation_factor))

    @property
    def acq_avg(self):
        averaging_status = self.txrx_txt('ACQ:AVG?')
        return self.averaging_status_inverse[averaging_status]

    @acq_avg.setter
    def acq_avg(self, averaging_status):
        averaging_status = bool(averaging_status)
        return self.tx_txt('ACQ:AVG {0}'.format(self.averaging_status[averaging_status]))

    @property
    def acq_trig(self):
        for trigger_prop in self.trigger_props:
            trigger_prop_val = self.txrx_txt('ACQ:TRIG:{0}?'.format(self.trigger_props_cmds[trigger_prop]))
            self.trigger_props[trigger_prop] = self.trigger_props_types[trigger_prop](trigger_prop_val)
        return self.trigger_props

    @acq_trig.setter
    def acq_trig(self, new_properties):
        for trigger_prop in new_properties:
            if trigger_prop in self.trigger_props:
                if trigger_prop == 'Level':
                    trigger_val = int(1e3 * new_properties[trigger_prop])
                    self.tx_txt('ACQ:TRIG:{0} {1} mV'.format(self.trigger_props_cmds[trigger_prop], trigger_val))
                else:
                    trigger_val = int(new_properties[trigger_prop])
                    self.tx_txt('ACQ:TRIG:{0} {1}'.format(self.trigger_props_cmds[trigger_prop], trigger_val))

    def acq_gain_set(self, ch, state):
        if state in ['LV', 'HV'] and ch in [1, 2]:
            self.tx_txt('ACQ:SOUR{0}:GAIN {1}'.format(ch, state))
            return 1
        else:
            print('Gain setting must be something of the following options: {0} and channel must be in {1}'.format(['LV', 'HV'], [1, 2]))
            return 0

    def acq_trig_src_set(self, source):
        if source not in self.trigger_sources:
            print('Trigger Source must be either of these: {0}'.format(self.trigger_sources))
        else:
            return self.tx_txt('ACQ:TRIG {0}'.format(source))
