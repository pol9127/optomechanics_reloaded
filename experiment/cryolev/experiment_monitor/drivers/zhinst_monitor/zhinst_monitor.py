import numpy as np
import zhinst.ziPython
import time

class zhinst_monitor(object):
    daq_channels = {}
    scopes = {}
    
    def __init__(self, data_server_ip):        
        self.daq = zhinst.ziPython.ziDAQServer(data_server_ip, 8004, 5)
    
    def getAuxIns(self, device):
        vals = []
        for i in range(2):
            vals.append(self.daq.getDouble('/%s/auxins/0/values/%i' % (device, i)))
           
        return np.array(vals)
    
    
    def addDAQmod(self, name, device, N, demodno, nodes):
        if name not in self.daq_channels:
            self.daq_channels[name] = {}
            self.daq_channels[name]['daq_module'] = self.daq.dataAcquisitionModule()
        dc = self.daq_channels[name]
        dc['daq_module'].finish()
        dc['daq_module'].unsubscribe('*')
        dc['daq_module'].set('device', device)
        dc['daq_module'].set('grid/cols', N)
        dc['daq_module'].set('type', 0)
        dc['clockbase'] = self.daq.getDouble('/%s/clockbase' % device)
        dc['device'] = device
        dc['demodno'] = str(demodno)
        dc['nodes'] = nodes
        
        for node in nodes:
            dc['daq_module'].subscribe('/%s/demods/%s/sample.%s' % (device, demodno, node))

    def getDemodSamples(self, name):        
        # note that calling this takes surprinsingly long: about 0.3s overhead!
        if name not in self.daq_channels:
            return None
        
        dc = self.daq_channels[name]
        dc['daq_module'].execute()
        while not dc['daq_module'].finished():
            time.sleep(1e-3)
        result = dc['daq_module'].read()
        
        device = dc['device']
        demodno = dc['demodno']
        nodes = dc['nodes']
        
        t = result[device]['demods'][demodno]['sample.%s' % nodes[0]][0]['timestamp'][0,:]
        t = t- t[0]
        t = t/ dc['clockbase']
        
        res = []
        for node in nodes:
            res.append(result[device]['demods'][demodno]['sample.%s' % node][0]['value'][0,:])
        return (t, res)
    
    def addScopeChannels(self, name, device, avg_weight, no_samples, inputselect1, inputselect2 = None):
        #inputselect: 0 = Signal Input, 1 = Current Input, 8 = Aux Input 1, 9 = Aux Input 2, for other see MFLI user manual, p. 368
        #returns: (length of data, df (Hz))
        if name not in self.scopes:
            self.scopes[name] = {}
            self.scopes[name]['scope_mod'] = self.daq.scopeModule()
        
        sc = self.scopes[name]
        sc['device'] = device
        sc['no_samples'] = no_samples
        sc['inputselect1'] = inputselect1
        sc['inputselect2'] = inputselect2
        # enable both scope channels. For a MFLI without the DIG option, this will only enable one channel
        self.daq.setInt('/%s/scopes/0/channel' % device, 3)
        self.daq.setInt('/%s/scopes/0/single' % device, 0) # repeat records
        self.daq.setInt('/%s/scopes/0/trigenable' % device, 0)  # no trigger
        sc['scope_mod'].set('fft/power', True)
        sc['scope_mod'].set('fft/spectraldensity', True)
        sc['scope_mod'].set('averager/weight', avg_weight)
        sc['scope_mod'].set('mode',3) #0=simple, 1=time avg, 3= fft avg
        
        clockbase = self.daq.getDouble('/%s/clockbase' % device)
        fSa = clockbase / 2**(self.daq.getDouble('/%s/scopes/0/time' % device))
        length = self.daq.getInt('/%s/scopes/0/length' % device)
        sc['df'] = fSa/float(length)
        return (length//2, sc['df'])
        
    def getScopeWave(self, name):
        if name not in self.scopes:
            return None
        
        sc = self.scopes[name]
        device = sc['device']
        self.daq.setInt('/%s/scopes/0/channels/0/inputselect' % device, sc['inputselect1'])
        if sc['inputselect2'] is not None:
            self.daq.setInt('/%s/scopes/0/channels/1/inputselect' % device, sc['inputselect2'])
        
        sc['scope_mod'].set('averager/restart', 1) # reset averager -> weight needs to be set by user before
        sc['scope_mod'].subscribe('%s/scopes/0/wave' % device)
        self.daq.sync()

        sc['scope_mod'].execute()
        self.daq.setInt('/%s/scopes/0/enable' % device, 1)

        while sc['scope_mod'].getInt('records') < sc['no_samples']:
            time.sleep(1e-3)
        result = sc['scope_mod'].read()
        self.daq.setInt('/%s/scopes/0/enable' % device, 0)
        sc['scope_mod'].finish()
        sc['scope_mod'].unsubscribe('*')
    
        wave = np.transpose(result[device]['scopes']['0']['wave'][-1][0]['wave'])
        return (sc['df'], wave)