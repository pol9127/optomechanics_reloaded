"""
@ author: Andrei Militaru
@ date: 17th of January 2020
"""

import sys
try:
    sys.path.remove('D:\\Programming\\Python_Modules')
    sys.path.append("D:\\Programming/New_Modules")
except:
    sys.path.append("D:\\Programming/New_Modules")

from optomechanics.experiment.device.daq.ni7852r_support import apirio
from optomechanics.experiment.device.daq.ni7852r_support.apirio import DaqManager, ticks_to_dt, extract_interleaving, volt_to_bit, bit_to_volt
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import numpy as np
from threading import Thread
import pyqtgraph.console
from pyqtgraph.dockarea import *
#from boltzmann_redpitaya import SwimmerManager

class AoPointer():
    
    def __init__(self, daq, AO=1, delta=10):
        self.daq = daq
        self._AO = AO
        self.delta = delta
        
    def up(self):
        attribute = 'AO' + str(self.AO)
        current_value = getattr(self.daq, attribute)
        setattr(self.daq, attribute, current_value + self.delta)
        self.update_information()
        
    def down(self):
        attribute = 'AO' + str(self.AO)
        current_value = getattr(self.daq, attribute)
        setattr(self.daq, attribute, current_value - self.delta)
        self.update_information()
        
    def update_information(self):
        global current_value, instructions_text
        global instructions
        controlled_channel = 'Channel controlled: Analog Output, AO' + str(self.AO) + '.\n'
        attribute = 'AO' + str(self.AO)
        current_value = getattr(self.daq, attribute)
        new_information = 'Current output value: {:d} bits (= {:.2f} mV)'.format(current_value, bit_to_volt(current_value)*1e3)
        current_value = controlled_channel + new_information
        instructions.setText(instructions_text + current_value)
        
    def set(self, new_value):
        setattr(self.daq, 'AO' + str(self.AO), new_value)
        self.update_information()
        
    @property
    def AO(self):
        return self._AO
        
    @AO.setter
    def AO(self, new_channel):
        if 0 <= new_channel <= 7:
            self._AO = new_channel
            self.update_information()
        else:
            raise Exception('Channel exist only from AO0 to AO7.')
        

class SingleChannelMonitorer(Thread):
    
    def __init__(self, fifo, blocksize, p, curve, ticks):
        Thread.__init__(self)
        self.fifo = fifo
        self.blocksize = blocksize
        self.p = p
        self.curve = curve
        dt = ticks_to_dt(ticks)
        self.time = np.linspace(0, blocksize*dt, blocksize)
        
    def run(self):
        first_run = True
        self.fifo.start()
        while True:
            data = self.fifo.read(self.blocksize, timeout_ms=-1).data
            self.curve.setData(self.time, data)
            if first_run:
                first_run = False
                self.p.enableAutoRange('xy', False)
                QtGui.QApplication.processEvents()
   
    
class TripleChannelMonitorer(Thread):
    
    def __init__(self, fifo, blocksize, px, curvex, py, curvey, pz, curvez, ticks):
        Thread.__init__(self)
        self.fifo = fifo
        self.blocksize = blocksize - (blocksize%3)
        self.px = px
        self.curvex = curvex
        self.py = py
        self.curvey = curvey
        self.pz = pz
        self.curvez = curvez
        dt = ticks_to_dt(ticks)
        self.time = np.linspace(0, blocksize/3*dt, int(blocksize/3))
        
    def run(self):
        first_run = True
        self.fifo.start()
        while True:
            data = self.fifo.read(self.blocksize, timeout_ms=-1).data
            (x, y, z) = extract_interleaving(data)
            self.curvex.setData(self.time, x)
            self.curvey.setData(self.time, y)
            self.curvez.setData(self.time, z)
            if first_run:
                first_run = False
                self.px.enableAutoRange('xy', False)
                self.py.enableAutoRange('xy', False)
                self.pz.enableAutoRange('xy', False)
                QtGui.QApplication.processEvents()



class TripleHistogramMonitorer(Thread):
    
    def __init__(self, fifo, blocksize, px, curvex, py, curvey, pz, curvez, ticks,
                 pfx, curvefx, pfy, curvefy, pfz, curvefz, alpha=0.4, bins=20, range=None):
        Thread.__init__(self)
        self.fifo = fifo
        self.blocksize = blocksize - (blocksize%3)
        self.px = px
        self.curvex = curvex
        self.py = py
        self.curvey = curvey
        self.pz = pz
        self.curvez = curvez
        dt = ticks_to_dt(ticks)
        self.time = np.linspace(0, int(blocksize/3)*dt, int(blocksize/3))*1e3
        self.pfx = pfx
        self.curvefx = curvefx
        self.pfy = pfy
        self.curvefy = curvefy
        self.pfz = pfz
        self.curvefz = curvefz
        self.curvefz = curvefz
        self.Lf = 2**int(np.log2(self.blocksize))
        self.freq = np.linspace(0, 1/(2*ticks_to_dt(ticks)), int(self.Lf/2)+1)/1e3 #in khz
        self.alpha = alpha
        self.bins = bins
        self.range = range
        
    def run(self):
        first_run = True
        self.fifo.start()
        while True:
            data = self.fifo.read(self.blocksize, timeout_ms=-1).data
            (x, y, z) = extract_interleaving(data)
            if first_run:
                if self.range is not None:
                    nx, binsx = np.histogram(x, self.bins, range=range, density=True)
                    ny, binsy = np.histogram(y, self.bins, range=range, density=True)
                    nz, binsz = np.histogram(z, self.bins, range=range, density=True)
                else:
                    nx, binsx = np.histogram(x, self.bins, density=True)
                    ny, binsy = np.histogram(y, self.bins, density=True)
                    nz, binsz = np.histogram(z, self.bins, density=True)
                nx0 = np.zeros_like(nx)
                ny0 = np.zeros_like(ny)
                nz0 = np.zeros_like(nz)
            else:
                nx0 = nx
                #binsx0 = binsx
                ny0 = ny
                #binsy0 = binsy
                nz0 = nz
                #binsz0 = binsz
                if self.range is not None:
                    nx, binsx = np.histogram(x, self.bins, range=range, density=True)
                    ny, binsy = np.histogram(y, self.bins, range=range, density=True)
                    nz, binsz = np.histogram(z, self.bins, range=range, density=True)
                else:
                    nx, binsx = np.histogram(x, self.bins, density=True)
                    ny, binsy = np.histogram(y, self.bins, density=True)
                    nz, binsz = np.histogram(z, self.bins, density=True)
            self.curvex.setData(self.time, x)
            self.curvey.setData(self.time, y)
            self.curvez.setData(self.time, z)
            self.curvefx.setData(binsx[:-1], self.alpha*nx0 + (1-self.alpha)*nx)
            self.curvefy.setData(binsy[:-1], self.alpha*ny0 + (1-self.alpha)*ny)
            self.curvefz.setData(binsz[:-1], self.alpha*nz0 + (1-self.alpha)*nz)
            if first_run:
                first_run = False
                self.px.enableAutoRange('x', False)
                self.py.enableAutoRange('x', False)
                self.pz.enableAutoRange('x', False)
                self.pfx.enableAutoRange('y', False)
                self.pfy.enableAutoRange('y', False)
                self.pfz.enableAutoRange('y', False)
                #QtGui.QApplication.processEvents()

class TripleFourierMonitorer(Thread):
    
    def __init__(self, fifo, blocksize, px, curvex, py, curvey, pz, curvez, ticks,
                 pfx, curvefx, pfy, curvefy, pfz, curvefz, alpha=0.4):
        Thread.__init__(self)
        self.fifo = fifo
        self.blocksize = blocksize - (blocksize%3)
        self.px = px
        self.curvex = curvex
        self.py = py
        self.curvey = curvey
        self.pz = pz
        self.curvez = curvez
        dt = ticks_to_dt(ticks)
        self.time = np.linspace(0, int(blocksize/3)*dt, int(blocksize/3))*1e3
        self.pfx = pfx
        self.curvefx = curvefx
        self.pfy = pfy
        self.curvefy = curvefy
        self.pfz = pfz
        self.curvefz = curvefz
        self.curvefz = curvefz
        self.Lf = 2**int(np.log2(self.blocksize))
        self.freq = np.linspace(0, 1/(2*ticks_to_dt(ticks)), int(self.Lf/2)+1)/1e3 #in khz
        self.alpha = alpha
        
    def run(self):
        first_run = True
        self.fifo.start()
        while True:
            data = self.fifo.read(self.blocksize, timeout_ms=-1).data
            (x, y, z) = extract_interleaving(data)
            if first_run:
                fftx = np.abs(np.fft.rfft(x, n=self.Lf))**2
                ffty = np.abs(np.fft.rfft(y, n=self.Lf))**2
                fftz = np.abs(np.fft.rfft(z, n=self.Lf))**2
            else:
                fftx = self.alpha*fftx + (1-self.alpha)*np.abs(np.fft.rfft(x, n=self.Lf))**2
                ffty = self.alpha*ffty + (1-self.alpha)*np.abs(np.fft.rfft(y, n=self.Lf))**2
                fftz = self.alpha*fftz + (1-self.alpha)*np.abs(np.fft.rfft(z, n=self.Lf))**2
            self.curvex.setData(self.time, x)
            self.curvey.setData(self.time, y)
            self.curvez.setData(self.time, z)
            self.curvefx.setData(self.freq, fftx)
            self.curvefy.setData(self.freq, ffty)
            self.curvefz.setData(self.freq, fftz)
            if first_run:
                first_run = False
                self.px.enableAutoRange('x', False)
                self.py.enableAutoRange('x', False)
                self.pz.enableAutoRange('x', False)
                self.pfx.enableAutoRange('xy', False)
                self.pfy.enableAutoRange('xy', False)
                self.pfz.enableAutoRange('xy', False)
                #QtGui.QApplication.processEvents()


app = QtGui.QApplication([])
window = QtGui.QMainWindow()
area = DockArea()
window.setCentralWidget(area)
window.resize(1000,600)
window.setWindowTitle('Data Streaming')

dconsole = Dock("Console", size=(900,100), closable=True)
dplots = Dock("Plots", size=(1000,500))
dAO = Dock('Change AO', size=(100,100))

area.addDock(dplots, 'top')
area.addDock(dconsole, 'bottom')
area.addDock(dAO, 'right', dconsole)     
window.showMaximized()

labelstyle = {}

win = pg.GraphicsWindow(title="ApiRio Monitoring")
win.showMaximized()
win.resize(1200,800)
win.setWindowTitle('Streaming data')
pg.setConfigOptions(antialias=True)

dplots.addWidget(win)

px = win.addPlot(title="X Channel")
curvex = px.plot()
px.setLabel('left', "X", units='bits', **labelstyle)
px.setLabel('bottom', "Time", units='ms', **labelstyle)
px.setYRange(-3000, 3000, padding=0)
px.showGrid(x=True, y=True, alpha=0.9)

py = win.addPlot(title="Y Channel")
curvey = py.plot()
py.setLabel('left', "Y", units='bits', **labelstyle)
py.setLabel('bottom', "Time", units='ms', **labelstyle)
py.setYRange(-3000, 3000, padding=0)
py.showGrid(x=True, y=True, alpha=0.9)

pz = win.addPlot(title="Z Channel")
curvez = pz.plot()
pz.setLabel('left', "Z", units='bits', **labelstyle)
pz.setLabel('bottom', "Time", units='ms', **labelstyle)
pz.setYRange(0, 700, padding=0)
pz.showGrid(x=True, y=True, alpha=0.9)

win.nextRow()

is_histogram = False
log_mode = not is_histogram

pfx= win.addPlot(title="PSDX Channel")
curvefx= pfx.plot()
pfx.setLabel('left', 'Sxx', units='bits^2/Hz', **labelstyle)
pfx.setLabel('bottom', "Frequency", units='kHz', **labelstyle)
pfx.setLogMode(x=False, y=log_mode)
if is_histogram:
    pfx.setXRange(-3000, 3000, padding=0)
pfx.showGrid(x=True, y=True, alpha=0.9)

pfy = win.addPlot(title="PSDY Channel")
curvefy = pfy.plot()
pfy.setLabel('left', 'Syy', units='bits^2/Hz', **labelstyle)
pfy.setLabel('bottom', "Frequency", units='kHz', **labelstyle)
pfy.setLogMode(x=False, y=log_mode)
if is_histogram:
    pfy.setXRange(-3000, 3000, padding=0)
pfy.showGrid(x=True, y=True, alpha=0.9)

pfz = win.addPlot(title="PSDZ Channel")
curvefz = pfz.plot()
pfz.setLabel('left', 'Szz', units='bits^2/Hz', **labelstyle)
pfz.setLabel('bottom', "Frequency", units='kHz', **labelstyle)
pfz.setLogMode(x=False, y=log_mode)
if is_histogram:
    pfz.setXRange(-0, 700, padding=0)
pfz.showGrid(x=True, y=True, alpha=0.9)

path = "D:\\Programming\\LabView\\Militaru_DAQ\\FPGA Bitfiles\\"
file = "MilitaruDAQ_FPGATarget_FirmwareForPytho_+lmkLI6ZzMk.lvbitx"

bitfile = path + file
bitfile = path + file
pointer_beginning = 300
AO = [0]*8
AO[1] = pointer_beginning
#AO[1] = volt_to_bit(0.1)
"""
swimmer_params = {'gain' : 0.6,
                  'output1' : 1,
                  'output2' : 4,
                  'input_to_use' : 0,
                  'divider' : 99,
                  'DC' : 0,
                  'cosine_adjuster' : 6,
                  'DC_adjuster' : 0} 
                  
swimmer = SwimmerManager(**swimmer_params)"""
print('About to start DAQ.')
daq = DaqManager(bitfile, 
                 Xchannel=3,
                 Ychannel=4,
                 Zchannel=6,
                 DCchannel=0,
                 auxChannel=1,
                 data_ticks=200,
                 DC_ticks=500,
                 aux_ticks=500,
                 data_elements=30000,
                 DC_elements=1000,
                 aux_elements=2000,
                 AO=AO)
print('Started DAQ.')      
daq.run_session()
print('Made it here')
if is_histogram:
    chosen_monitor = TripleHistogramMonitorer
else:
    chosen_monitor = TripleFourierMonitorer

monitor = chosen_monitor(daq.session.fifos['data-FIFO'],
                                      daq.data_elements,
                                      px,
                                      curvex,
                                      py,
                                      curvey,
                                      pz,
                                      curvez,
                                      daq.data_ticks,
                                      pfx,
                                      curvefx,
                                      pfy,
                                      curvefy,
                                      pfz,
                                      curvefz)

text = '''
Author: Andrei Militaru
Date: 17th of January 2020
Python console:
A the active DaqManager instance is already loaded under the name "daq". 
You can access any method and attribute of it, i.e.:
daq.AO0 = volt_to_bit(2) sets the analog output number 0 to 2 V.

The monitor thread is also loaded under the name "monitor".
Furthermore, the modules numpy (as np) and apirio have been already imported.'''

pointer = AoPointer(daq, AO=1, delta=20)
wconsole = pg.console.ConsoleWidget(namespace={'daq' : daq,
                                               'monitor' : monitor,
                                               'volt_to_bit' : volt_to_bit,
                                               'np' : np,
                                               'apirio' : apirio,
                                               'pointer' : pointer}, text=text)
                                               #'swimmer' : swimmer}, text=text)
dconsole.addWidget(wconsole)


wao = pg.LayoutWidget()
instructions_text = """To change to analog output channel: pointer.AO = <new value>
To change the increments: pointer.delta = <new value>

The command needs to be run in the console on the left.
Important Note: please change the value of the pointer analog channel only through the pointer API.
This guarantees that the piece of information written below is correctly updated.\n\n"""

current_value = ''

instructions = QtGui.QLabel(instructions_text + current_value)
upBtn = QtGui.QPushButton('++ Up')
downBtn = QtGui.QPushButton('-- Down')
wao.addWidget(instructions, row=0, col=0)
wao.addWidget(upBtn, row=1, col=0)
wao.addWidget(downBtn, row=2, col=0)
dAO.addWidget(wao)
upBtn.clicked.connect(pointer.up)
downBtn.clicked.connect(pointer.down)

pointer.set(pointer_beginning)
pointer.update_information()
monitor.start()

window.show()

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        try:
            QtGui.QApplication.instance().exec_()
        finally:
            try:
                swimmer.close()
                daq.session.close(reset_if_last_session=daq.session._reset_if_last_session_on_exit)
            except:
                pass