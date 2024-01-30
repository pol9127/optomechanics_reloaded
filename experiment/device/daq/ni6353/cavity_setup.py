from optomechanics.experiment.device.daq.ni6353.monitor import MonitorWidget
from PyQt5 import uic, QtCore, QtWidgets, QtGui, Qt
import sys
import numpy as np
import pyqtgraph as pg
import os
import peakutils as pu

def str2bool(s):
    if s == 'True' or s == '1':
        return True
    elif s == 'False' or s == '0':
        return False

class AddCalcWidget(QtWidgets.QMainWindow):
    attributes_dict = {'Choose Operation': [],
                       'Peak-Tracer': ['Channel', 'Range Min', 'Range Max', 'Averages', 'Follow Range'],
                       'Average': ['Channel', 'Range Min', 'Range Max', 'Averages'],
                       'Peak-to-Peak': ['Channel', 'Range Min', 'Range Max', 'Averages']}
    attributes_dtype_dict = {'Choose Operation': [],
                             'Peak-Tracer': [str, float, float, int, str2bool],
                             'Average': [str, float, float, int],
                             'Peak-to-Peak': [str, float, float, int]}

    def __init__(self, parent=None):
        super(__class__, self).__init__(parent)
        self.ui = uic.loadUi(os.path.join('gui', 'addMath.ui'), self)
        self.ui.comboBox.currentTextChanged.connect(self.set_attributes)
        self.ui.pushButton.clicked.connect(self.get_attributes)
        self.parent = parent

    def set_attributes(self, text):
        attributes = self.attributes_dict[text]
        self.ui.tableWidget.setRowCount(len(attributes))
        for row, attr in enumerate(attributes):
            item = QtWidgets.QTableWidgetItem(attr)
            self.ui.tableWidget.setVerticalHeaderItem(row, item)

    def get_attributes(self):
        operation = self.ui.comboBox.currentText()
        attributes = {key: self.ui.tableWidget.item(row, 0).text()
                      for row, key in enumerate(self.attributes_dict[operation])}
        dtypes = self.attributes_dtype_dict[operation]
        for key, dt in zip(attributes, dtypes):
            try:
                attributes[key] = dt(attributes[key])
            except:
                print('Datatype of attribute {0} not understood. Expected {1}. Aborting.'.format(key, dt.__name__))
                return
        attributes['Type'] = operation
        new_row_count = self.parent.ui.tableWidget.rowCount() + 1
        self.parent.ui.tableWidget.setRowCount(new_row_count)
        horizontal_header = [self.parent.ui.tableWidget.horizontalHeaderItem(i).text()
                             for i in range(self.parent.ui.tableWidget.columnCount())]
        for col, key in enumerate(horizontal_header):
            if key in attributes:
                item = QtWidgets.QTableWidgetItem(str(attributes[key]))
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.parent.ui.tableWidget.setItem(new_row_count - 1, col, item)


class MathWidget(QtWidgets.QDockWidget):
    clearTable = False
    def __init__(self):
        super(__class__, self).__init__()
        self.ui = uic.loadUi(os.path.join('gui', 'math.ui'), self)
        self.ui.pushButton.clicked.connect(self.add_calculation)
        self.ui.pushButton_2.clicked.connect(self.initiateTableClearing)

    def add_calculation(self):
        self.add_calc_widget = AddCalcWidget(self)
        self.add_calc_widget.show()

    def initiateTableClearing(self):
        self.clearTable = True




class CavityMonitoring(MonitorWidget):
    peak_tracer_maxima = {}
    average_vals = {}
    peak_to_peak_minima = {}
    peak_to_peak_maxima = {}

    def __init__(self, **kwargs):
        super(__class__, self).__init__(**kwargs)
        self.math_widget = MathWidget()
        self.ui.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.math_widget)

    def find_sweep_lines(self):
        if self.task.configuration['ttl_trigger']:
            cav_sweep_idx = pu.indexes(np.abs(np.diff(self.data[0])), thres=0.5)
        else:
            cav_sweep_idx = np.array([np.argmin(self.data[0]), np.argmax(self.data[0])])
        cav_sweep_rt = self.task.runtime[cav_sweep_idx]
        for rt in cav_sweep_rt:
            if 'cavity sweep' in self.plotWidgets:
                self.plotWidgets['cavity sweep'].addItem(pg.InfiniteLine(rt))
            if 'cavity transmission' in self.plotWidgets:
                self.plotWidgets['cavity transmission'].addItem(pg.InfiniteLine(rt))
            if 'cavity reflection' in self.plotWidgets:
                self.plotWidgets['cavity reflection'].addItem(pg.InfiniteLine(rt))
            if 'PDH signal' in self.plotWidgets:
                self.plotWidgets['PDH signal'].addItem(pg.InfiniteLine(rt))

    def do_calculations(self):
        operations = [self.math_widget.ui.tableWidget.item(row, 0).text() + '_{0}'.format(row)
                      for row in range(self.math_widget.ui.tableWidget.rowCount())]
        for row, op in enumerate(operations):
            if 'Peak-Tracer' in op:
                params = [self.math_widget.ui.tableWidget.item(row, col).text()
                          if self.math_widget.ui.tableWidget.item(row, col) is not None else ''
                          for col in range(1, self.math_widget.ui.tableWidget.columnCount())]
                params.pop(3)
                runtime, New_Range_Min, New_Range_Max = self.peak_tracer(op, params[0], float(params[1]),
                                                                         float(params[2]), int(params[3]),
                                                                         str2bool(params[4]))

                item = QtWidgets.QTableWidgetItem(str(np.mean(self.peak_tracer_maxima[op])))
                self.math_widget.ui.tableWidget.setItem(row, 4, item)
                item = QtWidgets.QTableWidgetItem(str(New_Range_Min))
                self.math_widget.ui.tableWidget.setItem(row, 2, item)
                item = QtWidgets.QTableWidgetItem(str(New_Range_Max))
                self.math_widget.ui.tableWidget.setItem(row, 3, item)

                self.plotWidgets[params[0]].addItem(pg.InfiniteLine(runtime))

            elif 'Average' in op:
                params = [self.math_widget.ui.tableWidget.item(row, col).text()
                          if self.math_widget.ui.tableWidget.item(row, col) is not None else ''
                          for col in range(1, self.math_widget.ui.tableWidget.columnCount())]
                params.pop(3)
                self.average(op, params[0], float(params[1]), float(params[2]), int(params[3]))
                item = QtWidgets.QTableWidgetItem(str(np.mean(self.average_vals[op])))
                self.math_widget.ui.tableWidget.setItem(row, 4, item)
                self.plotWidgets[params[0]].addItem(pg.InfiniteLine(np.mean(self.average_vals[op]), angle=0))

            elif 'Peak-to-Peak' in op:
                params = [self.math_widget.ui.tableWidget.item(row, col).text()
                          if self.math_widget.ui.tableWidget.item(row, col) is not None else ''
                          for col in range(1, self.math_widget.ui.tableWidget.columnCount())]
                params.pop(3)
                self.peak_to_peak(op, params[0], float(params[1]), float(params[2]), int(params[3]))
                item = QtWidgets.QTableWidgetItem(str(np.mean(self.peak_to_peak_maxima[op]) - np.mean(self.peak_to_peak_minima[op])))
                self.math_widget.ui.tableWidget.setItem(row, 4, item)
                self.plotWidgets[params[0]].addItem(pg.InfiniteLine(np.mean(self.peak_to_peak_maxima[op]), angle=0))
                self.plotWidgets[params[0]].addItem(pg.InfiniteLine(np.mean(self.peak_to_peak_minima[op]), angle=0))

    def peak_tracer(self, Operation_ID, Channel, Range_Min, Range_Max, Averages, Follow_Range):
        channel_labels = np.array(self.task.configuration['channel_labels'])

        if Channel in channel_labels:
            si_scaling = self.plotWidgets[Channel].plotItem.getScale('bottom').autoSIPrefixScale
            ch_idx = np.arange(len(channel_labels))[channel_labels == Channel][0]
            runtime_idx_range = np.searchsorted(self.task.runtime, [Range_Min / si_scaling, Range_Max / si_scaling])

            maximum_idx = np.argmax(self.data[ch_idx][runtime_idx_range[0]:runtime_idx_range[1]])

            maximum = self.data[ch_idx][runtime_idx_range[0]:runtime_idx_range[1]][maximum_idx]
            runtime = self.task.runtime[runtime_idx_range[0]:runtime_idx_range[1]][maximum_idx]
            if Operation_ID in self.peak_tracer_maxima:
                self.peak_tracer_maxima[Operation_ID].append(maximum)
                if len(self.peak_tracer_maxima[Operation_ID]) > Averages:
                    self.peak_tracer_maxima[Operation_ID] = self.peak_tracer_maxima[Operation_ID][-Averages:]
            else:
                self.peak_tracer_maxima[Operation_ID] = [maximum]

            if Follow_Range:
                New_Range_Min = runtime * si_scaling - (Range_Max - Range_Min) / 2
                New_Range_Max = runtime * si_scaling + (Range_Max - Range_Min) / 2
                return runtime, New_Range_Min, New_Range_Max
            else:
                return runtime, Range_Min, Range_Max

    def average(self, Operation_ID, Channel, Range_Min, Range_Max, Averages):
        channel_labels = np.array(self.task.configuration['channel_labels'])

        if Channel in channel_labels:
            si_scaling = self.plotWidgets[Channel].plotItem.getScale('bottom').autoSIPrefixScale
            ch_idx = np.arange(len(channel_labels))[channel_labels == Channel][0]
            runtime_idx_range = np.searchsorted(self.task.runtime, [Range_Min / si_scaling, Range_Max / si_scaling])

            average = np.mean(self.data[ch_idx][runtime_idx_range[0]:runtime_idx_range[1]])

            if Operation_ID in self.average_vals:
                self.average_vals[Operation_ID].append(average)
                if len(self.average_vals[Operation_ID]) > Averages:
                    self.average_vals[Operation_ID] = self.average_vals[Operation_ID][-Averages:]
            else:
                self.average_vals[Operation_ID] = [average]

    def peak_to_peak(self, Operation_ID, Channel, Range_Min, Range_Max, Averages):
        channel_labels = np.array(self.task.configuration['channel_labels'])

        if Channel in channel_labels:
            si_scaling = self.plotWidgets[Channel].plotItem.getScale('bottom').autoSIPrefixScale
            ch_idx = np.arange(len(channel_labels))[channel_labels == Channel][0]
            runtime_idx_range = np.searchsorted(self.task.runtime, [Range_Min / si_scaling, Range_Max / si_scaling])

            minimum = np.min(self.data[ch_idx][runtime_idx_range[0]:runtime_idx_range[1]])
            maximum = np.max(self.data[ch_idx][runtime_idx_range[0]:runtime_idx_range[1]])

            if Operation_ID in self.peak_to_peak_minima:
                self.peak_to_peak_minima[Operation_ID].append(minimum)
                self.peak_to_peak_maxima[Operation_ID].append(maximum)
                if len(self.peak_to_peak_minima[Operation_ID]) > Averages:
                    self.peak_to_peak_minima[Operation_ID] = self.peak_to_peak_minima[Operation_ID][-Averages:]
                    self.peak_to_peak_maxima[Operation_ID] = self.peak_to_peak_maxima[Operation_ID][-Averages:]
            else:
                self.peak_to_peak_minima[Operation_ID] = [minimum]
                self.peak_to_peak_maxima[Operation_ID] = [maximum]


    def manipulate_plot(self):
        if self.ui.checkBox.isChecked():
            self.find_sweep_lines()
        self.do_calculations()
        if self.math_widget.clearTable:
            self.math_widget.ui.tableWidget.clearContents()
            self.math_widget.ui.tableWidget.setRowCount(0)
            self.math_widget.clearTable = False
            self.peak_tracer_maxima = {}
            self.average_vals = {}
            self.peak_to_peak_minima = {}
            self.peak_to_peak_maxima = {}


app = QtWidgets.QApplication(sys.argv)
window = CavityMonitoring(data_len=4600,
                          sample_rate=130e3,
                          channels=[
                              'ai0',
                              'ai1',
                              'ai2',
                              'ai3',
                              'ai8',
                              # 'ai9',
                              # 'ai10',
                              'ai11'
                          ],
                          channel_labels=[
                              'trigger',
                              'cavity sweep',
                              'cavity transmission',
                              'cavity reflection',
                              'PDH signal',
                              # 'Cavity Homodyne M-',
                              # 'Cavity Homodyne M+',
                              'Cavity Homodyne RF'
                          ],
                          voltage_range=[-10, 10],
                          trigger_slope=None,
                          trigger_level=1,
                          ttl_trigger=True)
#
# window = CavityMonitoring(data_len=4600,
#                        sample_rate=200e3,
#                        channels=['ai9', 'ai10', 'ai11'],
#                        channel_labels=['Cavity Homodyne M-', 'Cavity Homodyne M+', 'Cavity Homodyne RF'],
#                        voltage_range=[-10, 10],
#                        trigger_slope=None,
#                        trigger_level=-1.0)

sys.exit(app.exec_())

# m = MathWidget()