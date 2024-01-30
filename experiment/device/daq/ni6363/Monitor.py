##
# Simple GUI showing different channels
##

from PyQt5 import uic, QtCore, QtWidgets, QtGui
from NI6363 import MonitorWidget
import sys


app = QtWidgets.QApplication(sys.argv)
window = MonitorWidget(data_len=2000,
                       sample_rate=20000,
                       channels=['ai0','ai16'],
                       channel_labels=['X', 'Laser Power'],
                       voltage_range=[-1., 1.])
sys.exit(app.exec_())