import os

from collections import OrderedDict
import sys
from PyQt5 import QtWidgets, QtCore, Qt, uic
from copy import copy
from optomechanics.experiment.lab_control.tools import CustomDockWidget, BackgroundClose, QIPythonWidget
from optomechanics.experiment.lab_control.scope import Scope
from optomechanics.experiment.lab_control.device import Device
import optomechanics.experiment.lab_control.hdf5 as h5
from datetime import datetime as dt

class LabControl(QtWidgets.QMainWindow, object):
    hdf5_file = None
    operation = None
    sync_lock = [False, None]
    # log_file = open('log.log', 'w')

    def __init__(self, parent=None):
        super(LabControl, self).__init__(parent)
        uic.loadUi(os.path.join('ui_files', 'lab_control.ui'), self)

        self.pushButton_connect_manual.clicked.connect(self.connect_manual)
        self.pushButton_export_settings.clicked.connect(self.export_settings)
        self.pushButton_export.clicked.connect(self.export)

        self.dockWidgets = OrderedDict()
        device_classes = [device for device in Device.__subclasses__() if not device.isDevice] + [Device]
        subclasses = [device_class.__subclasses__() for device_class in device_classes]
        self.implemented_devices = {device.__name__ : device for subclass in subclasses for device in subclass if device.isDevice}
        self.connected_devices = {device.__name__ : {} for subclass in subclasses for device in subclass if device.isDevice}

        self.initialize_gui_elements()

        self.ipyConsole = QIPythonWidget(customBanner="Welcome to the embedded ipython console\n")
        self.gridLayout_terminal.addWidget(self.ipyConsole)

    def export_settings(self):
        export_dialog = h5.Frontend(self)
        export_dialog.show()

    def export(self):
        if self.hdf5_file is not None and self.operation is not None:
            all_devices = [dev for dev_type in self.connected_devices.values() for dev in dev_type.values()]
            operations = []
            for device in all_devices:
                device.operation = copy(self.operation)
                operations.append(device.export(kind='collective'))

            self.hdf5_file.open()
            self.hdf5_file.modified = dt.now()
            self.hdf5_file.add_operation(operations)
            self.hdf5_file.close()

    def get_devices(self):
        devices = [{device_class + '_' + device : self.connected_devices[device_class][device]} for device_class in self.connected_devices for device in self.connected_devices[device_class]]
        devices_flattened = {}
        for dev_dict in devices:
            devices_flattened.update(dev_dict)
        self.ipyConsole.pushVariables({'devices' : devices_flattened})


    def initialize_gui_elements(self):
        self.comboBox_availableScopes.clear()
        for scope in self.implemented_devices:
            self.comboBox_availableScopes.addItem(scope)

    def open_dock(self):
        selected_device_class = self.comboBox_availableScopes.currentText()
        selected_device_name = self.lineEdit_connect_manual_name.text()

        connection_settings = self.connect_manual_dialog.update_connection_settings()
        self.connect_manual_dialog.close()
        if selected_device_name.strip() == '':
            selected_device_name = 'default'

        device = self.implemented_devices[selected_device_class](selected_device_name, connection_settings, self)

        self.connected_devices[selected_device_class][selected_device_name] = device

        dockWidget = CustomDockWidget((selected_device_class, selected_device_name), self)
        dockWidget.setObjectName("dockWidget")
        device.dock = dockWidget

        dockWidgetContents = device.ControlWidget()

        dockWidgetContents.setObjectName("dockWidgetContents")
        dockWidget.setWidget(dockWidgetContents)
        dockWidget.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)

        created_docks = list(self.dockWidgets.values())
        if created_docks == []:
            self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dockWidget)
        else:
            self.tabifyDockWidget(created_docks[-1], dockWidget)
        self.dockWidgets[(selected_device_class, selected_device_name)] = dockWidget

        self.get_devices()

    def connect_manual(self):
        selected_scope = self.comboBox_availableScopes.currentText()

        self.connect_manual_dialog = self.implemented_devices[selected_scope].ManualInit(self)
        self.connect_manual_dialog.pushButton_connect.clicked.connect(self.open_dock)
        self.connect_manual_dialog.show()

    def closeEvent(self, *args, **kwargs):
        super(QtWidgets.QMainWindow, self).closeEvent(*args, **kwargs)
        print('disconnecting from devices', self.connected_devices)
        for dev in self.connected_devices:
            for cdev in self.connected_devices[dev].values():
                close_helper = BackgroundClose(cdev.dock.widget().close)
                close_helper.start()
        # self.log_file.close()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    prog = LabControl()
    prog.show()
    sys.exit(app.exec_())