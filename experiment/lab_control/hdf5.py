import h5py
import os
from datetime import datetime
import numpy as np
from PyQt5 import QtWidgets, QtCore, Qt, uic, QtGui
from collections import OrderedDict


class Frontend(QtWidgets.QMainWindow):
    param_types = [float, int, str]
    hdf5_file = None
    def __init__(self, parent_gui=None, parent_data=None):
        super(Frontend, self).__init__(parent_gui)
        uic.loadUi(os.path.join('ui_files', 'export.ui'), self)
        if parent_data is None:
            self.parent = parent_gui
        else:
            self.parent = parent_data

        self.pushButton_open.clicked.connect(self.file_dialog)
        self.lineEdit_filename.textChanged.connect(lambda fname : self.open_file(fname))
        self.checkBox_edit.stateChanged.connect(lambda state : self.textEdit_general_description.setReadOnly({0 : True, 2 : False}[state]))
        self.pushButton_plus.clicked.connect(self.add_row)
        self.pushButton_minus.clicked.connect(self.remove_row)
        self.pushButton_save.clicked.connect(self.save_and_close)
        self.textEdit_general_description.setReadOnly(True)


        self.param_types_dict = OrderedDict([(param.__name__, param) for param in self.param_types])
        self.equip_param_table()

    def file_dialog(self):
        fname = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', os.path.split(__file__)[0], '*.hdf5')
        if fname[0] != '':
            if not fname[0].endswith('hdf5'):
                self.lineEdit_filename.setText(fname[0] + '.hdf5')
            else:
                self.lineEdit_filename.setText(fname[0])

    def open_file(self, fname):
        if fname == '':
            return

        self.hdf5_file = Hdf5Container(fname)
        self.lineEdit_n_measurement.setText(str(self.hdf5_file.n_measurement))
        self.lineEdit_n_calibration.setText(str(self.hdf5_file.n_calibration))
        self.textEdit_general_description.setText(self.hdf5_file.description)
        if self.hdf5_file.description == '':
            self.checkBox_edit.setCheckState(2)

        for type in self.hdf5_file.types:
            self.comboBox_type.addItem(type)

        self.comboBox_type.setCurrentText(self.hdf5_file.types[0])

    def equip_param_table(self):
        n_row = self.tableWidget_parameters.rowCount()
        for row in range(n_row):
            combo = QtWidgets.QComboBox()
            for type in self.param_types_dict:
                combo.addItem(type)
            self.tableWidget_parameters.setCellWidget(row, 2, combo)

    def add_row(self):
        n_row = self.tableWidget_parameters.rowCount()
        self.tableWidget_parameters.setRowCount(n_row + 1)
        combo = QtWidgets.QComboBox()
        for type in self.param_types_dict:
            combo.addItem(type)
        self.tableWidget_parameters.setCellWidget(n_row, 2, combo)

    def remove_row(self):
        n_row = self.tableWidget_parameters.rowCount()
        self.tableWidget_parameters.setRowCount(n_row - 1)

    def save_and_close(self):
        self.parent.hdf5_file = self.hdf5_file
        if self.hdf5_file is None:
            self.parent.operation = None
        else:
            self.parent.operation = Operation(name=self.lineEdit_name.text().strip())
            self.parent.operation.attributes['description'] = self.textEdit_description.toPlainText().strip()
            self.parent.operation.attributes['type'] = self.comboBox_type.currentText().strip()
            n_params = self.tableWidget_parameters.rowCount()
            params = []
            for i in range(n_params):
                param = self.tableWidget_parameters.item(i, 0)
                if param is not None:
                    param = param.text().strip()
                    if param != '':
                        typ = self.param_types_dict[self.tableWidget_parameters.cellWidget(i, 2).currentText().strip()]
                        value = typ(self.tableWidget_parameters.item(i, 1).text().strip())
                        params.append((param, value))
            self.parent.operation.add_parameters(params)

            general_description = self.textEdit_general_description.toPlainText().strip()
            if self.parent.hdf5_file.description == '' and general_description != '':
                del self.parent.hdf5_file.description
                self.parent.hdf5_file.description = general_description
            elif self.parent.hdf5_file.description != '' and self.parent.hdf5_file.description != general_description:
                del self.parent.hdf5_file.description
                self.parent.hdf5_file.description = general_description
        self.close()


class Hdf5Container(object):
    hdf5 = None
    appending = None
    closed = None
    types = ['measurement', 'calibration']

    def __init__(self, filename):
        if os.path.isfile(filename):
            self.appending = True
        else:
            self.appending = False

        self.hdf5 = h5py.File(filename, 'a')
        self.filename = filename

        if not self.appending:
            self.hdf5.attrs['n_calibration'] = 0
            self.hdf5.attrs['n_measurement'] = 0
            self.hdf5.attrs['description'] = ''
            self.hdf5.attrs['created'] = str(datetime.now())
            self.hdf5.attrs['modified'] = str(datetime.now())

        self.closed = False

    @property
    def modified(self):
        return self.hdf5.attrs['modified']

    @modified.setter
    def modified(self, dt):
        if isinstance(dt, str):
            self.hdf5.attrs['modified'] = dt
        else:
            self.hdf5.attrs['modified'] = str(dt)

    @property
    def description(self):
        return self.hdf5.attrs['description']

    @description.setter
    def description(self, desc):
        if self.hdf5.attrs['description'] != '':
            print('ERROR: Description is aleady existing. If you want to change it first delete the old one.')
        else:
            self.hdf5.attrs['description'] = desc

    @description.deleter
    def description(self):
        self.hdf5.attrs['description'] = ''

    @property
    def n_measurement(self):
        return self.hdf5.attrs['n_measurement']

    @n_measurement.setter
    def n_measurement(self, n):
        self.hdf5.attrs['n_measurement'] = n

    @property
    def n_calibration(self):
        return self.hdf5.attrs['n_calibration']

    @n_calibration.setter
    def n_calibration(self, n):
        self.hdf5.attrs['n_calibration'] = n

    @property
    def created(self):
        return self.hdf5.attrs['created']

    def add_operation(self, operation):
        if isinstance(operation, list):
            for op in operation:
                if op.data is None:
                    continue
                typ = op.attributes['type']
                if typ not in self.hdf5.keys():
                    self.hdf5.create_group(typ)

                if typ == 'measurement':
                    dset = self.hdf5[typ].create_dataset(str(self.n_measurement).zfill(4) + '_' + op.name, data=op.data)
                elif typ == 'calibration':
                    dset = self.hdf5[typ].create_dataset(str(self.n_calibration).zfill(4) + '_' + op.name, data=op.data)

                for key in op.attributes:
                    dset.attrs[key] = op.attributes[key]

            if typ == 'measurement':
                self.n_measurement += 1
            elif typ == 'calibration':
                self.n_calibration += 1


        else:
            if operation.data is None:
                return
            typ = operation.attributes['type']
            if typ not in self.hdf5.keys():
                self.hdf5.create_group(typ)

            if typ == 'measurement':
                dset = self.hdf5[typ].create_dataset(str(self.n_measurement).zfill(4) + '_' + operation.name, data=operation.data)
                self.n_measurement += 1
            elif typ == 'calibration':
                dset = self.hdf5[typ].create_dataset(str(self.n_calibration).zfill(4) + '_' + operation.name, data=operation.data)
                self.n_calibration += 1

            for key in operation.attributes:
                dset.attrs[key] = operation.attributes[key]

    def del_operation(self, operation_name, typ='measurement'):
        if typ not in self.hdf5.keys():
            print('ERROR: No ' + typ + ' found.')
            return
        if operation_name not in self.hdf5[type].keys():
            print('ERROR: No ' + typ +' found.')
            return

    def close(self):
        if self.closed:
            return
        else:
            self.hdf5.close()
            self.closed = True

    def open(self):
        if self.closed:
            self.hdf5 = h5py.File(self.filename, 'a')
            self.closed = False


class Operation(object):
    attributes = {'description' : '',
                  'timestamp' : str(datetime.now()),
                  'label' : [],
                  'unit' : [],
                  'type' : ''}
    def __init__(self, name='', data=None):
        self.name = name
        self.data = data

    def add_parameters(self, parameters):
        self.attributes.update(parameters)



if __name__ == '__main__':
    fname = 'text.hdf5'
    testfile = Hdf5Container(fname)

    test_meas = Operation('test', data=np.random.rand(10,4))
    testfile.add_measurement(test_meas)
