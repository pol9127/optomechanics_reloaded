import os
os.system('pyuic5 -x gage_sig_gui.ui -o ui_gage_sig_gui.py')

import sys
from PyQt5 import QtWidgets, QtCore, Qt
from glob import iglob
from optomechanics.post_processing.read.gage_sig import MetaData
from ui_gage_sig_gui import Ui_MainWindow


class GaGeSIGGui(Ui_MainWindow):
    def __init__(self, dialog_):
        Ui_MainWindow.__init__(self)
        self.dialog = dialog_
        self.setupUi(dialog_)
        self.pushButton_root.clicked.connect(self.show_dialog)


    def show_dialog(self):
        text= QtWidgets.QFileDialog.getExistingDirectory(self.dialog, 'Root Directory', os.path.expanduser('~'))
        if text != '':
            self.lineEdit_root.setText(str(text))
            self.fill_table()

    def fill_table(self):

        metadata_parameters = ['timestamp', 'channel', 'comment', 'pre trigger samples', 'post trigger samples',
                               'trigger source', 'sample rate', 'coupling', 'capture gain']

        self.tableWidget.setColumnCount(1 + len(metadata_parameters))
        self.tableWidget.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem('Filename'))
        counter_col = 1
        for mp in metadata_parameters:
            self.tableWidget.setHorizontalHeaderItem(counter_col, QtWidgets.QTableWidgetItem(mp))
            counter_col += 1

        directory = self.lineEdit_root.text()
        files = iglob(os.path.join(directory,'**','*.sig'), recursive=True)
        counter_row = 0
        for fl in files:
            filename = os.path.split(fl)[-1]
            self.tableWidget.setRowCount(counter_row + 1)
            item = QtWidgets.QTableWidgetItem(filename)
            metadata = MetaData(fl).data

            item.setFlags(QtCore.Qt.ItemIsSelectable |  QtCore.Qt.ItemIsEnabled )
            self.tableWidget.setItem(counter_row, 0, item)
            counter_col = 1
            for mp in metadata_parameters:
                item_param = QtWidgets.QTableWidgetItem(str(metadata[mp]))
                item_param.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self.tableWidget.setItem(counter_row, counter_col, item_param)
                counter_col += 1

            counter_row += 1


if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv)
	dialog = QtWidgets.QMainWindow()

	prog = GaGeSIGGui(dialog)

	dialog.show()
	sys.exit(app.exec_())