from PyQt5 import uic, QtCore, QtWidgets, QtGui
import os
import sys

class GUI(QtWidgets.QMainWindow):
    def __init__(self, **kwargs):
        QtWidgets.QDialog.__init__(self)
        self.ui = uic.loadUi('mainWindow.ui')
        self.ui.show()
        self.ui.pushButton.clicked.connect(self.do_sth)

    def do_sth(self):
        print('lulululul')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    sys.exit(app.exec_())
