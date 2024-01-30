from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QTableWidgetItem
from sqrt_gui_window import Ui_MainWindow
from dialog import Ui_Dialog
import sys
from paramiko import client
from io import StringIO


class ssh:
    client = None

    def __init__(self, address, username, password):
        print("Connecting to server.")
        self.client = client.SSHClient()
        self.client.set_missing_host_key_policy(client.AutoAddPolicy())
        self.client.connect(address, username=username, password=password, look_for_keys=False)

    def sendCommand(self, command):
        if (self.client):
            stdin, stdout, stderr = self.client.exec_command(command)
            while not stdout.channel.exit_status_ready():
                # Print data when available
                if stdout.channel.recv_ready():
                    alldata = stdout.channel.recv(1024)
                    prevdata = b"1"
                    while prevdata:
                        prevdata = stdout.channel.recv(1024)
                        alldata += prevdata

                    print(str(alldata, "utf8"))
        else:
            print("Connection not opened.")


ADDRESS = 'red-pitaya-03.ee.ethz.ch'
USERNAME = 'root'
PASSWORD = ''


class mydialog(QtWidgets.QDialog):
    def __init__(self):
        super(mydialog, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.pushButton_2.clicked.connect(self.abort)
        self.ui.pushButton.clicked.connect(self.load_information)

    def abort(self):
        self.close()
        sys.exit()

    def load_information(self):
        global ADDRESS
        global USERNAME
        global PASSWORD

        ADDRESS = self.ui.lineEdit.text()
        USERNAME = self.ui.lineEdit_2.text()
        PASSWORD = self.ui.lineEdit_3.text()
        self.close()


class mywindow(QtWidgets.QMainWindow):
    def __init__(self):
        global ADDRESS
        global USERNAME
        global PASSWORD

        application = mydialog()
        self.table = {'c': '10000',
                      'a': '800',
                      'delta': '0',
                      'offset': '0',
                      'DC': '0',
                      'coarse_c': '3',
                      'DAC1': '1',
                      'DAC2': '5'}

        application.show()
        application.exec()
        self.hostname = USERNAME + '@' + ADDRESS
        self.connection = ssh(ADDRESS, USERNAME, PASSWORD)
        super(mywindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.rows = 8
        self.cols = 2

        self.ui.lineEdit.setText('Connected to: ' + self.hostname)
        self.ui.lineEdit_3.setText('bitfiles/sqrt_LUT_2.bit')
        self.ui.lineEdit.setReadOnly(True)
        self.ui.tableWidget.setColumnCount(self.cols)
        self.ui.tableWidget.setRowCount(self.rows)
        self.ui.tableWidget.setHorizontalHeaderLabels(('Quantity', 'Value'))
        self.ui.tableWidget.setVerticalHeaderLabels(('', '', '', '', '', '', '', ''))
        row = 0
        for key in self.table:
            pair = (key, self.table[key])
            col = 0
            for item in pair:
                cellinfo = QTableWidgetItem(item)
                self.ui.tableWidget.setItem(row, col, cellinfo)
                if col == 0:
                    cellinfo.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                col += 1
            row += 1

        font = QtGui.QFont()
        font.setPointSize(9)
        self.ui.tableWidget.setFont(font)

        for col in range(self.cols + 1):
            self.ui.tableWidget.setColumnWidth(col, int(150 / self.cols))

        for row in range(self.rows + 1):
            self.ui.tableWidget.setRowHeight(row, int(150 / (self.rows + 1)))

        self.ui.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.ui.tableWidget.verticalHeader().setStretchLastSection(True)
        self.update_all_gpio()
        self.ui.tableWidget.cellChanged.connect(self.updateRegister)
        self.ui.lineEdit_2.editingFinished.connect(self.newCommand)
        self.ui.pushButton.clicked.connect(self.btnClicked)
        self.ui.pushButton_2.clicked.connect(self.updateParameters)

    def btnClicked(self):
        bitfile = self.ui.lineEdit_3.text()
        self.ui.textEdit.append('Loading ' + bitfile + ' file...')
        command = 'cat ' + bitfile + ' > /dev/xdevcfg'
        self.connection.sendCommand(command)
        self.ui.textEdit.append('Bitstream file loaded.')

    def updateParameters(self):
        self.ui.textEdit.append('Updating parameters:')
        for row in range(self.rows):
            key = self.ui.tableWidget.item(row, 0).text()
            value = self.ui.tableWidget.item(row, 1).text()
            self.table[key] = value
            self.ui.textEdit.append(key + ': ' + value)
        self.update_all_gpio()

    @staticmethod
    def gpio_address(gpio):
        if gpio == 0:
            return '0x41200000'
        elif gpio == 1:
            return '0x41210000'
        elif gpio == 2:
            return '0x41220000'
        else:
            return None

    @staticmethod
    def signed2unsigned(n, bits=16, base=2):
        if n >= 0 and n < base ** (bits - 1):
            return n
        elif n < 0 and n >= -base ** (bits - 1):
            return base ** bits + 1 + n
        else:
            raise Exception('Number of bits too low to encoding the input.')

    @staticmethod
    def assign_to_gpio(key):
        if key == 'DAC1' or key == 'DAC2' or key == 'delta' or key == 'offset':
            return 0
        elif key == 'c' or key == 'a':
            return 1
        elif key == 'DC' or key == 'coarse_c':
            return 2
        else:
            return None

    def update_all_gpio(self):
        self.update_gpio0()
        self.update_gpio1()
        self.update_gpio2()

    def update_gpio0(self):
        gpio_address = self.gpio_address(0)
        command = '/opt/redpitaya/bin/monitor ' + gpio_address + ' '
        DAC1 = int(self.table['DAC1'])
        DAC2 = int(self.table['DAC2'])
        delta = int(self.table['delta'])
        offset = self.signed2unsigned(int(self.table['offset']), bits=16)
        value = DAC1 + 2 ** 3 * DAC2 + 2 ** 6 * delta + 2 ** 16 * offset
        command += str(value)
        self.ui.textEdit.append('Updating address ' + gpio_address + ' to ' + str(value) + '.')
        self.connection.sendCommand(command)

    def update_gpio1(self):
        gpio_address = self.gpio_address(1)
        command = '/opt/redpitaya/bin/monitor ' + gpio_address + ' '
        c = int(self.table['c'])
        a = self.signed2unsigned(int(self.table['a']), bits=16)
        value = c + 2 ** 16 * a
        command += str(value)
        self.ui.textEdit.append('Updating address ' + gpio_address + ' to ' + str(value) + '.')
        self.connection.sendCommand(command)

    def update_gpio2(self):
        gpio_address = self.gpio_address(2)
        command = '/opt/redpitaya/bin/monitor ' + gpio_address + ' '
        DC = self.signed2unsigned(int(self.table['DC']), bits=14)
        coarse_c = int(self.table['coarse_c'])
        value = DC + 2 ** 14 * coarse_c
        command += str(value)
        self.ui.textEdit.append('Updating address ' + gpio_address + ' to ' + str(value) + '.')
        self.connection.sendCommand(command)

    def updateRegister(self, row, col):
        key = self.ui.tableWidget.item(row, 0).text()
        gpio = self.assign_to_gpio(key)
        self.table[key] = self.ui.tableWidget.item(row, 1).text()

        if gpio == 0:
            self.update_gpio0()
        elif gpio == 1:
            self.update_gpio1()
        elif gpio == 2:
            self.update_gpio2()
        else:
            raise Exception('GPIO value not available.')

    def newCommand(self):
        text = self.ui.lineEdit_2.text()
        self.ui.textEdit.append('Run Command: ' + text)
        self.ui.lineEdit_2.clear()
        old_stdout = sys.stdout
        temp_stdout = StringIO()
        sys.stdout = temp_stdout
        self.connection.sendCommand(text)
        text = temp_stdout.getvalue()
        self.ui.textEdit.append(text + '\n')
        sys.stdout = old_stdout


app = QtWidgets.QApplication([])

application = mywindow()

application.show()

sys.exit(app.exec())
