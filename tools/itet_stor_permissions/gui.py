import paramiko as pa
from PyQt5 import uic, QtCore, QtWidgets
import sys

HOSTNAME = 'login.ee.ethz.ch'
relative = False
if relative:
    base_path = ''
else:
    base_path = sys.path[0][:-16]


class ItetStor:
    file_parameters = ['Filename', 'Owner', 'Owner Group', 'Perm. Owner', 'Perm. Group', 'Perm. Others']
    currentDir = b'/itet-stor'
    myPermission = None

    def __init__(self, hostname, username='root', password='root'):
        self.currentDir = self.currentDir + b'/' + bytes(username, 'utf-8') + b'/photonics'
        self.client = pa.SSHClient()
        self.client.set_missing_host_key_policy(pa.AutoAddPolicy())
        self.client.load_system_host_keys()
        try:
            self.client.connect(hostname, username=username, password=password)
            self.connection_failed = False
        except:
            self.connection_failed = True
            return
        # self.currentDir = self.pwd
        self.myPermission = self.my_permission

    @property
    def pwd(self):
        stdin, stdout, stderr = self.client.exec_command('pwd')
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip()
            return stdout

    @property
    def ls(self):
        stdin, stdout, stderr = self.client.exec_command('ls ' + self.currentDir.decode('utf-8', 'backslashreplace'))
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return stdout

    def _owner_(self, fn):
        fn = (self.currentDir + b'/' + fn).decode('utf-8', 'backslashreplace')
        fn = fn.replace(' ', '\\ ').replace('(', '\(').replace(')', '\)')
        stdin, stdout, stderr = self.client.exec_command('ls -ld ' + fn + " | awk '{print $3}'")
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return stdout[0]

    def _group_(self, fn):
        fn = (self.currentDir + b'/' + fn).decode('utf-8', 'backslashreplace')
        fn = fn.replace(' ', '\\ ').replace('(', '\(').replace(')', '\)')
        stdin, stdout, stderr = self.client.exec_command('ls -ld ' + fn + " | awk '{print $4}'")
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return stdout[0]

    def _permissions_(self, fn):
        fn = (self.currentDir + b'/' + fn).decode('utf-8', 'backslashreplace')
        fn = fn.replace(' ', '\\ ').replace('(', '\(').replace(')', '\)')
        stdin, stdout, stderr = self.client.exec_command('ls -ld ' + fn + " | awk '{print $1}'")
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return stdout[0]

    @property
    def owner(self):
        return [self._owner_(fn) for fn in self.ls]

    @property
    def group(self):
        return [self._group_(fn) for fn in self.ls]

    @property
    def permission(self):
        perm = [self._permissions_(fn) for fn in self.ls]
        perm_owner = [p[1:4] for p in perm]
        perm_group = [p[4:7] for p in perm]
        perm_other = [p[7:10] for p in perm]
        return perm_owner, perm_group, perm_other

    def cd(self, direct):
        if direct == b'..':
            stdin, stdout, stderr = self.client.exec_command('cd ' + self.currentDir.rsplit(b'/', maxsplit=1)[0].decode('utf-8', 'backslashreplace'))
            stderr = stderr.read()
            if stderr != b'':
                return 1, stderr.strip()
            else:
                self.currentDir = self.currentDir.rsplit(b'/', maxsplit=1)[0]
                return 0, self.currentDir
        else:
            stdin, stdout, stderr = self.client.exec_command('cd ' + (self.currentDir + b'/' + direct).decode('utf-8', 'backslashreplace'))
            stderr = stderr.read()
            if stderr != b'':
                return 1, stderr.strip()
            else:
                self.currentDir = self.currentDir + b'/' + direct
                return 0, self.currentDir

    def _group_members_(self, group):
        stdin, stdout, stderr = self.client.exec_command('getent group '+ group)
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')[0].rsplit(b':', maxsplit=1)[-1].split(b',')
            return stdout

    @property
    def whoami(self):
        stdin, stdout, stderr = self.client.exec_command('whoami')
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')[0].rsplit(b':', maxsplit=1)[-1].split(b',')
            return stdout[0]

    @property
    def my_permission(self):
        if self.whoami in self._group_members_('itet-isg-photonics_internal'):
            return 'internal'
        elif self.whoami in self._group_members_('itet-isg-photonics'):
            return 'guest'
        else:
            return None

    def chmod(self, fn, perms, recursive=False):
        fn = (self.currentDir + b'/' + fn).decode('utf-8', 'backslashreplace')
        fn = fn.replace(' ', '\\ ').replace('(', '\(').replace(')', '\)')
        perms = perms.decode('utf-8', 'backslashreplace')
        if recursive:
            stdin, stdout, stderr = self.client.exec_command('chmod -R {0} {1}'.format(perms, fn))
        else:
            stdin, stdout, stderr = self.client.exec_command('chmod {0} {1}'.format(perms, fn))
        stderr = stderr.read()
        if stderr != b'':
            return 1, stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return 0, stdout[0]

    def chgrp(self, fn, g, recursive=False):
        fn = (self.currentDir + b'/' + fn).decode('utf-8', 'backslashreplace')
        fn = fn.replace(' ', '\\ ').replace('(', '\(').replace(')', '\)')
        g = g.decode('utf-8', 'backslashreplace')
        if recursive:
            stdin, stdout, stderr = self.client.exec_command('chgrp -R {0} {1}'.format(g, fn))
        else:
            stdin, stdout, stderr = self.client.exec_command('chgrp {0} {1}'.format(g, fn))
        stderr = stderr.read()
        if stderr != b'':
            return 1, stderr.strip()
        else:
            stdout = stdout.read().strip().split(b'\n')
            return 0, stdout[0]

    @property
    def groups(self):
        stdin, stdout, stderr = self.client.exec_command('groups')
        stderr = stderr.read()
        if stderr != b'':
            return stderr.strip()
        else:
            stdout = stdout.read().strip().split(b' ')
            return stdout



class EditWidget(QtWidgets.QMainWindow):
    header = ['read', 'write', 'execute']
    letters = ['r', 'w', 'x']
    def __init__(self, parent=None):
        super(__class__, self).__init__(parent)
        self.ui = uic.loadUi(base_path + 'editWidget.ui', self)
        self.parent = parent
        self.ui.lineEdit.setText(self.parent.selectedFilename.decode('utf-8', 'backslashreplace'))
        self.ui.pushButton.clicked.connect(self.applyChanges)
        self.ui.tableWidget.setColumnCount(3)
        self.ui.tableWidget.setRowCount(len(self.parent.stor.file_parameters) - 3)
        for row, fp in enumerate(self.parent.stor.file_parameters[3:]):
            item = QtWidgets.QTableWidgetItem(fp)
            self.ui.tableWidget.setVerticalHeaderItem(row, item)

        for col, headerItem in enumerate(self.header):
            item = QtWidgets.QTableWidgetItem(headerItem)
            self.ui.tableWidget.setHorizontalHeaderItem(col, item)
        perms = [self.parent.selectedPerm_owner.decode('utf-8', 'backslashreplace'),
                 self.parent.selectedPerm_group.decode('utf-8', 'backslashreplace'),
                 self.parent.selectedPerm_other.decode('utf-8', 'backslashreplace')]
        for row, perm in enumerate(perms):
            for col in range(len(self.header)):
                item = QtWidgets.QTableWidgetItem()
                item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                if perm[col] == '-':
                    item.setCheckState(QtCore.Qt.Unchecked)
                else:
                    item.setCheckState(QtCore.Qt.Checked)
                self.ui.tableWidget.setItem(row, col, item)

        for g in self.parent.stor.groups:
            self.ui.comboBox.addItem(g.decode('utf-8', 'backslashreplace'))
        self.ui.comboBox.setCurrentText(self.parent.selectedGroup.decode('utf-8', 'backslashreplace'))

    def applyChanges(self):
        perms = []
        recursive = (self.ui.checkBox.checkState() == QtCore.Qt.Checked)
        for row in range(self.ui.tableWidget.rowCount()):
            perm = []
            for col, l in enumerate(self.letters):
                item = self.ui.tableWidget.item(row, col)
                if item.checkState() == QtCore.Qt.Unchecked:
                    perm.append('-')
                else:
                    perm.append(l)
            perm = str(self.perm_str_to_props(''.join(perm)))
            perms.append(perm)
        perms = bytes(''.join(perms), 'utf-8')
        g = bytes(self.ui.comboBox.currentText(), 'utf-8')
        retval = self.parent.stor.chgrp(self.parent.selectedFilename, g, recursive)
        if retval[0] != 0:
            self.parent.ui.lineEdit.setText(retval[1].decode('utf-8', 'backslashreplace'))
        else:
            retval = self.parent.stor.chmod(self.parent.selectedFilename, perms, recursive)
            if retval[0] != 0:
                self.parent.ui.lineEdit.setText(retval[1].decode('utf-8', 'backslashreplace'))
            else:
                self.close()
                self.parent.ui.lineEdit.setText('Obtaining information about directory. Please wait.')
                self.parent.ui.lineEdit.repaint()
                self.parent.currentFiles()

    def perm_str_to_props(self, perm_string):
        val = 0
        if perm_string[0] != '-':
            val += 4
        if perm_string[1] != '-':
            val += 2
        if perm_string[2] != '-':
            val += 1
        return val

    def props_to_perm_str(self, val):
        if val >= 4:
            val -= 4
            perm_string_0 = 'r'
        else:
            perm_string_0 = '-'
        if val >= 2:
            val -= 2
            perm_string_1 = 'w'
        else:
            perm_string_1 = '-'
        if val >= 1:
            val -= 1
            perm_string_2 = 'x'
        else:
            perm_string_2 = '-'
        perm_string = perm_string_0 + perm_string_1 + perm_string_2
        return perm_string

class LoginWidget(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(__class__, self).__init__(parent)
        self.ui = uic.loadUi(base_path + 'login.ui', self)
        self.parent = parent
        self.ui.pushButton.clicked.connect(self.login)

    def login(self):
        username = self.ui.lineEdit_username.text().strip()
        password = self.ui.lineEdit_password.text().strip()
        self.ui.parent.init(username, password)

    def connectionFailed(self):
        self.ui.lineEdit_debug.setText('Connection Failed. Try again.')

class GUI(QtWidgets.QMainWindow):
    def __init__(self, **kwargs):
        QtWidgets.QDialog.__init__(self)
        self.loginWidget = LoginWidget(self)
        self.loginWidget.show()

    def init(self, username, password):
        self.stor = ItetStor(HOSTNAME, username, password)
        if self.stor.connection_failed:
            self.loginWidget.connectionFailed()
            return
        else:
            self.loginWidget.close()

        # Set up the user interface from Designer.
        self.ui = uic.loadUi(base_path + 'mainWindow.ui')
        self.ui.show()

        self.ui.tableWidget.setColumnCount(len(self.stor.file_parameters))
        self.ui.tableWidget.setRowCount(1)
        for col, fp in enumerate(self.stor.file_parameters):
            item = QtWidgets.QTableWidgetItem(fp)
            self.ui.tableWidget.setHorizontalHeaderItem(col, item)

        self.currentFiles()
        self.ui.tableWidget.cellDoubleClicked.connect(lambda row, col: self.enterDirectory(row, col))
        self.ui.pushButton_topLevel.clicked.connect(self.topLevel)
        self.ui.pushButton_edit.clicked.connect(self.editEntry)

    def editEntry(self):
        selectedIndex = self.ui.tableWidget.selectedIndexes()
        if not selectedIndex:
            return

        selectedIndex = selectedIndex[0].row()
        filename = bytes(self.ui.tableWidget.item(selectedIndex, 0).text(), 'utf-8')
        filename_idx = self.filenames.index(filename)
        self.selectedFilename = filename
        self.selectedOwner = self.owners[filename_idx]
        self.selectedGroup = self.groups[filename_idx]
        self.selectedPerm_owner = self.perm_owner[filename_idx]
        self.selectedPerm_group = self.perm_group[filename_idx]
        self.selectedPerm_other = self.perm_other[filename_idx]
        self.editWidget = EditWidget(self)
        self.editWidget.show()

    def currentFiles(self):
        filenames_with_dir = self.stor.ls
        self.owners = self.stor.owner
        self.groups = self.stor.group
        self.perm_owner, self.perm_group, self.perm_other = self.stor.permission
        self.perm_owner = self.perm_owner
        self.perm_group = self.perm_group
        self.perm_other =self.perm_other
        self.filenames = [fn.rsplit(b'/', maxsplit=1)[-1] for fn in filenames_with_dir]
        self.ui.tableWidget.setRowCount(len(self.filenames))
        for row, (fn, own, gr, po, pg, pot) in enumerate(zip(self.filenames, self.owners, self.groups, self.perm_owner, self.perm_group, self.perm_other)):
            item = QtWidgets.QTableWidgetItem(fn.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 0, item)
            item = QtWidgets.QTableWidgetItem(own.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 1, item)
            item = QtWidgets.QTableWidgetItem(gr.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 2, item)
            item = QtWidgets.QTableWidgetItem(po.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 3, item)
            item = QtWidgets.QTableWidgetItem(pg.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 4, item)
            item = QtWidgets.QTableWidgetItem(pot.decode('utf-8', 'backslashreplace'))
            self.ui.tableWidget.setItem(row, 5, item)
        self.ui.lineEdit.setText('')


    def enterDirectory(self, row, col):
        if col != 0:
            self.ui.lineEdit.setText('Double Click first column to enter directory.')
            return
        self.ui.lineEdit.setText('Obtaining information about directory. Please wait.')
        self.ui.lineEdit.repaint()
        self.update()
        item = self.ui.tableWidget.item(row, col)
        directory = bytes(item.text(), 'utf-8')
        retval = self.stor.cd(directory)
        if retval[0] != 0:
            self.ui.lineEdit.setText(retval[1].decode('utf-8', 'backslashreplace'))
        else:
            self.currentFiles()

    def topLevel(self):
        self.ui.lineEdit.setText('Obtaining information about directory. Please wait.')
        self.ui.lineEdit.repaint()
        retval = self.stor.cd(b'..')
        if retval[0] != 0:
            self.ui.lineEdit.setText(retval[1].decode('utf-8', 'backslashreplace'))
        else:
            self.currentFiles()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = GUI(hostname=HOSTNAME)
    sys.exit(app.exec_())
