from PyQt5 import QtWidgets, QtCore
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager
from IPython.lib import guisupport

class BackgroundClose(QtCore.QThread):
    busy = False
    mutex = QtCore.QMutex()
    def __init__(self, process):
        self.process = process
        QtCore.QThread.__init__(self)

    def __del__(self):
        self.wait()

    def run(self):
        self.mutex.lock()
        self.busy = True
        self.mutex.unlock()
        self.process()
        self.mutex.lock()
        self.busy = False
        self.mutex.unlock()


class CustomDockWidget(QtWidgets.QDockWidget):
    def __init__(self, scope_description, *__args):
        super(QtWidgets.QDockWidget, self).__init__(*__args)
        self.scope_description = scope_description
        self.main = __args[0]
        self.background_closer = None

    def closeEvent(self, *args, **kwargs):
        super(QtWidgets.QDockWidget, self).closeEvent(*args, **kwargs)
        print('disconnecting from', self.main.connected_devices[self.scope_description[0]][self.scope_description[1]])

        self.background_closer = BackgroundClose(self.widget().close)
        self.background_closer.start()
        del self.main.connected_devices[self.scope_description[0]][self.scope_description[1]]
        print(self.main.connected_devices)


class QIPythonWidget(RichJupyterWidget):
    """ Convenience class for a live IPython console widget. We can replace the standard banner using the customBanner argument"""
    def __init__(self,customBanner=None,*args,**kwargs):
        if customBanner!=None: self.banner=customBanner
        super(QIPythonWidget, self).__init__(*args,**kwargs)
        self.kernel_manager = kernel_manager = QtInProcessKernelManager()
        kernel_manager.start_kernel()
        kernel_manager.kernel.gui = 'qt4'
        self.kernel_client = kernel_client = self._kernel_manager.client()
        kernel_client.start_channels()

        def stop():
            kernel_client.stop_channels()
            kernel_manager.shutdown_kernel()
            guisupport.get_app_qt4().exit()
        self.exit_requested.connect(stop)

    def pushVariables(self,variableDict):
        """ Given a dictionary containing name / value pairs, push those variables to the IPython console widget """
        self.kernel_manager.kernel.shell.push(variableDict)
    def clearTerminal(self):
        """ Clears the terminal """
        self._control.clear()
    def printText(self,text):
        """ Prints some plain text to the console """
        self._append_plain_text(text)
    def executeCommand(self,command):
        """ Execute a command in the frame of the console widget """
        self._execute(command,False)

