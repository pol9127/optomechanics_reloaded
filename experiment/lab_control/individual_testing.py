if __name__ == '__main__':
    from PyQt5 import QtWidgets
    from optomechanics.experiment.lab_control.scope import *
    from optomechanics.experiment.lab_control.device import *
    import sys
    app = QtWidgets.QApplication(sys.argv)
    myScope = GaGe('default')
    # connection_settings = {'port': 7,
    #                        'baudrate': 9600,
    #                        'timeout': 0.1}

    # myGauge = Gauge('default', connection_settings)
    prog = myScope.ControlWidget()

    # prog = myGauge.ControlWidget()
    prog.show()
    sys.exit(app.exec_())
