from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):

    _count = 0
    _new = 'New'
    _cascade = 'Cascade'
    _tiled = 'Tiled'

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        
        self._splitter = QSplitter(Qt.Vertical)
        self._mdi = QMdiArea()
        self.setCentralWidget(self._mdi)
        self._bar = self.menuBar()

        self._file = self._bar.addMenu('File')
        self._file.addAction(self._new)
        self._file.addAction(self._cascade)
        self._file.addAction(self._tiled)
        self._file.triggered[QAction].connect(self._action_)
        self.setWindowTitle('Lab Master')

    def _action_(self, q):
        print('triggered')

        choice = q.text()
        if choice == self._new:
            MainWindow._count += 1
            sub_window = QMdiSubWindow()
            sub_window.setWidget(QTextEdit())
            sub_window.setWindowTitle('Sub Window {}'.format(MainWindow._count))
            self._mdi.addSubWindow(sub_window)
            sub_window.show()
        elif choice == self._cascade:
            self._mdi.cascadeSubWindows()
        elif choice == self._tiled:
            self._mdi.tileSubWindows()
