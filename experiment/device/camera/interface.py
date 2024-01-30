from PyQt5 import uic, QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel
import os
import sys
import cv2
import numpy as np

class Interface(QtWidgets.QMainWindow):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        QtWidgets.QDialog.__init__(self)

        # Set up the user interface from Designer.
        self.ui = uic.loadUi('MainWindow.ui')


        self.cap = cv2.VideoCapture(1)
        ret, frame = self.cap.read()
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

        label = QLabel(self)
        # label.setPixmap(QPixmap(qImg))
        pix = QPixmap(r'C:\Python\git\optomechanics\experiment\device\camera\test.jpg')
        print(pix.height())
        label.setPixmap(pix)
        # label.setText('Hello')
        self.ui.layout().addWidget(label)
        self.ui.show()

    def __del__(self):
        self.cap.release()
        # cv2.destroyAllWindows()

# class glWidget(QGLWidget):
#
#
#     def __init__(self, parent):
#         QGLWidget.__init__(self, parent)
#         self.setMinimumSize(640, 480)
#
#     def paintGL(self):
#
#
#         glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
#         glLoadIdentity()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Interface()
    sys.exit(app.exec_())
