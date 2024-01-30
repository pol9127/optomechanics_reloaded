from __future__ import division, unicode_literals

from .roboclaw import Roboclaw
from time import sleep
import os
from PyQt4 import QtCore, QtGui, uic


class ValveControl(object):
    """Class representing the Roboclaw for the valve control.

    This class allows control of the valves using Roboclaw.
    """

    motor_settings = [{'accel': 200, 'decel': 0, 'speed': 1000},
                      {'accel': 200, 'decel': 0, 'speed': 1000}]

    def __init__(self, port, baudrate=115200, address=128,
                 verbose=False):
        """Initialize serial connection to Arduino.

        Parameters
        ----------
        port : str or int
            Identifier for the port Roboclaw is connected to. A list
            of port names may be retrieved using
            `serial.tools.list_ports`.
        baudrate : int, optional
        address : int, optional
            Address of the Roboclaw. Refer to the Roboclaw manual for
            details.
        """
        self.address = address
        self.verbose = verbose

        self.roboclaw = Roboclaw()
        self.roboclaw.Open(port, baudrate)

    def __del__(self):
        self.close()

    def open(self):
        """Open serial connection to Roboclaw."""
        if not self.roboclaw.ser.isOpen():
            self.roboclaw.ser.open()

    def close(self):
        """Close serial connection to Roboclaw."""
        if self.roboclaw.ser.isOpen():
            self.roboclaw.ser.close()

    def configure_motor(self, motor_number, acceleration=None,
                        deceleration=None, speed=None):
        assert motor_number in (0, 1)

        if acceleration is not None:
            self.motor_settings[motor_number]['accel'] = acceleration

        if deceleration is not None:
            self.motor_settings[motor_number]['decel'] = deceleration

        if speed is not None:
            self.motor_settings[motor_number]['speed'] = speed

    def drive_motor_position(self, motor_number, position,
                             clear_buffer=1):
        """Drive specified motor to valve position.

        Parameters
        ----------
        motor_number : int
            Identifier for the motor. Must be in (0,1).
        position : int
            New position of the motor.
        clear_buffer : bool, optional
            If True, this command stops any previous motor commands and
            executes this most recent one.
        """
        assert motor_number in (0, 1)

        start_position = self.current_motor_position(motor_number)

        if motor_number is 0:
            self.roboclaw.SpeedAccelDeccelPositionM1(
                self.address,
                self.motor_settings[motor_number]['accel'],
                self.motor_settings[motor_number]['speed'],
                self.motor_settings[motor_number]['decel'],
                position, clear_buffer)
        elif motor_number is 1:
            self.roboclaw.SpeedAccelDeccelPositionM2(
                self.address,
                self.motor_settings[motor_number]['accel'],
                self.motor_settings[motor_number]['speed'],
                self.motor_settings[motor_number]['decel'],
                position, clear_buffer)

        # Check if the motor is running in the right direction. In
        # some cases, the motor runs into the wrong direction. In
        # this case it will get immediately stopped and the command
        # is repeated.
        if clear_buffer:
            sleep(0.1)
            if (abs(position -
                        self.current_motor_position(motor_number))
                    - 10 > abs(position - start_position)):
                if self.verbose:
                    print('WARNING: Motor ran into the wrong '
                          'direction.')
                if motor_number is 0:
                    self.roboclaw.DutyM1(self.address, 0)
                elif motor_number is 1:
                    self.roboclaw.DutyM2(self.address, 1)
                sleep(0.1)
                if self.verbose:
                    print('INFO: Motor was stopped at '
                          'position {}.'.format(
                        self.current_motor_position(motor_number)))
                self.drive_motor_position(motor_number, position, 1)


    def current_motor_position(self, motor_number):
        """Return current position of the motor.

        Parameters
        ----------
        motor_number : int
            Identifier for the motor. Must be in (0,1).
        """
        assert motor_number in (0, 1)

        if motor_number is 0:
            return self.roboclaw.ReadEncM1(self.address)[1]
        if motor_number is 1:
            return self.roboclaw.ReadEncM2(self.address)[1]

    def reset_position_value(self, motor_number, position=0):
        """Reset the encoder value of the motor to a given value.

        According to an email from Nathan Scherdin from Ion MC:
        You need to stop the motor(not run the PID). To do that send
        a M1Duty with 0 duty and M2Duty with 0 duty (for motor1 and
        motor2). This will disable using the PID. Then zero the
        encoder. Then send a new position command with position set
        to 0. This will reactivate the PID but since the encoder is
        already at 0 the motor should not move.

        Parameters
        ----------
        motor_number : int
            Identifier for the motor. Must be in (0,1).
        position : int, optional
            New position value to be set. Default is 0.
        """
        if motor_number is 0:
            self.roboclaw.DutyM1(self.address, 0)
            self.roboclaw.SetEncM1(self.address, position)
            self.roboclaw.SpeedAccelDeccelPositionM1(
                self.address,
                self.motor_settings[motor_number]['accel'],
                self.motor_settings[motor_number]['speed'],
                self.motor_settings[motor_number]['decel'],
                position, 1)
        elif motor_number is 1:
            self.roboclaw.DutyM2(self.address, 0)
            self.roboclaw.SetEncM2(self.address, position)
            self.roboclaw.SpeedAccelDeccelPositionM2(
                self.address,
                self.motor_settings[motor_number]['accel'],
                self.motor_settings[motor_number]['speed'],
                self.motor_settings[motor_number]['decel'],
                position, 1)

    def open_gui(self, update_interval=0.5):
        self.gui_app = QtGui.QApplication.instance()
        if self.gui_app is None:
            from sys import argv
            self.gui_app = QtGui.QApplication(argv)
        self.gui = GUI(self, update_interval)
        self.gui.show()


class GUI(QtGui.QMainWindow):
    def __init__(self, valve_control, update_interval=1):
        super(GUI, self).__init__()
        ui_file_name = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'roboclaw_gui.ui')
        uic.loadUi(ui_file_name, self)

        self.valve_control = valve_control

        self.update_values()

        self.M1_pos_spinBox.setValue(
                self.valve_control.current_motor_position(0))
        self.M2_pos_spinBox.setValue(
                self.valve_control.current_motor_position(1))

        self.M1_pos_spinBox.setSingleStep(100)
        self.M2_pos_spinBox.setSingleStep(100)
        self.M1_pos_slider.setSingleStep(100)
        self.M2_pos_slider.setSingleStep(100)
        self.M1_pos_slider.setPageStep(1000)
        self.M2_pos_slider.setPageStep(1000)

        #self.M1_pos_spinBox.editingFinished.connect(self.move_M1)
        self.M1_pos_slider.sliderReleased.connect(self.move_M1)
        self.M1_pos_slider.valueChanged.connect(self.move_M1)
        #self.M2_pos_spinBox.editingFinished.connect(self.move_M2)
        self.M2_pos_slider.sliderReleased.connect(self.move_M2)
        self.M2_pos_slider.valueChanged.connect(self.move_M2)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_values)

        self.update_interval(update_interval)
        self.update_timer(True)

        self.show()

    def update_interval(self, time):
        self.timer.setInterval(int(time*1000))

    def update_timer(self, state):
        if state:
            self.timer.start()
        else:
            self.timer.stop()

    def update_values(self):
        from numpy import array

        pos_M1 = self.valve_control.current_motor_position(0)
        pos_M2 = self.valve_control.current_motor_position(1)
        min_max_1 = array(self.valve_control.roboclaw.ReadM1PositionPID(
                self.valve_control.address)[6:8]).astype('int32')
        min_max_2 = array(self.valve_control.roboclaw.ReadM2PositionPID(
                self.valve_control.address)[6:8]).astype('int32')

        self.M1_act_pos_spinBox.setValue(pos_M1)
        self.M2_act_pos_spinBox.setValue(pos_M2)
        self.M1_pos_spinBox.setRange(*min_max_1)
        self.M1_pos_slider.setRange(*min_max_1)
        self.M2_pos_spinBox.setRange(*min_max_2)
        self.M2_pos_slider.setRange(*min_max_2)

    def move_M1(self, x=0):
        if not self.M1_pos_slider.isSliderDown():
            self.valve_control.drive_motor_position(
                    0, self.M1_pos_spinBox.value())

    def move_M2(self, x=0):
        if not self.M2_pos_slider.isSliderDown():
            self.valve_control.drive_motor_position(
                    1, self.M2_pos_spinBox.value())

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
