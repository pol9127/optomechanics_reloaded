from __future__ import division, print_function, unicode_literals

from experiment.controller.pid import PIDController
from experiment.controller.bakeout import BakeOutControl
from experiment.controller.pressure import PressureControl


class PumpDown(object):

    pump_down_active = False

    def __init__(self, pressure_function, motor_step_function,
                 pressure_limit=2e1, threshold_percentage=0.1,
                 loop_interval=5):
        self.pressure_function = pressure_function
        self.motor_step_function = motor_step_function

        self.pressure_limit = pressure_limit
        self.threshold_percentage = threshold_percentage
        self.loop_interval = loop_interval

    def start_pump_down(self):
        pass

    def stop_pump_down(self):
        pass

    def pump_down_loop(self):
        pass
