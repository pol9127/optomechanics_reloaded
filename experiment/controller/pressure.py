from __future__ import division, print_function, unicode_literals

from .pid import PIDController
from time import sleep
from threading import Thread
from numpy import log10


class PressureControl(object):
    pid_active = False
    pid_thread = None

    def __init__(self, pressure_function, motor_position_function,
                 set_pressure=None, pid_step=2.0, k_p=-100, k_i=-1000,
                 k_d=0.0, min_position=0, max_position=100000,
                 init_position=0):
        self.pressure_function = pressure_function
        self.motor_position_function = motor_position_function

        self.pid_step = pid_step

        self.pid = PIDController(k_p, k_i, k_d,
                                 output_max=max_position,
                                 output_min=min_position,
                                 init_output=init_position)

        if set_pressure is not None:
            self.set_pressure = set_pressure

    @property
    def set_pressure(self):
        return 10**self.pid.set_point

    @set_pressure.setter
    def set_pressure(self, new_pressure):
        self.pid.set_point = log10(new_pressure)

    def start_pid(self, set_pressure=None):
        # type: float
        if set_pressure is not None:
            self.set_pressure = set_pressure

        if not self.pid_active:
            self.pid_active = True
            self.pid_thread = Thread(target=self.pid_loop)
            self.pid_thread.start()

    def stop_pid(self):
        if self.pid_active:
            self.pid_active = False
            self.pid_thread.join()

    def pid_loop(self):
        while self.pid_active:
            new_position = int(self.pid.update(
                    log10(self.pressure_function()), self.pid_step))

            self.motor_position_function(new_position)

            sleep(self.pid_step)
