from __future__ import division, print_function, unicode_literals

from .pid import PIDController
from time import sleep
from threading import Thread


class BakeOutControl(object):
    # https://groups.google.com/d/msg/diy-pid-control/xffzfaK-XVc/AutiEa4CDjUJ

    pid_active = False
    pwm_active = False

    pid_thread = None
    pwm_thread = None

    def __init__(self, temperature_function, heater_function,
                 set_temperature, pid_step=2.0, k_p=10, k_i=0.05,
                 k_d=0.0, pwm_time=2.0):
        self.temperature_function = temperature_function
        self.heater_function = heater_function
        self.heating_rate = 0

        self.pid_step = pid_step
        self.pwm_time = pwm_time

        self.pid = PIDController(k_p, k_i, k_d,
                                 set_point=set_temperature,
                                 output_max=100, output_min=0)

    @property
    def set_temperature(self):
        return self.pid.set_point

    @set_temperature.setter
    def set_temperature(self, new_temperature):
        self.pid.set_point = new_temperature

    def start_pid_bake(self, set_temperature=None):
        # type: float
        if set_temperature is not None:
            self.set_temperature = set_temperature

        if not self.pwm_active:
            self.pwm_active = True
            self.pwm_thread = Thread(target=self.heater_pwm)
            self.pwm_thread.start()
        if not self.pid_active:
            self.pid_active = True
            self.pid_thread = Thread(target=self.pid_loop)
            self.pid_thread.start()

    def stop_pid_bake(self):
        if self.pid_active:
            self.pid_active = False
            self.pid_thread.join()
        if self.pwm_active:
            self.pwm_active = False
            self.pwm_thread.join()

    def start_pwm_bake(self, heating_rate=None):
        # type: float
        if heating_rate is not None:
            self.heating_rate = heating_rate

        if not self.pwm_active:
            self.pwm_active = True
            self.pwm_thread = Thread(target=self.heater_pwm)
            self.pwm_thread.start()

    def stop_pwm_bake(self):
        if self.pwm_active:
            self.pwm_active = False
            self.pwm_thread.join()

    def pid_loop(self):
        while self.pid_active:
            self.heating_rate = self.pid.update(
                self.temperature_function(), self.pid_step)/100

            sleep(self.pid_step)

    def heater_pwm(self):
        if type(self.heater_function) is not tuple:
            self.heater_function = (self.heater_function, )

        while self.pwm_active:
            for heater in self.heater_function:
                heater.write(1)

            sleep(self.heating_rate*self.pwm_time)

            for heater in self.heater_function:
                heater.write(0)

            sleep((1-self.heating_rate)*self.pwm_time)
