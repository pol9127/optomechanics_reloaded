from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
import json
import numpy as np
from time import sleep
from os import path


ATTENUATION_SETTING_FILE = 'attenuation_power_settings.json'
POWER_THRESHOLD = 10


class AttenuateAfterTrapping(BaseExperiment):

    name = 'attenuate after trapping'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[KoherasLaserViewModel])

    def run(self):
        laser = self.tools.laser
        if not self.tools.laser.connected:
            laser.connect()

        start_power = laser.get_fitted_power(laser.device_set_point)
        end_power = 300
        power_step = 10
        time_step = 0.5
        if path.isfile(ATTENUATION_SETTING_FILE):
            with open(ATTENUATION_SETTING_FILE, 'r') as file:
                data = json.loads(file.read())
                end_power = data['end_power']
                power_step = data['power_step']
                time_step = data['time_step']

        powers = np.flip(np.arange(end_power, start_power, power_step))
        for power in powers:
            laser.power = round(power, 2)
            laser.set_on_laser()
            sleep(time_step)

        print('Attenuating laser power...')

        print('Laser power now at: {} mW'.format(laser.get_fitted_power(laser.device_set_point)))

