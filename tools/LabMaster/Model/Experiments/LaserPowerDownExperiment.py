from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
import json
import numpy as np


POWER_SETTING_FILE = 'trapping_power_settings.json'
POWER_THRESHOLD = 10


class LaserPowerDownExperiment(BaseExperiment):

    name = 'laser power down'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[KoherasLaserViewModel])

    def get_current_power_index(self, powers):
        idx = np.argmin(np.abs(powers - self.tools.laser.power))
        if abs(self.tools.laser.power - powers[idx]) > POWER_THRESHOLD:
            return -1
        return idx

    def run(self):
        if not self.tools.laser.connected:
            self.tools.laser.connect()

        data = None
        with open(POWER_SETTING_FILE, 'r') as file:
            data = json.loads(file.read())

        if data is None:
            print('Unable to get power setting.')
            return

        powers = np.array(data['power'])
        power_index = self.get_current_power_index(powers)

        setting = powers[0]
        if power_index != -1 and power_index < len(powers) - 1:
            setting = powers[power_index + 1]

        print('Setting power to: {} mW'.format(setting))
        self.tools.laser.power = setting
        self.tools.laser.set_on_laser()
