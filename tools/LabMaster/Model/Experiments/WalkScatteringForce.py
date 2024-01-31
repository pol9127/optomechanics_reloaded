from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
from os import path
from time import sleep, time
import numpy as np
import json

SCATTERING_FORCE_WALK_FILE = 'scattering_force_walk_settings.json'


class WalkScatteringForce(BaseExperiment):

    name = 'walk scattering force'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[KoherasLaserViewModel])

    def run(self):
        laser = self.tools.laser

        start_power = 1000
        end_power = 280
        power_step = 5
        time_step = 1

        print('Loading settings:')
        if path.isfile(SCATTERING_FORCE_WALK_FILE):
            with open(SCATTERING_FORCE_WALK_FILE, 'r') as file:
                data = json.loads(file.read())
                start_power = data['start_power']
                end_power = data['end_power']
                power_step = data['power_step']
                time_step = data['time_step']
        else:
            print('Unable to load data... using default settings.')
        print('\n***************************************************')
        print('*                    SETTINGS                      ')
        print('***************************************************')
        print('start power: {} mW'.format(start_power))
        print('end power: {} mW'.format(end_power))
        print('power step: {} mW'.format(power_step))
        print('time step: {} s'.format(time_step))
        print('***************************************************\n')

        print('starting experiment:')
        print('starting measurement...')
        start_time = time()
        for power in np.flip(np.arange(end_power, start_power + power_step, power_step)):
            timestamp = time() - start_time
            print('{};setting power: {} mW'.format(timestamp, power))
            laser.power = power
            laser.set_on_laser()
            sleep(time_step)
        print('Experiment finished.')


