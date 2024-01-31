from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
from ViewModel.ToolViewModels.TBS2104ViewModel import TBS2104ViewModel
from datetime import datetime
import traceback
import os
import numpy as np


def ensure_connection(device):
    if not device.connected:
        device.connect()

        if not device.connected:
            raise Exception('Unable to connect to "{}".'.format(device.name))


def get_name():
    now = datetime.now()
    return 'Trapping_{}.{}.{}_{}-{}-{}'.format(now.day, now.month, now.year, now.hour, now.minute, now.second)


class TrappingExperiment(BaseExperiment):

    name = 'Trapping Experiment'
    results_path = 'Z:/shared/Master Thesis/Experiments/TrappingExperiment/'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[KoherasLaserViewModel, TBS2104ViewModel])
        self.laser_current = 6.0

    def run(self):
        print('Starting trapping experiment...')

        run = 1
        question_text = 'Do you want to measure{}?'
        experiment_name = get_name()

        results_folder = self.results_path + experiment_name

        try:
            laser = self.tools.laser
            scope = self.tools.tbs

            ensure_connection(laser)
            ensure_connection(scope)

            scope.select_all()
            laser.current_set_point = self.laser_current
            laser.set_on_laser()

            laser.emission_on = True
            if not laser.emission_on:
                raise Exception('Laser not running.')

            os.mkdir(results_folder)

            while self.question(question_text.format('' if run == 1 else ' again', 'Trapping Experiment')):
                if not laser.emission_on:
                    laser.emission_on = True

                print('To drop the particles, press the "trig" button on the Signal Generator.')
                x, y = scope.single_and_measure()
                data = np.vstack([x, y]).transpose()
                filename = '{}/voltage_data_run{}.h5'.format(results_folder, run)
                print('saving data...')
                self.save_h5(filename, data)
                print('data saved in "{}".'.format(filename))

                if not self.question('Do you want to keep the laser on?'):
                    laser.emission_on = False

                run += 1

        except Exception as e:
            print('Experiment failed. {}{}'.format(traceback.print_exc(), e))
        finally:
            laser.disconnect()
            scope.disconnect()

        print('done.')
