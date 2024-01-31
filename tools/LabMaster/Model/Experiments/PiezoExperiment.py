from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
from ViewModel.ToolViewModels.TBS2104ViewModel import TBS2104ViewModel
from ViewModel.ToolViewModels.ArduinoTriggerViewModel import ArduinoTriggerViewModel
from datetime import datetime
from time import sleep
import os
import numpy as np
import traceback


def ensure_connection(device):
    if not device.connected:
        device.connect()

        if not device.connected:
            raise Exception('Unable to connect to "{}".'.format(device.name))


def get_name():
    now = datetime.now()
    return 'Piezo_{}.{}.{}_{}-{}-{}'.format(now.day, now.month, now.year, now.hour, now.minute, now.second)


class PiezoExperiment(BaseExperiment):
    name = 'Piezo acceleration experiment'
    results_path = 'Z:/shared/Master Thesis/Experiments/PiezoExperiment/'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[KoherasLaserViewModel, TBS2104ViewModel, ArduinoTriggerViewModel])
        self.laser_current = 1

    def run(self):
        store_data = False

        signal_trigger = self.tools.signal_trigger
        experiment_name = get_name()

        results_folder = self.results_path + experiment_name

        try:
            laser = self.tools.laser
            scope = self.tools.tbs
            old_state = scope.get_data_after_measurement
            scope.get_data_after_measurement = False

            ensure_connection(laser)
            ensure_connection(scope)

            scope.select_all()
            sleep(1)
            scope.selected_channel = 0
            laser.current_set_point = self.laser_current
            laser.set_on_laser()

            laser.emission_on = True
            if not laser.emission_on:
                raise Exception('Laser not running.')

            os.mkdir(results_folder)

            print('Starting measurement...')

            scope.single()  # start measurement
            sleep(1)
            #signal_trigger.trigger()    # trigger burst on piezo

            while scope.measurement_in_progress:
                sleep(0.1)

            laser.emission_on = False

            if store_data:
                scope.get_data_and_wait()
                x_ch1 = scope.x
                y_ch1 = scope.y
                sleep(1)

                scope.selected_channel = 1
                sleep(1)
                scope.get_data_and_wait()
                x_ch2 = scope.x
                y_ch2 = scope.y

                data_ch1 = np.vstack([x_ch1, y_ch1]).transpose()
                data_ch2 = np.vstack([x_ch2, y_ch2]).transpose()
                filename_ch1 = '{}/voltage_data_PD.h5'.format(results_folder)
                filename_ch2 = '{}/voltage_data_trigger.h5'.format(results_folder)
                print('saving data...')
                self.save_h5(filename_ch1, data_ch1)
                print('data saved in "{}".'.format(filename_ch1))
                print('saving data...')
                self.save_h5(filename_ch2, data_ch2)
                print('data saved in "{}".'.format(filename_ch2))

            scope.get_data_after_measurement = old_state

        except Exception as e:
            print('Experiment failed. {}{}'.format(traceback.print_exc(), e))

        print('done.')
