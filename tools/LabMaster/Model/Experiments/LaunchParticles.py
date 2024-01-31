from QtModularUiPack.Framework.Experiments import BaseExperiment
from ViewModel.ToolViewModels.PeakTech4055MVViewModel import PeakTech4055MVViewModel
from ViewModel.ToolViewModels.RelayControlViewModel import RelayControlViewModel
from ViewModel.ToolViewModels.KoherasLaserViewModel import KoherasLaserViewModel
from time import sleep
import json
import os

PARTICLE_LAUNCH_SETTINGS_FILE = 'particle_launch_settings.json'


class LaunchParticles(BaseExperiment):
    """
    This script serves the purpose of launching particles
    """

    name = 'launch particles'

    def __init__(self, tools):
        super().__init__(tools, required_tools=[PeakTech4055MVViewModel, RelayControlViewModel, KoherasLaserViewModel])

    def run(self):
        if not self.tools.laser.emission_on:
            self.message('Laser is not turned on.', 'Warning')
        else:
            signal_duration = 0.1
            if os.path.isfile(PARTICLE_LAUNCH_SETTINGS_FILE):
                with open(PARTICLE_LAUNCH_SETTINGS_FILE, 'r') as file:
                    data = json.loads(file.read())
                    signal_duration = data['signal_duration']

            signal_generator = self.tools.sig_gen
            relay = self.tools.relay
            if not signal_generator.output_enabled:
                signal_generator.output_enabled = True
            relay.enabled = True
            sleep(signal_duration)
            relay.enabled = False
