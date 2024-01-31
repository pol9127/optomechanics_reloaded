from QtModularUiPack.ViewModels import BaseContextAwareViewModel
from QtModularUiPack.ModularApplications.ToolFrameViewModels.experiment_frame_view_model import ExperimentOverviewViewModel
from ArduinoTrigger.ArduinoTrigger import ArduinoTrigger
from QtModularUiPack.Framework import is_non_strict_type
import os
import json


class ArduinoTriggerViewModel(BaseContextAwareViewModel):
    """
    Data context for Arduino trigger User interface
    """

    _Arduino_trigger_config_file_ = 'ArduinoTriggerConfig.json'

    @property
    def port(self):
        """
        Gets the serial port of the device
        """
        return self._arduino_trigger.port

    @port.setter
    def port(self, value):
        """
        Sets the serial port of the device
        """
        self._arduino_trigger.port = value
        self.notify_change('port')

    @property
    def experiment_index(self):
        return self._experiment_index

    @experiment_index.setter
    def experiment_index(self, value):
        self._experiment_index = value
        self.notify_change('experiment_index')

    @property
    def connected(self):
        """
        True if the device is connected
        """
        return self._arduino_trigger.connected

    @property
    def experiment_available(self):
        """
        True if experiment frame is open in the application.
        :return: true or false
        """
        for vm in self.other_data_contexts:
            if is_non_strict_type(type(vm), ExperimentOverviewViewModel):
                return True
        return False

    @property
    def trigger_experiment(self):
        """
        Gets if the trigger starts the experiment
        :return:
        """
        return self._trigger_experiment and self.experiment_available

    @trigger_experiment.setter
    def trigger_experiment(self, value):
        """
        Sets if the trigger start the experiment
        :param value: true or false
        """
        self._trigger_experiment = value
        self.notify_change('trigger_experiment')

    def __init__(self):
        super().__init__()
        self._arduino_trigger = ArduinoTrigger(None)
        self._arduino_trigger.triggered.append(self.trigger)
        self._trigger_experiment = False
        self._experiment_index = 0
        self._experiment_handlers = list()
        self.load_configuration()

    def _other_data_context_added_(self, data_context):
        super()._other_data_context_added_(data_context)
        if is_non_strict_type(type(data_context), ExperimentOverviewViewModel):
            self._experiment_handlers.append(data_context)
            self.notify_change('trigger_experiment')
            self.notify_change('experiment_available')

    def _other_data_context_removed_(self, data_context):
        super()._other_data_context_removed_(data_context)
        if is_non_strict_type(type(data_context), ExperimentOverviewViewModel):
            if data_context in self._experiment_handlers:
                self._experiment_handlers.remove(data_context)
            self.notify_change('trigger_experiment')
            self.notify_change('experiment_available')

    def __del__(self):
        self._arduino_trigger.close()

    def connect(self):
        """
        Connect to the device
        """
        try:
            self._arduino_trigger.open()
        except Exception as e:
            print('Unable to connect to Arduino. {}'.format(e))

    def disconnect(self):
        """
        Disconnects the device
        """
        self._arduino_trigger.close()

    def trigger(self):
        """
        Triggers the device (and optionally experiments)
        """
        if not self.connected:
            self.connect()

        if self.connected:
            self._arduino_trigger.trigger()
            print('trigger')

            if self.trigger_experiment:
                for experiment_handler in self._experiment_handlers:
                    if self.experiment_index >= 0 and self.experiment_index < len(experiment_handler.experiments):
                        experiment_handler.experiments[self.experiment_index].run()

    def save_configuration(self):
        """
        Saves the configuration to a json file
        """
        data = {'port': self.port, 'trigger_experiment': self._trigger_experiment, 'experiment_index': self._experiment_index}
        with open(self._Arduino_trigger_config_file_, 'w') as file:
            file.write(json.dumps(data))

    def load_configuration(self):
        """
        Loads the configuration from a json file
        """
        if os.path.isfile(self._Arduino_trigger_config_file_):
            with open(self._Arduino_trigger_config_file_, 'r') as file:
                data = json.loads(file.read())
                self.port = data['port']
                self.trigger_experiment = data['trigger_experiment']
                self.experiment_index = data['experiment_index']
                if self.port is not None:
                    self.connect()
