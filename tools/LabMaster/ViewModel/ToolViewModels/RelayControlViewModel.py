from QtModularUiPack.ViewModels import BaseViewModel
from ArduinoRelayControl.ArduinoRelayControl import ArduinoRelayControl
import json
import os

RELAY_SETTINGS_FILE = 'relay_settings.json'


class RelayControlViewModel(BaseViewModel):

    name = 'relay'

    @property
    def port(self):
        """
        Gets the serial port of the device
        """
        return self._relay.port

    @port.setter
    def port(self, value):
        """
        Sets the serial port of the device
        """
        self._relay.port = value
        self.notify_change('port')

    @property
    def connect_button_text(self):
        return self._connect_button_text

    @connect_button_text.setter
    def connect_button_text(self, value):
        self._connect_button_text = value
        self.notify_change('connect_button_text')

    @property
    def enabled(self):
        return self._relay.enabled

    @enabled.setter
    def enabled(self, value):
        self._relay.enabled = value
        self.notify_change('enabled')

    def __init__(self):
        super().__init__()
        self._relay = ArduinoRelayControl(None)
        self._connect_button_text = 'connect'
        self.load_configuration()

    def __del__(self):
        self._relay.close()

    def connect_disconnect(self):
        if self._relay.connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """
        Connect to the device
        """
        try:
            self._relay.open()
            self.connect_button_text = 'disconnect'
            self.notify_change('enabled')
        except Exception as e:
            print('Unable to connect to Arduino. {}'.format(e))

    def disconnect(self):
        """
        Disconnects the device
        """
        self._relay.close()
        self.connect_button_text = 'connect'

    def save_configuration(self):
        """
        Saves the configuration to a json file
        """
        data = {'port': self.port}
        with open(RELAY_SETTINGS_FILE, 'w') as file:
            file.write(json.dumps(data))

    def load_configuration(self):
        """
        Loads the configuration from a json file
        """
        if os.path.isfile(RELAY_SETTINGS_FILE):
            with open(RELAY_SETTINGS_FILE, 'r') as file:
                data = json.loads(file.read())
                self.port = data['port']
                if self.port is not None:
                    self.connect()
