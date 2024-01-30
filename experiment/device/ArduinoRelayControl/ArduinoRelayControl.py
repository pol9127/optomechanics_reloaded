from SerialDevice import SerialDevice
from time import sleep, time


class ArduinoRelayControl(SerialDevice):
    """
    This is the python control for a Arduino controlled relay
    """

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value
        self.write('E' if value else 'D')

    def __init__(self, port):
        super().__init__(port)
        self.timeout = 3
        self._enabled = False

    def open(self):
        answer = super().open()

        start_time = time()
        while answer != b'OK':
            answer = self.query('R')
            sleep(0.1)
            if time() - start_time > self.timeout:
                raise TimeoutError('Timeout while trying to connect to Arduino.')
        self._enabled = self.query('S') == b'ENABLED'
        self.command_pause = 0.02
        print('Connected to arduino relay control.')
