from SerialDevice import SerialDevice
from time import sleep, time
from threading import Thread


class ArduinoTrigger(SerialDevice):
    """
    This is the python control for a very simple Arduino setup where a serial command causes the board to quickly set
    one of its port to high as a trigger.
    """

    def __init__(self, port):
        super().__init__(port)
        self.timeout = 3
        self.triggered = list()

    def _trigger_watcher_(self):
        while self.connected:
            if self.read() == b'T':
                for callback in self.triggered:
                    callback()

    def open(self):
        answer = super().open()

        start_time = time()
        while answer != b'OK':
            answer = self.query('R')
            sleep(0.1)
            if time() - start_time > self.timeout:
                raise TimeoutError('Timeout while trying to connect to Arduino.')
        print('Connected to arduino trigger.')
        watcher_thread = Thread(target=self._trigger_watcher_, daemon=True)
        watcher_thread.start()

    def trigger(self):
        if self.connected:
            self.write('T')
