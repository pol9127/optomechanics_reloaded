import visa
from pyvisa import errors


class VisaDevice(object):
    """
    This is a base class for network capable VISA devices
    """

    @property
    def host(self):
        """
        Gets the hostname or ip address of the device
        """
        return self._host

    @host.setter
    def host(self, value):
        """
        Sets the hostname or ip address of the device
        """
        if self.instance is not None:
            self.instance.close()
            self.instance = None
        self._host = value

    @property
    def idn(self):
        """
        Gets the idn of the device
        """
        return self.instance.query('*IDN?').replace('\n', '')

    def query(self, message):
        """
        Sends query to the device and returns the response
        :param message: query
        :return: reply from the device
        """
        return self._call_backend_(self.instance.query, message)

    def write(self, message):
        """
        Sends command to the device
        :param message: command
        """
        self._call_backend_(self.instance.write, message)

    def open(self):
        """
        Opens communication with the device
        """
        if self.host is None:
            return
        self.instance = self._rm.open_resource('TCPIP::{}::INSTR'.format(self.host))
        self.verify_instance(self.instance)

    def close(self):
        """
        Closes communication with the device
        """
        if self.instance is not None:
            self.instance.close()
            self.instance = None

    def verify_instance(self, loaded_instance):
        """
        This method can be overwritten in child classes to check if the connected device matches given criteria
        :param loaded_instance: device instance
        """
        pass

    def _call_backend_(self, func, *args):
        """
        Call function in the PyVisa backend and handle connection errors.
        :param func: function to call
        :param args: arguments
        :return: return value
        """
        if self.instance is not None:
            try:
                return func(*args)
            except errors.VisaIOError as e:
                if e.error_code == errors.VI_ERROR_CONN_LOST or e.error_code == errors.VI_ERROR_CLOSING_FAILED:
                    print('PyVisa: connection to device lost...')
                    if self.reconnect_on_connection_lost:
                        print('PyVisa: Attempting to restore connection to "{}"...'.format(self.host))
                        self.open()
                    else:
                        self.instance = None
                        print('PyVisa: No attempt was made to restore the connection.')
                else:
                    raise e
        else:
            print('PyVisa: No connection established.')

    def __init__(self, host=None, reconnect_on_connection_lost=True):
        self._host = host
        self._rm = visa.ResourceManager()
        self.instance = None
        self.reconnect_on_connection_lost = reconnect_on_connection_lost

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
