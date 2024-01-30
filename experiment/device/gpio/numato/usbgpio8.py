import serial
from time import sleep


class USBGPIO8:
    """Class that represents the Numato USBGPIO 8 Channel device."""
    serial_port = None
    state_conversion = {0: b"0", 1: b"1"}
    gpio_num_conversion = {0: b"0", 1: b"1", 2: b"2", 3: b"3", 4: b"4", 5: b"5", 6: b"6", 7: b"7"}
    def __init__(self, serPort):
        '''To initalize the device the portName has to be given.
        On windows this should be of the form: "COM1"
        On Linux something like: "/dev/ttyUSB0"'''
        self.serial_port = serial.Serial(portName, 19200, timeout=1)

    def __del__(self):
        """Upon destruction of the class the connection is opened again. If the code crashes and cannot delte the
        object properly the connection might get stuck in a blocked state."""
        if self.serial_port is not None:
            self.serial_port.close()

    def get_gpio(self, gpio_num):
        """Function to return the state of a given GPIO Pin.
        gpioNum: Integer representing the gpio port of interest"""
        gpio_num = self.gpio_num_conversion[gpio_num]
#        self.serial_port.write(b"gpio read 2\r")
        self.serial_port.flushInput()
        self.serial_port.write(b"gpio read " + gpio_num + b"\r")
        response = self.serial_port.readall()
        return int(response.split(b'\n\r')[-2])

    def set_gpio(self, gpio_num, state):
        """Function to set the state of a gpio port
        gpioNum: Integer representing the gpio port of interest
        state: int or bool, new state the gpio port should have"""
        if isinstance(state, int):
            state = bool(state)

        gpio_num = self.gpio_num_conversion[gpio_num]
        self.serial_port.flushInput()
        if state:
            self.serial_port.write(b"gpio set " + gpio_num + b"\r")
        else:
            self.serial_port.write(b"gpio clear " + gpio_num + b"\r")


    def wait_trigger(self, gpio_num, trigger_state=1, refresh_rate=0.001):
        """Function used to trigger software. The function waits for a GPIO pin to obtain a desired state.
        The refresh_rate gives the rate at which the state is being checked. If the rate is faster than the
        communication this might run into a problem
        gpioNum: Integer representing the gpio port of interest
        trigger_state: Integer (0 or 1) the state upon which to trigger
        refresh_rate: float, seconds between gpio-read events"""
        while True:
            sleep(refresh_rate)
            state = self.get_gpio(gpio_num)
            if (trigger_state and state) or (not trigger_state and not state):
                print("I triggered!!")
                return


if __name__ == '__main__':
    # Here we connect to the comport
    portName = 'COM8'
    usbgpio = USBGPIO8(portName)
    # Now we set the GPIO Port 0 to High
    usbgpio.set_gpio(0, 1)
    # Wait for command to execute
    sleep(0.1)
    # Now we wait for GPIO Pin 1 to get to a high state. When we shortcut pins 0 and 1 this should trigger.
    usbgpio.wait_trigger(gpio_num=1, trigger_state=1)
