from subprocess import Popen, PIPE
from threading import Thread
from time import time, sleep


PROCESS_DEAD_STDIN_READTIME = 0.1
PROCESS_DEAD_TERMINATION_COUNT = 10


class ProcessHandler(object):
    """
    This process handler allows a subprocess in the form of an external application to be launched and supplied with
    command lines inputs continually. The main purpose is that the process stays open and and does not get terminated
    after each command.
    """

    @property
    def running(self):
        """
        Gets if the process is running
        """
        return self._running

    def __init__(self, process: str, verbose=True, eol='\r\n', response_wait_time=0.01, timeout=1, encoding='ascii'):
        """
        Initialize the process handler
        :param process: process name to be launched (i.e. path to executable)
        :param verbose: if set to true the return values of the process will be printed to the console
        :param eol: line feed for commands (used in write(), query() methods)
        :param response_wait_time: delay time after running command
        :param timeout: timeout to wait for a response from the process
        :param encoding: encoding of the commands
        """
        self._process = Popen([process], stdin=PIPE, stdout=PIPE)
        self.verbose = verbose
        self.eol = eol
        self.encoding = encoding
        self.response_wait_time = response_wait_time
        self.timeout = timeout
        self._output = None     # keep track of gathered output
        self._expect_output = False     # indicate to the reader thread if output has to be collected
        self._return_count = 1  # indicate to the reader thread how much output is expected
        self._response_index = 0    # return value index (used during return value collection)
        self._process_dead_count = 0    # counter to check if thread is dead
        self._running = True    # assume the process is running at the beginning
        self._reader_thread = Thread(target=self._read_worker_, daemon=True)    # initialize reader thread
        self._reader_thread.start()     # start reader thread

    def terminate(self):
        """
        Terminates the process
        """
        if self._process is not None:
            self._running = False   # end the reader thread
            self._process.terminate()   # terminate the process

    def write(self, command):
        """
        Write command to process input (assuming command line behaviour)
        :param command: command
        """
        try:
            self._process.stdin.write(bytes('{}{}'.format(command, self.eol), encoding=self.encoding))
            self._process.stdin.flush()     # needs to be done to execute the command
            sleep(self.response_wait_time)  # wait for a response
        except Exception as e:
            if self.verbose:
                print(e)

    def query(self, command, return_count=1):
        """
        Execute a command and wait for a defined number of responses
        :param command: command to execute
        :param return_count: number of expected responses (Exception thrown if response count not met)
        :return: list of responses
        """
        while self._expect_output:  # only one query at a time can be processed, wait for the last one to be completed
            sleep(0.01)

        self._output = list()   # reset response list
        self._return_count = return_count   # communicate the desired number of responses
        self._expect_output = True      # communicate the collection of output values
        self._response_index = 0    # reset the response index
        start_time = time()     # start the timeout timer
        self.write(command)     # run the command

        # wait for the output
        while len(self._output) != return_count and time() - start_time < self.timeout:
            sleep(0.01)
        if len(self._output) != return_count:   # check if timeout occurred
            self._expect_output = False
            raise TimeoutError('The request timed out.')
        return self._output

    def _read_worker_(self):
        """
        This is the reader thread logic that watches the process and collects return values.
        """
        while self._running:
            try:
                read_start = time()     # keep track how long it takes until new data arrives in the reader buffer
                out = self._process.stdout.readline().decode(self.encoding).replace(self.eol, '')   # read the buffer
                read_duration = time() - read_start

                # if output is expected from a query command save the response
                if self._expect_output:
                    if self._response_index > 0:    # skip first response as it is the command itself
                        self._output.append(out)    # append response
                    self._response_index += 1   # increase response index
                    if len(self._output) == self._return_count:     # no more responses needed if demand was met
                        self._expect_output = False

                # if the result is empty and the read duration was really short there is a chance that the process
                # was terminated. If this happens multiple times in a row exit the reader procedure and prepare
                # proper termination.
                if out == '' and read_duration < PROCESS_DEAD_STDIN_READTIME:
                    self._process_dead_count += 1
                else:
                    self._process_dead_count = 0
                    if self.verbose and out != '' and out is not None:
                        print(out)
                if self._process_dead_count >= PROCESS_DEAD_TERMINATION_COUNT:
                    self._running = False
                    break
            except Exception as e:
                if self.verbose:
                    print(e)
        if self.verbose:
            print('Exiting process...')
        self._process.wait()    # wait for process to terminate
        if self.verbose:
            print('process terminated.')


if __name__ == '__main__':
    # Source:Apply:SQUare 91.5kHz, 300mVpp, 1.120Vdc
    handler = ProcessHandler('C:/Data/Projects/optomechanics/devices/PeakTech4055MV/CPP/PeakTech4055MV/Release/PeakTech4055MV.exe')

    while handler.running:
        cmd = input('>')
        #result = handler.query(cmd, 1)
        #print('QUERY RESULT: {}'.format(result))
        #handler.write(cmd)

