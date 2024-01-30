import PySimpleGUI as sg
from optomechanics.experiment.cryotrap_control.monitor import monitor_with_auxIn
import time
import threading
import queue
"""This file uses multithread to achieve the communication between the user interface and the
monitor. The GUI is in the main thread. When user clicked the "start" button on user interface,
the monitor starts running in the child thread. We use Queue to get the current measurement values
from monitor, then read the queue in this GUI file, thus show the current values on the user interface.
The capacity of the queue is set to be one, so only the directionary contains the value measured at the current time
can be saved in the queue.
If more measurement is needed, the user only need to add a new item in the layout after the existed items, and expand the directionary
in the queue reading part.
"""


class GUI:
    def long_function_wrapper(self, queue, folder, work_id, window, T_RPi, T_atto, T_zimon, T_zispectra, T_thyra):
        # This is the thread running monitor_with_auxIn, it runs at the same time as the mainthread GUI
        try:
            show_current_value = True
            exp_mon = monitor_with_auxIn.ExperimentMonitor(folder, show_current_value=True,
                                                T_RPi=T_RPi, T_atto=T_atto, T_zimon=T_zimon, T_zispectra=T_zispectra, T_thyra=T_thyra,
                                                que=queue)
            exp_mon.start_monitoring()
            window.write_event_value('-THREAD DONE-', work_id)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)

    ############################# Begin GUI code #############################
    def the_gui(self):
        sg.theme('Light Brown 3')

        column_start = [
            [
                sg.Text("Data_save Folder"),
                sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
                sg.FolderBrowse()],
            [sg.Button("Thyra"),
             sg.Button("RaspberryPi"),
             sg.Button("AutoDry"),
             sg.Button("Zispectra"),
             sg.Button("Zimon")],
            [sg.Button("Start")],
            [sg.Text(size=(25, 1), key='-OUTPUT-')]
        ]

        column_measurement = [
            [sg.Text("4KstageTemp (K)")],
            [sg.Text("Pressure (mbar)")],
            [sg.Text("SampleTemp (K)")],
            [sg.Text("LO power (V)")],
            [sg.Text("Trap power (V)")],
            [sg.Text("ThyraPressure (mBar)")]

        ]

        column_value = [
            [sg.Text(size=(40, 1), key="4KstageTemp (K)")],
            [sg.Text(size=(40, 1), key="Pressure (mbar)")],
            [sg.Text(size=(40, 1), key="SampleTemp (K)")],
            [sg.Text(size=(40, 1), key="LO power (V)")],
            [sg.Text(size=(40, 1), key="Trap power (V)")],
            [sg.Text(size=(40, 1), key="ThyraPressure (mBar)")]
        ]

        column_end = [
            [sg.Button("Stop")],
            [sg.Button("Click Me")]

        ]

        layout = [
            [
                sg.Column(column_start),
                sg.Column(column_measurement),
                sg.Column(column_value),
                sg.Column(column_end)
            ]
        ]
        window = sg.Window('Multithreaded Window', layout)
        # --------------------- EVENT LOOP ---------------------
        T_RPi = -1
        T_atto = -1
        T_zispectra = -1
        T_thyra = -1
        T_zimon = -1

        work_id = 0
        while True:
            # wait for up to 100 ms for a GUI event
            event, values = window.read(timeout=1000)
            if event in (sg.WIN_CLOSED, 'Exit'):
                break

            if event == "-FOLDER-":
                print('event=FOLDER')
                folder = values["-FOLDER-"]

            if event == "Thyra":
                T_thyra = 1

            if event == "RaspberryPi":
                T_RPi = 1
                print(T_RPi)

            if event == 'AutoDry':
                T_atto = 1

            if event == 'Zispectra':
                T_zispectra = 1

            if event == "Zimon":
                T_zimon = 1

            if event == 'Start':
                print(T_RPi)
                q = queue.Queue(maxsize=1)
                # Set the long-running function(montitor) as a thread.
                thre = threading.Thread(target=self.long_function_wrapper, args=(
                    q, folder, work_id, window, T_RPi, T_atto, T_zimon, T_zispectra, T_thyra))
                thre.setDaemon(True)
                thre.start()
            # Just a button to check if the GUI is still alive
            if event == 'Click Me':
                print('Your GUI is alive and well')
            if event == "Stop":
                print("stop is pressed")
                break

            if event == "-THREAD DONE-":
                completed_work_id = values['-THREAD DONE-']
                window['-OUTPUT2-'].update(
                    'Complete Work ID "{}"'.format(completed_work_id))
                window[completed_work_id].update(text_color='green')
            # Get data from the queue
            try:
                if not q.empty():
                    current_values = q.get()
            except:
                # If there isn't data in the queue yet, the default values of the measurements are 0.
                current_values = {"4KstageTemp (K)": 0,
                                  "Pressure (mbar)": 0,
                                  "SampleTemp (K)": 0,
                                  "LO power (V)": 0,
                                  "Trap power (V)": 0,
                                  "ThyraPressure (mBar)": 0}
            for key, values in current_values.items():
                window[key].update(str(current_values[key]))
        window.close()

############################# Main #############################


if __name__ == '__main__':
    f = GUI()
    f.the_gui()
    print('Exiting Program')
