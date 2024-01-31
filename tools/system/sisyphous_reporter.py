# Import for sending emails
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Standard Imports
from time import sleep, time
import glob
import os
import numpy as np
import psutil
import subprocess
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta as td

# Imports for the network printers
from optomechanics.tools.system.network_printer import HPLaserJetP3015, HPLaserJetMFP


class Job:
    '''This is the baseclass for jobs executed by the reporter. It ensures that the reporting protocl is the same for
    all jobs. If you create a job inherit from this baseclass and modify the to_do function as desired.'''
    receivers = []
    subject = None
    content = None
    send_mail = False
    last_mail = None

    def __init__(self, receivers=[], subject=None, content=None):
        '''When creating a Job the receivers for emails, as well as a subject and content of the emails can be passed'''
        self.receivers = receivers
        self.subject = subject
        self.content = content

    def to_do(self):
        '''This function will be called periodically by the reporter. Since this is the base class nothing is specified
        here.'''
        pass


class CheckVibrationLogger(Job):
    '''This Job checks if the vibration logs at the given diretories are still being updated.'''
    directories = {'M45' : r'/media/vibration/m',
                   'D31' : r'/media/vibration/d'}
    latest_updates = {}

    def to_do(self):
        '''This function checks the latest updates of the files in self.directories and initiates and email report if
        any update is longer ago then a specific time (24h). It will also at most send out an report every 24 hours
        to avoid spamming of the responsibles.'''
        self.latest_updates = {}
        current_timestamp = dt.fromtimestamp(time())

        for dir in self.directories:
            list_of_files = glob.glob(self.directories[dir] + '/*.h5')
            latest_file = max(list_of_files, key=os.path.getctime)
            latest_timestamp = dt.fromtimestamp(os.path.getctime(latest_file))
            self.latest_updates[dir] = [latest_file,
                                        latest_timestamp,
                                        (current_timestamp - latest_timestamp) > td(days=1)]
        unavailability_all = np.array([self.latest_updates[dir][-1] for dir in self.latest_updates])
        unavailability = any(unavailability_all)
        if unavailability:
            devices = np.array([dir for dir in self.directories])[unavailability_all]
            self.subject = 'VibrationLoggingError: No Logging to Cerberous for a day!'
            self.content = 'There was no logging to cerberous within 24 hours. The following devices failed: \n'
            self.content += 'Device\t\tLatest File\t\t\t\tLatest Edit\n'
            for dev in devices:
                ts = self.latest_updates[dev][1].strftime('%Y-%m-%d %H:%M:%S')
                self.content += dev + '\t\t' + self.latest_updates[dev][0] + '\t\t' + ts + '\n'

            if self.last_mail is None:
                self.last_mail = current_timestamp
                self.send_mail = True
            else:
                time_since_last_mail = current_timestamp - self.last_mail
                if time_since_last_mail > td(days=1):
                    self.last_mail = current_timestamp
                    self.send_mail = True


class ReniceService(Job):
    '''The job changes the niceness of tasks periodically to avoid jamming of the CPU.'''
    get_processes_command = b'ps -T -e -o spid,comm,user'
    renice_process_command = [b'sudo', b'renice', b'-n', b'', b'-p', b'']
    header = [b'SPID', b'COMMAND', b'USER']
    niceness_dict = {b'comsol' : 19}

    def to_do(self):
        '''This function will update the niceness of any task that contains the words mentioned in self.niceness_dict.
        If more processes should be limited add more entries to the dictionary.'''
        process = subprocess.Popen(self.get_processes_command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = np.array([o for o in output.split(b'\n')][:-1])
        split = output[0].split(self.header[0])
        for h in self.header[1:]:
            split = [s.split(h) for s in split]
            split = [item for sublist in split for item in sublist]
        header_lens = [len(split[0]) + len(self.header[0]), len(split[1])+ len(split[2]) + len(self.header[1])]
        indices = [0] + header_lens
        indices = [sum(indices[:n]) for n in range(1, len(indices) + 1)]
        processes = np.array([[s[i:j] for i,j in zip(indices, indices[1:]+[None])] for s in output])
        for processtype in self.niceness_dict:
            relevant_processes = processes[np.array([processtype in p for p in processes[:, 1]])]
            for p in relevant_processes:
                self.renice_process_command[-1] = p[0]
                self.renice_process_command[-3] = bytes(str(self.niceness_dict[processtype]), 'utf-8')
                process = subprocess.Popen(self.renice_process_command, stdout=subprocess.PIPE)
                process.communicate()


def memory_info_rss(*args, **kwargs):
    '''Helper Function to obtain memory information'''
    return psutil.Process.memory_info(*args, **kwargs).rss


def memory_info_vms(*args, **kwargs):
    '''Helper Function to obtain memory information'''
    return psutil.Process.memory_info(*args, **kwargs).vms


def cpu_times_user(*args, **kwargs):
    '''Helper Function to obtain cpu information'''
    return psutil.Process.cpu_times(*args, **kwargs).user


def cpu_times_system(*args, **kwargs):
    '''Helper Function to obtain cpu information'''
    return psutil.Process.cpu_times(*args, **kwargs).system


class SystemStats(Job):
    '''The job logs system information periodically to cerberous.'''
    logfile = r'/media/sisy_on_cerb/resource_monitoring/'
    information_header = {
        'cpu_num' : psutil.Process.cpu_num,
        'cpu_percent': psutil.Process.cpu_percent,
        'cpu_times_user': cpu_times_user,
        'cpu_times_system': cpu_times_system,
        'memory_info_rss': memory_info_rss,
        'memory_info_vms': memory_info_vms,
        'create_time': psutil.Process.create_time,
        'name': psutil.Process.name,
        'username': psutil.Process.username
    }
    last_log = None

    def to_do(self):
        '''The function checks if it is time to log system information again and acquires them.'''
        timestamp = dt.now()
        if self.last_log is None:
            time_since_last_log = td(seconds=0)
        else:
            time_since_last_log = timestamp - self.last_log
        if (self.last_log is None) or (time_since_last_log > td(hours=1)):
            self.last_log = timestamp
            processes = psutil.process_iter()
            information = []
            for p in list(processes):
                try:
                    info_tmp = []
                    for info in self.information_header:
                        info_tmp.append(self.information_header[info](p))
                    information.append(info_tmp)
                except:
                    pass
            information = pd.DataFrame(data=information, columns=self.information_header)

            filename = self.logfile + dt.now().strftime('%Y-%m') + '.h5'
            key = dt.now().strftime('%Y-%m-%dT%H-%M-%S')
            information.to_hdf(filename, key, mode='a')


class PrinterCheck(Job):
    '''This Job connects to the webinterface of the network printer. Scrapes them for cartridge information and saves
    them to cerberous.'''
    logfile = r'/media/sisy_on_cerb/printer_monitoring/'
    last_log = None
    last_mail = None
    def __init__(self, receivers=[], subject=None, content=None):
        super().__init__(receivers, subject, content)
        self.color_printer = HPLaserJetMFP()
        self.bw_printer = HPLaserJetP3015()

    def to_do(self):
        timestamp = dt.now()
        if self.last_log is None:
            time_since_last_log = td(seconds=0)
        else:
            time_since_last_log = timestamp - self.last_log
        if (self.last_log is None) or (time_since_last_log > td(hours=1)):
            self.last_log = timestamp
            information_color = pd.DataFrame(self.color_printer.properties)
            information_bw = pd.DataFrame(self.bw_printer.properties)
            filename_color = self.logfile + 'color_' + dt.now().strftime('%Y-%m') + '.h5'
            filename_bw = self.logfile + 'bw_' + dt.now().strftime('%Y-%m') + '.h5'
            key = dt.now().strftime('%Y-%m-%dT%H-%M-%S')
            information_color.to_hdf(filename_color, key, mode='a')
            information_bw.to_hdf(filename_bw, key, mode='a')
            self.content = ''
            cartridge_low = [False, False]
            for n, (information, printer) in enumerate(zip([information_color, information_bw], [self.color_printer, self.bw_printer])):
                colors = np.array(information.columns.values)
                levels = np.array([int(cap.replace('%', '')) for cap in information.loc['capacity'].values])
                low_lvl_mask = (levels < 10)
                if any(low_lvl_mask):
                    colors = colors[low_lvl_mask]
                    levels = levels[low_lvl_mask]
                    self.subject = 'PrinterCheck: Cartridge level of Printer is critical!'
                    self.content += 'For printer ' + printer.type + ' the cartridge level of the following colors are critical: \n'
                    self.content += 'Color\t\tLevel\n'
                    for col, lev in zip(colors, levels):
                        self.content += col + '\t\t' + str(lev) + '%\n'
                    self.content += '\n'
                    cartridge_low[n] = True
            if any(cartridge_low):
                if self.last_mail is None:
                    self.last_mail = timestamp
                    self.send_mail = True
                else:
                    time_since_last_mail = timestamp - self.last_mail
                    if time_since_last_mail > td(days=1):
                        self.last_mail = timestamp
                        self.send_mail = True


class Reporter:
    '''This is the main class that runs all the jobs. The connections details to the gmail account are stored here. The
    wait_time indicates how often the jobs are executed. If for specific jobs a lower interval is desired this should be
    inidicated in the job itself. The wait_time here gives the minimal time resolution.'''
    wait_time = 5 * 60
    port = 465  # For SSL
    context = None
    account = "sisyphous.photonics@gmail.com"
    password = "Sisyphous.123"
    COMMASPACE = ', '
    jobs = []
    admin = 'dwindey@ethz.ch'
    last_mail = {}
    shared_folders = ['/media/vibration', '/media/sisy_on_cerb']
    def __init__(self, jobs):
        '''Upon initialization a connection to the mail account is set up. Then the cerberous network shares are mounted
        and the job execution is triggered. If the network mount fails an email is sent to the admin.'''
        self.context = ssl.create_default_context()
        self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.port, context=self.context)
        self.server.login(self.account, self.password)
        self.jobs = jobs
        try:
            self.mount_shared()
        except:
            error_message = 'Mounting of network shares failed:\n'
            self.send_mail([self.admin], 'Mounting of network shares failed', error_message)
        self.run_jobs()

    def send_mail(self, receivers, subject, content):
        '''Function that sends a mail to specified receivers wit given subject and content.'''
        try:
            msg = MIMEMultipart()
            msg['Subject'] = subject
            msg['From'] = self.account
            msg['To'] = self.COMMASPACE.join(receivers)
            msg.attach(MIMEText(content, 'plain'))
            self.server.sendmail(self.account, receivers, msg.as_string())
        except:
            print('Lost Mail Server connection. Trying to reconnect after waiting 5 minutes.')
            sleep(5 * 60)
            self.context = ssl.create_default_context()
            self.server = smtplib.SMTP_SSL("smtp.gmail.com", self.port, context=self.context)
            self.server.login(self.account, self.password)
            self.send_mail(receivers, subject, content)


    def mount_shared(self):
        '''Mounting a remote folder if it is not mounted yet.'''
        mounts = subprocess.check_output('mount').split(b'\n')
        for drive in self.shared_folders:
            drive_mounted = any([drive in m.decode('utf-8') for m in mounts])
            if not drive_mounted:
                os.system('mount ' + drive)

    def run_jobs(self):
        '''Loop that continously runs the jobs specified. If an execution fails and email is send to the admin.'''
        while(True):
            for job in self.jobs:
                try:
                    job.to_do()
                    if job.send_mail:
                        self.send_mail(job.receivers, job.subject, job.content)
                        job.send_mail = False
                except:
                    error_message = 'A Job Execution on Sisyphous failed! The name of the job is:\n'
                    error_message += job.__class__.__name__
                    if job.__class__.__name__ not in self.last_mail:
                        self.send_mail([self.admin], 'Job Execution Failed', error_message)
                        self.last_mail[job.__class__.__name__] = dt.now()
                    else:
                        time_since_last_mail = dt.now() - self.last_mail[job.__class__.__name__]
                        if time_since_last_mail > td(days=1):
                            self.send_mail([self.admin], 'Job Execution Failed', error_message)
                            self.last_mail[job.__class__.__name__] = dt.now()

            sleep(self.wait_time)


if __name__ == '__main__':
    rep = Reporter([CheckVibrationLogger(['dwindey@ethz.ch', 'ebonvin@ethz.ch']),
                    ReniceService(),
                    SystemStats(),
                    PrinterCheck(['dwindey@ethz.ch'])])
    # rep = Reporter([])
