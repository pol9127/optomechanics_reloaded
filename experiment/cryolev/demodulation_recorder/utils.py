import numpy as np
import time


def config_scope_settings(zidrec, dev, samp_rate, T, channels, enables, pwr_two=True, bwlimit=(1, 1)):
    clockbase = zidrec.lock_in.getInt('/{:s}/clockbase'.format(dev))
    samp_rate = clockbase / 2 ** round(np.log2(clockbase / samp_rate))
    if pwr_two:
        T_pts = 2 ** round(np.log2(samp_rate * T))
    else:
        T_pts = round(samp_rate * T)

    # Settings scope
    zidrec.lock_in.setInt('/{:s}/scopes/0/time'.format(dev), int(np.log2(clockbase / samp_rate)))  # 60/2**4 = 3.75 MHz
    zidrec.lock_in.setInt('/{:s}/scopes/0/length'.format(dev), T_pts)

    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/0/inputselect'.format(dev), channels[0])
    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/1/inputselect'.format(dev), channels[1])

    zidrec.lock_in.setInt('/{:s}/scopes/0/channel'.format(dev),
                          np.sum([(idx + 1) * is_enable for idx, is_enable in enumerate(enables)]))

    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/0/bwlimit'.format(dev), bwlimit[0])
    zidrec.lock_in.setInt('/{:s}/scopes/0/channels/1/bwlimit'.format(dev), bwlimit[1])


def config_scope_trigger(zidrec, dev, channel, slope, level, hysteresis=0, holdoff=0, reference=0.5, delay=0):
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigchannel'.format(dev), channel)  # 3=trigger in 2
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigslope'.format(dev), slope)  # 1=rise
    zidrec.lock_in.setDouble('/{:s}/scopes/0/triglevel'.format(dev), level)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trighysteresis/absolute'.format(dev), hysteresis)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigholdoff'.format(dev), holdoff)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigreference'.format(dev), reference)
    zidrec.lock_in.setDouble('/{:s}/scopes/0/trigdelay'.format(dev), delay)


def enable_scope_trigger(zidrec, dev, enable):
    zidrec.lock_in.setInt('/{:s}/scopes/0/trigenable'.format(dev), enable)


def config_scope_module(zidrec, mode, averages=1, history=0):
    zidrec.scope = zidrec.lock_in.scopeModule()
    zidrec.scope.set('mode', mode)  # Time mode
    zidrec.scope.set("averager/weight", averages)  # no averages
    zidrec.scope.set('averager/restart', 1)
    zidrec.scope.set('historylength', history)
    zidrec.scope.set('fft/power', 1)  # Time mode
    zidrec.scope.set('fft/spectraldensity', 1)  # Time mode
    zidrec.scope.set('fft/window', 1)  # 1=Hann


def get_data_scope(zidrec, dev, num_records=1, timeout=300, verbose=False):
    zidrec.scope.set('averager/restart', 1)
    zidrec.scope.subscribe('/{:s}/scopes/0/wave'.format(dev))
    # get_scope_records
    zidrec.scope.execute()
    zidrec.lock_in.setInt('/{:s}/scopes/0/enable'.format(dev), 1)
    zidrec.lock_in.sync()
    start = time.time()
    records = 0
    progress = 0
    while (records < num_records) or (progress < 1.0):
        time.sleep(0.5)
        records = zidrec.scope.getInt('records')
        progress = zidrec.scope.progress()[0]
        if verbose:
            print(
                f"Scope module has acquired {records} records (requested {num_records}). "
                f"Progress of current segment {100.0 * progress}%.",
                end="\r",
            )

    zidrec.lock_in.setInt('/{:s}/scopes/0/enable'.format(dev), 0)
    data = zidrec.scope.read(True)
    zidrec.scope.finish()
    return data


def set_calibration_tone(zidrec, freq, ampl):
    # Set calibration tone from HF2. Connected to output 2
    zidrec.lock_in.setDouble('/dev255/oscs/0/freq', freq)
    zidrec.lock_in.setDouble('/dev255/sigouts/1/amplitudes/0', ampl)
    zidrec.lock_in.setInt('/dev255/sigouts/1/enables/0', 1)
    zidrec.lock_in.setInt('/dev255/sigouts/1/on', 0)


def toggle_calibration_tone(zidrec, status):
    zidrec.lock_in.setInt('/dev255/sigouts/1/on', status)
