from optomechanics.post_processing.read.gage_sig import OsciData
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
from optomechanics.post_processing.math.median_calc import median_confidence_interval
from lmfit.models import LorentzianModel, LinearModel
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from copy import copy


def triangular(x, amp, period, offset_x, offset_y):
    x_offseted = x + offset_x
    x_scaled = x_offseted * (2 * np.pi) / (period)
    triangular = abs(signal.sawtooth(x_scaled, 1) * amp) + offset_y
    return triangular


def swipe_delimiters_from_triangular(x, amp, period, offset_x, offset_y):
    points = (np.arange(x[0], x[-1], period / 2) - offset_x)[1:]
    return points


class FinesseCalc:
    def __init__(self, filename):
        self.reflection = OsciData(filename + '01.sig', convert_to_volt=True)
        self.transmission = OsciData(filename + '02.sig', convert_to_volt=True)
        self.piezo_voltage = OsciData(filename + '04.sig', convert_to_volt=False)

        self.find_windows()
        self.find_peaks_spacing()
        self.lorentzian_fit()


    def find_windows(self):
        if (len(self.piezo_voltage.runtime) > 80000):
            piezo_voltage_runtime = self.piezo_voltage.runtime[::1000]
        else:
            piezo_voltage_runtime = self.piezo_voltage.runtime


        delta_time = piezo_voltage_runtime[1] - piezo_voltage_runtime[0]
        freq_spec = abs(np.fft.rfft(self.piezo_voltage.measured_data_raw[::1000]))
        freq_spec[0] = 0
        freq = np.fft.rfftfreq(len(self.piezo_voltage.measured_data_raw[::1000]), delta_time)[np.argmax(freq_spec)]

        guess_amp = (np.max(self.piezo_voltage.measured_data_raw[::1000]) - np.min(self.piezo_voltage.measured_data_raw[::1000]))
        guess_offset_y = abs(np.min(self.piezo_voltage.measured_data_raw[::1000]))
        guess_period = 1 / freq
        guess_offset_x = 0.

        fit = curve_fit(triangular, piezo_voltage_runtime, self.piezo_voltage.measured_data_raw[::1000],
                        p0=[guess_amp, guess_period, guess_offset_x, guess_offset_y])
        points = swipe_delimiters_from_triangular(piezo_voltage_runtime, *fit[0])
        slope_0 = triangular(points[:2], *fit[0])
        if slope_0[1] - slope_0[0] > 0:
            self.rising_idx = 0
            self.falling_idx = 1
        else:
            self.rising_idx = 1
            self.falling_idx = 0


        delimiters = [np.argmin(abs(self.piezo_voltage.runtime - p)) for p in points]
        self.transmission_windowed = np.hsplit(self.transmission.measured_data_volt, delimiters)[1:-2]
        self.reflection_windowed = np.hsplit(self.reflection.measured_data_volt, delimiters)[1:-2]
        self.transmission_runtime_windowed = np.hsplit(self.transmission.runtime, delimiters)[1:-2]
        self.reflection_runtime_windowed = np.hsplit(self.reflection.runtime, delimiters)[1:-2]
        self.piezo_windowed = np.hsplit(self.piezo_voltage.measured_data_raw, delimiters)[1:-2]
        self.piezo_runtime_windowed = np.hsplit(self.piezo_voltage.runtime, delimiters)[1:-2]

        # fig, ax = plt.subplots()
        # ax.plot(self.piezo_voltage.runtime, self.piezo_voltage.measured_data_raw)
        # ax.plot(self.piezo_voltage.runtime, triangular(self.piezo_voltage.runtime, *fit[0]))
        # plt.grid()
        # plt.show()



    def find_peaks_spacing(self, kind='runtime'):
        if kind == 'runtime':
            maxima = np.array([[self.transmission_runtime_windowed[i][np.argmax(self.transmission_windowed[i][:len(self.transmission_windowed[i]) // 2])],
                                self.transmission_runtime_windowed[i][len(self.transmission_windowed[i]) // 2 + np.argmax(self.transmission_windowed[i][len(self.transmission_windowed[i]) // 2:])]]
                               for i in range(len(self.transmission_windowed))])
        elif kind == 'piezo':
            maxima = np.array([[self.piezo_windowed[i][np.argmax(self.transmission_windowed[i][:len(self.transmission_windowed[i]) // 2])],
                                self.piezo_windowed[i][len(self.transmission_windowed[i]) // 2 + np.argmax(self.transmission_windowed[i][len(self.transmission_windowed[i]) // 2:])]]
                               for i in range(len(self.transmission_windowed))])

        width = maxima[:,1] - maxima[:,0]
        self.maxima_rising = maxima[self.rising_idx::2]
        self.maxima_falling = maxima[self.falling_idx::2]
        self.fsr_rising = width[self.rising_idx::2]
        self.fsr_falling = width[self.falling_idx::2]

        self.width_rising_median_interval = median_confidence_interval(self.fsr_rising)
        self.width_falling_median_interval = median_confidence_interval(self.fsr_falling)

        # ax.vlines(maxima[0], *ax.get_ylim(), colors='red', linestyles='--')
        # ax2 = plt.twinx(ax)
        #
        # ax2.plot(piezo_voltage_runtime, self.piezo_voltage.measured_data_raw[::1000], color='red')
        # fit = curve_fit(triangular, self.piezo_voltage.runtime[::1000], self.piezo_voltage.measured_data_raw[::1000], p0=[23128-5692, 0.036-0.003, 0.0165 - 0.0031, 5692])
        # ax2.plot(piezo_voltage_runtime, triangular(piezo_voltage_runtime, *fit[0]), color='blue')
        # points = swipe_delimiters_from_triangular(piezo_voltage_runtime, *fit[0])
        # ax2.plot(points, triangular(points, *fit[0]), 'go')
        #
        #
        # plt.grid()
        # plt.show()
        # #
        # fig, ax = plt.subplots()
        # plt.scatter(0.9*np.ones(len(self.width_rising)), self.width_rising)
        # plt.scatter(1.1*np.ones(len(self.width_falling)), self.width_falling)

    def lorentzian_fit(self, kind='runtime'):
        if kind == 'runtime':
            x = self.transmission_runtime_windowed
        elif kind == 'piezo':
            x = self.piezo_windowed

        Lorentz_mod = LorentzianModel()
        Linear_mod = LinearModel()
        mod = Lorentz_mod + Linear_mod

        linewidth = []
        fit_out = []
        for trans, x_tmp in zip(self.transmission_windowed, x):
            trans_len = len(trans)
            pars_left = Linear_mod.make_params(intercept=trans[:trans_len // 2].min(), slope=0)
            pars_left += Lorentz_mod.guess(trans[:trans_len // 2], x=x_tmp[:trans_len // 2])
            out_left = mod.fit(trans[:trans_len // 2], pars_left, x=x_tmp[:trans_len // 2])

            pars_right = Linear_mod.make_params(intercept=trans[trans_len // 2:].min(), slope=0)
            pars_right += Lorentz_mod.guess(trans[trans_len // 2:], x=x_tmp[trans_len // 2:])
            out_right = mod.fit(trans[trans_len // 2:], pars_right, x=x_tmp[trans_len // 2:])

            linewidth.append([out_left.params['fwhm'], out_right.params['fwhm']])
            fit_out.append([out_left, out_right])

        linewidth = np.array(linewidth)
        fit_out = np.array(fit_out)

        self.linewidth_rising = linewidth[self.rising_idx::2]
        self.linewidth_falling = linewidth[self.falling_idx::2]
        self.fit_out_rising = fit_out[self.rising_idx::2]
        self.fit_out_falling = fit_out[self.falling_idx::2]


    @property
    def finesse_rising(self):
        return self.fsr_rising[:, None] / self.linewidth_rising

    @property
    def finesse_falling(self):
        return self.fsr_falling[:, None] / self.linewidth_falling

    def plot_window(self, id):
        plt.figure()
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        ax2 = host.twinx()
        ax3 = host.twinx()

        offset = 60
        new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
        ax3.axis["right"] = new_fixed_axis(loc="right",
                                           axes=ax3,
                                           offset=(offset, 0))

        ax3.axis["right"].toggle(all=True)

        ref = host.plot(self.transmission_runtime_windowed[id], self.transmission_windowed[id], label='transmission', color='orange')
        trans = ax2.plot(self.reflection_runtime_windowed[id], self.reflection_windowed[id], label='reflection', color='blue')
        piezo = ax3.plot(self.piezo_runtime_windowed[id], self.piezo_windowed[id], label='piezo', color='green')

        host.set_xlabel('runtime [s]')
        host.set_ylabel('Transmission [V]')
        ax2.set_ylabel('Reflection [V]')
        ax3.set_ylabel('Piezo Driving [a.u.]')

        host.axis["left"].label.set_color(ref[0].get_color())
        ax2.axis["right"].label.set_color(trans[0].get_color())
        ax3.axis["right"].label.set_color(piezo[0].get_color())

        plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_fitted_lorentzians(self, kind='rising', indices=None, mark_sidebands=False):
        if kind == 'rising':
            runtimes = self.transmission_runtime_windowed[self.rising_idx::2]
            transmissions = self.transmission_windowed[self.rising_idx::2]
            fit_out = self.fit_out_rising
        elif kind == 'falling':
            runtimes = self.transmission_runtime_windowed[self.falling_idx::2]
            transmissions = self.transmission_windowed[self.falling_idx::2]
            fit_out = self.fit_out_falling

        if indices is None:
            indices = range(len(fit_out))
        elif isinstance(indices, int):
            indices = [indices]
        else:
            print('datatype of indices must be list, numpy.ndarray or integer')
            return

        if mark_sidebands:
            ylims = [-1 * fit_out[indices[0]][0].params['height'].value / 5, fit_out[indices[0]][0].params['height'].value / 3]

        fig, ax = plt.subplots()
        for id in indices:
            if id < len(fit_out):
                ax.plot(runtimes[id], transmissions[id])
                ax.plot(runtimes[id][:len(runtimes[id])//2], fit_out[id][0].best_fit, linestyle='--', color='red')
                ax.plot(runtimes[id][len(runtimes[id]) // 2:], fit_out[id][1].best_fit, linestyle='--', color='red')
                if mark_sidebands:
                    ax.vlines(self._side_band_spacing(id, kind), *ylims, colors='black', linestyles='--')

        plt.grid()
        plt.show()


    def plot_measurement(self):
        plt.figure()
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)
        ax2 = host.twinx()
        ax3 = host.twinx()

        offset = 60
        new_fixed_axis = ax3.get_grid_helper().new_fixed_axis
        ax3.axis["right"] = new_fixed_axis(loc="right",
                                           axes=ax3,
                                           offset=(offset, 0))

        ax3.axis["right"].toggle(all=True)

        ref = host.plot(self.transmission.runtime, self.transmission.measured_data_volt, label='transmission', color='orange')
        trans = ax2.plot(self.reflection.runtime, self.reflection.measured_data_volt, label='reflection', color='blue')
        piezo = ax3.plot(self.piezo_voltage.runtime, self.piezo_voltage.measured_data_raw, label='piezo', color='green')

        host.set_xlabel('runtime [s]')
        host.set_ylabel('Transmission [V]')
        ax2.set_ylabel('Reflection [V]')
        ax3.set_ylabel('Piezo Driving [a.u.]')

        host.axis["left"].label.set_color(ref[0].get_color())
        ax2.axis["right"].label.set_color(trans[0].get_color())
        ax3.axis["right"].label.set_color(piezo[0].get_color())

        plt.grid()
        plt.tight_layout()
        plt.show()

    def _side_band_spacing(self, id, kind='rising'):
        if kind == 'rising':
            x = copy(self.transmission_runtime_windowed[id * 2 + self.rising_idx])
            y = copy(self.transmission_windowed[id * 2 + self.rising_idx])
            out_left = self.fit_out_rising[id][0]
            out_right = self.fit_out_rising[id][1]

        elif kind == 'falling':
            x = copy(self.transmission_runtime_windowed[id * 2 + self.falling_idx])
            y = copy(self.transmission_windowed[id * 2 + self.falling_idx])
            out_left = self.fit_out_falling[id][0]
            out_right = self.fit_out_falling[id][1]

        trans_len = len(y)

        x_left = x[:trans_len // 2]
        x_right = x[trans_len // 2:]
        y_left = y[:trans_len // 2]
        y_right = y[trans_len // 2:]
        width_left = out_left.params['fwhm'].value
        width_right = out_right.params['fwhm'].value
        center_left = out_left.params['center'].value
        center_right = out_right.params['center'].value
        y_left -= out_left.best_fit
        y_right -= out_right.best_fit

        delim_1 = 30
        delim_2 = 8
        y_window_left_left = y_left[(x_left > center_left - width_left * delim_1) * (x_left < center_left - width_left * delim_2)]
        x_window_left_left = x_left[(x_left > center_left - width_left * delim_1) * (x_left < center_left - width_left * delim_2)]
        y_window_left_right = y_left[(x_left > center_left + width_left * delim_2) * (x_left < center_left + width_left * delim_1)]
        x_window_left_right = x_left[(x_left > center_left + width_left * delim_2) * (x_left < center_left + width_left * delim_1)]

        y_window_right_left = y_right[(x_right > center_right - width_right * delim_1) * (x_right < center_right - width_right * delim_2)]
        x_window_right_left = x_right[(x_right > center_right - width_right * delim_1) * (x_right < center_right - width_right * delim_2)]
        y_window_right_right = y_right[(x_right > center_right + width_right * delim_2) * (x_right < center_right + width_right * delim_1)]
        x_window_right_right = x_right[(x_right > center_right + width_right * delim_2) * (x_right < center_right + width_right * delim_1)]

        Lorentz_mod = LorentzianModel()
        Linear_mod = LinearModel()
        mod = Lorentz_mod + Linear_mod

        pars_left_left = Linear_mod.make_params(intercept=y_window_left_left.min(), slope=0)
        pars_left_left += Lorentz_mod.guess(y_window_left_left, x=x_window_left_left)
        out_left_left = mod.fit(y_window_left_left, pars_left_left, x=x_window_left_left)
        pars_left_right = Linear_mod.make_params(intercept=y_window_left_right.min(), slope=0)
        pars_left_right += Lorentz_mod.guess(y_window_left_right, x=x_window_left_right)
        out_left_right = mod.fit(y_window_left_right, pars_left_right, x=x_window_left_right)

        pars_right_left = Linear_mod.make_params(intercept=y_window_right_left.min(), slope=0)
        pars_right_left += Lorentz_mod.guess(y_window_right_left, x=x_window_right_left)
        out_right_left = mod.fit(y_window_right_left, pars_right_left, x=x_window_right_left)
        pars_right_right = Linear_mod.make_params(intercept=y_window_right_right.min(), slope=0)
        pars_right_right += Lorentz_mod.guess(y_window_right_right, x=x_window_right_right)
        out_right_right = mod.fit(y_window_right_right, pars_right_right, x=x_window_right_right)


        center = np.array([out_left_left.params['center'].value, out_left_right.params['center'].value,
                           out_right_left.params['center'].value, out_right_right.params['center'].value])

        # plt.figure()
        # plt.plot(x_window_left_left, y_window_left_left)
        # # plt.plot(x_window_left_left, out_left_left.best_fit, 'r--')
        # plt.plot(x_window_left_right, y_window_left_right)
        # # plt.plot(x_window_left_right, out_left_right.best_fit, 'r--')
        #
        # plt.plot(x_window_right_left, y_window_right_left)
        # # plt.plot(x_window_right_left, out_right_left.best_fit, 'r--')
        #
        # plt.plot(x_window_right_right, y_window_right_right)
        # # plt.plot(x_window_right_right, out_right_right.best_fit, 'r--')
        #
        #
        # plt.grid()
        # plt.show()
        return center

    @property
    def side_band_spacing_rising(self):
        sbs_rising = []
        for id in range(self.fit_out_rising.shape[0]):
            center_left = self.fit_out_rising[id][0].params['center'].value
            center_right = self.fit_out_rising[id][1].params['center'].value
            sidebands = self._side_band_spacing(id, kind='rising')
            sidebands[:2] -= center_left
            sidebands[2:] -= center_right
            sbs_rising.append(list(sidebands))
        sbs_rising = np.array(sbs_rising)
        return sbs_rising

    @property
    def side_band_spacing_falling(self):
        sbs_falling = []
        for id in range(self.fit_out_falling.shape[0]):
            center_left = self.fit_out_falling[id][0].params['center'].value
            center_right = self.fit_out_falling[id][1].params['center'].value
            sidebands = self._side_band_spacing(id, kind='falling')
            sidebands[:2] -= center_left
            sidebands[2:] -= center_right
            sbs_falling.append(list(sidebands))
        sbs_falling = np.array(sbs_falling)
        return sbs_falling
