"""
@author: Andrei Militaru
@date: 12th June 2020
@description: just a collection of functions useful for the active swimmer paper.
"""

import matplotlib.pyplot as plt
from ...visualization.set_axes import set_ax
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch
from numba import njit
from tqdm import tqdm
import math


@njit()
def running_variance(x, n=100):
    N = len(x)
    output = np.zeros_like(x)
    for i in range(n, N):
        if i < n:
            padding = np.zeros(n-i)
            temp = np.concatenate((padding, x[:i]))
            output[i] = np.std(temp)
        else:
            output[i] = np.std(x[i-n:i])
    return output


@njit()
def running_average(x, n=100):
    N = len(x)
    output = np.zeros_like(x)
    for i in range(n, N):
        if i < n:
            padding = np.zeros(n-i)
            temp = np.concatenate((padding, x[:i]))
            output[i] = np.mean(temp)
        else:
            output[i] = np.mean(x[i-n:i])
    return output


@njit()
def indicator(scanned_trace, threshold=200):
    output = np.zeros_like(scanned_trace)
    for i in range(len(scanned_trace)):
        output[i] = 0 if scanned_trace[i] < threshold else 1
    return output


def exp_fit(x, A, R):
    return A*np.exp(-np.abs(x)*R)


@njit()
def jump_hist(hshifted, fs=40e6/64):
    jumps_up = []
    jumps_down = []
    counter = 0
    dt = 1/fs
    state = hshifted[0]
    for i in range(1, len(hshifted)):
        prev = state
        state = hshifted[i]
        counter += 1
        if prev != state and counter > 50:
            if prev == 0:
                jumps_up.append(counter*dt)
            elif prev == 1:
                jumps_down.append(counter*dt)
            counter = 0
        else:
            pass
    jumps_up = np.array(jumps_up)
    jumps_down = np.array(jumps_down)
    return (jumps_up, jumps_down)


def hopping_correlation(hshifted, fs=40e6/64, n_of_interest=None):
    nperseg = n_of_interest if n_of_interest is not None else fs
    freq, Shh = welch(hshifted, fs, nperseg=nperseg)
    Rhh = np.fft.fftshift(np.fft.irfft(Shh))
    Rhh /= np.max(Rhh)*np.var(hshifted)
    Rhh -= Rhh[-1]
    Rt = np.linspace(-len(Rhh)/fs/2, len(Rhh)/fs/2, len(Rhh))
    return Rt, Rhh


@njit()
def right_shift(vector, n):
    output = np.zeros_like(vector)
    for i in range(len(vector)-n):
        output[i+n] = vector[i]
    return output


@njit()
def onesided_correlation_old(hshifted, n_of_interest=None):
    output = np.zeros_like(hshifted)
    N = len(hshifted)
    range_ext = N if n_of_interest is None else n_of_interest
    for i in range(range_ext):
        output[i] = np.sum(hshifted*right_shift(hshifted,i))/(N-i)
    return output


@njit()
def onesided_correlation(hshifted, n_of_interest=None):
    N = len(hshifted)
    range_ext = N if n_of_interest is None else n_of_interest
    output = np.zeros(range_ext)
    for i in range(range_ext):
        output[i] = np.sum(hshifted[i:]*hshifted[:N-i])/(N-i)
    return output


#@njit()
def onesided_correlation_fft(hshifted, fs=40e6/80, tmax=None):
    nperseg = len(hshifted) if tmax is None else int(tmax*fs)
    segments = int(len(hshifted)/nperseg)
    first = True
    """
    for i in range(segments):
        new_transform = np.abs(np.fft.rfft(hshifted[i*nperseg:(i+1)*nperseg]))**2
        if first:
            first = False
            psd = np.zeros(len(new_transform))
        psd += new_transform
    """
    freq, psd = welch(hshifted, fs, nperseg=nperseg)
    Rhh = np.fft.irfft(psd)
    Rhh = Rhh[:math.floor(len(Rhh)/2)]
    Rhh = Rhh/Rhh[0]*np.var(hshifted)
    return Rhh


def get_rate(hshifted, fs=40e6/80, split=10, monitor=True, tmax=None):
    N = len(hshifted)
    nperseg = int(N/split)
    Tseg = nperseg/fs
    T = N/fs
    gammas = np.zeros(split)
    for i in tqdm(range(split)):
        #corr = onesided_correlation_fft(hshifted[i*nperseg: (i+1)*nperseg], tmax=tmax)
        corr = onesided_correlation(hshifted[i*nperseg: (i+1)*nperseg], n_of_interest=int(tmax*fs))
        log_corr = np.log(np.abs(corr))
        local_time = np.linspace(0, tmax, len(corr))
        gamma = np.dot((log_corr-log_corr[0]), local_time)/(
            np.dot(local_time, local_time))
        gammas[i] = gamma
        if monitor:
            plt.semilogy(local_time, np.abs(corr), 'o')
            plt.semilogy(local_time, np.abs(corr[0])*np.exp(gamma*local_time), '--')
            plt.show()
    return np.mean(gammas), np.std(gammas)


def get_rate_old(Rt, Rhh, plot_monitoring=True):
    param, cov = curve_fit(exp_fit, Rt, Rhh)
    if plot_monitoring:
        xlabel = 'Time [ms]'
        ylabel = r'$R_{hh}$'
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        ax.plot(Rt/1e-3, Rhh, '.', linewidth=2, label='Measured')
        ax.plot(Rt/1e-3, exp_fit(Rt, *param), linewidth=2, label='Fit')
        set_ax(ax, xlabel=xlabel, ylabel=ylabel, legend=True)
        plt.tight_layout()
        plt.show()
    return (param[1], cov[1,1])


def low_pass(f, A, fc):
    return A/(f**2 + fc**2)


def get_rate_psd(h, fs=40e6/80, resolution=10, plot_fit=True):
    nperseg = int(fs/resolution)
    freq, psd = welch(h, fs, nperseg=nperseg)
    param, cov = curve_fit(low_pass, freq, psd)
    if plot_fit:
        fig = plt.figure(figsize=(6,3))
        ax = fig.add_subplot(111)
        ax.loglog(freq/1e3, psd, 'o', label='measurement')
        ax.loglog(freq/1e3, low_pass(freq, *param), '--', linewidth=2, label='fit')
        ax.loglog([param[1]/1e3]*2, [np.min(psd), np.max(psd)], '--', linewidth=2, label='$f_c$')
        set_ax(ax, xlabel='Frequency [kHz]', ylabel=r'$S_{hh}$ [Hz$^{-1}$]', legend=True)
        plt.show()
    return np.abs(param[1])*2*np.pi


def get_rate_psd_split(hshifted, fs=40e6/80, split=10, **kwargs):
    N = len(hshifted)
    nperseg = int(N/split)
    Tseg = nperseg/fs
    T = N/fs
    gammas = np.zeros(split)
    for i in tqdm(range(split)):
        gamma = get_rate_psd(hshifted[i*nperseg: (i+1)*nperseg], fs=fs, **kwargs)
        gammas[i] = gamma
    return np.mean(gammas), np.std(gammas)