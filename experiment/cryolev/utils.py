# -*- coding: utf-8 -*-
from __future__ import print_function
import pickle
import os
import numpy as np
import scipy as sp


def ellipse_covariance(cm, n=100, thr=1e-10):
    _, eigvec = np.linalg.eig(cm)
    D = (eigvec.T @ cm @ eigvec)
    D[D < thr] = 0
    t = np.linspace(0, 2*np.pi, n)
    x = np.zeros((2, len(t)))
    for j in range(len(t)):
        x[:, j] = np.array([[np.cos(t[j]), np.sin(t[j])]]
                           ) @ np.sqrt(D) @ eigvec.T
    return x


def lor(f, fc, g, a, n):
    """
    Lorentzian function
    :param f: frequency
    :param fc: center frequency
    :param g: FWHM
    :param a: area
    :param n: background noise
    """
    return a / np.pi * g / 2. / ((f - fc)**2 + (g / 2.)**2) + n


def find_ind(tar, tval):
    """
    Find index in axfrray
    :param tar: array
    :param tval: value for which the index has to be found
    """
    return np.where(abs(tar - tval) == np.min(abs(tar - tval)))[0][0]


def bin_array(arr, wnd):
    """
    Reshape 1D array.
    """
    N = int(int(arr.shape[0] / wnd) * wnd)
    if np.mod(arr.shape[0], wnd) != 0:
        pass
#         print("WARNING: input array size not multiple of window size")
    return arr[:N].reshape(int(N / wnd), wnd).mean(axis=1)


def unwrap(y):
    counter_p = 0
    counter_m = 0
    y_tmp = np.zeros(len(y))
    y_tmp[0] = y[0]
    for i in range(len(y) - 1):
        if (y[i] - y[i - 1]) > 0.9 * np.pi:
            counter_p += 1
        elif (y[i] - y[i - 1]) < -0.9 * np.pi:
            counter_m += 1
        y_tmp[i] = y[i] - (counter_p - counter_m) * np.pi
    return y_tmp


def exp_dec(t, tc, a, n):
    return a * np.exp(-t / tc) + n


def exp_heat(t, tc, a, n):
    return a * (1.0 - np.exp(-t / tc)) + n


def print_dict(v, prefix=''):
    if isinstance(v, dict):
        for k, v2 in v.items():
            p2 = "{}['{}']".format(prefix, k)
            print_dict(v2, p2)
    elif isinstance(v, list):
        for i, v2 in enumerate(v):
            p2 = "{}[{}]".format(prefix, i)
            print_dict(v2, p2)
    else:
        print('{}'.format(prefix))


def demod_filter(freq, TC, order):
    return 1/(np.abs(1+1j*2*np.pi*freq*TC)**2)**order


def demod_filter_tf(freq, TC, order):
    return 1/((1+1j*2*np.pi*freq*TC))**order


def compute_psd(y, dt, n_seg=1, estimator='mean', fast=True):
    N = len(y)
    pt_seg = N//n_seg
    N_fft = int(pt_seg*n_seg)
    if fast:
        pt_seg = 2**int(np.log2(N/n_seg))
        n_seg = N//pt_seg
        N_fft = int(pt_seg*n_seg)
    wnd = sp.signal.windows.hann(pt_seg)
    wnd /= np.sqrt(np.mean(wnd**2))
    wnd = np.repeat(wnd[np.newaxis, :], n_seg, axis=0)
    if estimator == 'mean':
        psd = np.mean(abs(sp.fftpack.fftshift(sp.fftpack.fft(
            wnd * y[(N-N_fft)//2:(N+N_fft)//2].reshape((n_seg, pt_seg)), axis=1)/pt_seg))**2, axis=0)
    elif estimator == 'median':
        psd = np.median(abs(sp.fftpack.fftshift(sp.fftpack.fft(
            wnd * y[(N-N_fft)//2:(N+N_fft)//2].reshape((n_seg, pt_seg)), axis=1)/pt_seg))**2, axis=0)
    f = sp.fftpack.fftshift(sp.fftpack.fftfreq(pt_seg, d=dt))
    return f, psd


def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     length=100,
                     fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


red = (227/255.0, 26/255.0, 28/255.0)
slightred = (234/255.0, 64/255.0, 64/255.0)
sslightred = (247/255.0, 81/255.0, 81/255.0)
lightred = (250/255.0, 152/255.0, 151/255.0)
blue = (32/255.0, 120/255.0, 180/255.0)
slightblue = (38/255.0, 146/255.0, 217/255.0)
sslightblue = (88/255.0, 163/255.0, 203/255.0)
lightblue = (166/255.0, 206/255.0, 227/255.0)
green = (53/255.0, 161/255.0, 46/255.0)
lightgreen = (175/255.0, 221/255.0, 138/255.0)
orange = (235/255.0, 109/255.0, 27/255.0)
lightorange = (250/255.0, 163/255.0, 46/255.0)
darkorange = (222/255.0, 129/255.0, 1/255.0)
midpurple = (115/255.0, 75/255.0, 158/255.0)
lightpurple = (155/255.0, 89/255.0, 182/255.0)
purple = (88/255.0, 57/255.0, 121/255.0)
darkred = (173/255.0, 20/255.0, 20/255.0)
darkblue = (23/255.0, 91/255.0, 136/255.0)
darkgreen = (50/255.0, 117/255.0, 40/255.0)
darkpurple = (116/255.0, 61/255.0, 122/255.0)
black = 'k'
white = 'w'
gray = 'gray'
lightgray = 'lightgray'

allcolors = [red, slightred, sslightred, lightred, blue, slightblue, sslightblue, lightblue, green, lightgreen,
             orange, lightorange, darkorange, midpurple, lightpurple, purple, darkred, darkblue, darkgreen, darkpurple]
allcolornames = ["red", "slightred", "sslightred", "lightred", "blue", "slightblue", "sslightblue", "lightblue", "green", "lightgreen",
                 "orange", "lightorange", "darkorange", "midpurple", "lightpurple", "purple", "darkred", "darkblue", "darkgreen", "darkpurple"]

color_cycle = [slightblue, red, green, orange, lightpurple,
               lightblue, lightred, lightgreen, lightorange]

# ratiocolor = (244/255.0,177/255.0, 131/255.0)
# ratiocolordark = (197/255.0, 90/255.0, 17/255.0)


def colors1():
    return [blue, green, red, midpurple, orange, lightblue, lightgreen, lightred, lightpurple, lightorange]


def darkcolors1():
    return [darkblue, darkgreen, darkred, darkpurple, darkorange, blue, green, red, purple, orange]


def lightcolors1():
    return [lightblue, lightgreen, lightred, lightpurple, lightorange]


def sns_reds():
    return [(0.99506343743380377, 0.859653987604029, 0.7986620650571935),
            (0.98823529481887817, 0.6866743746925803, 0.57788544519274843),
            (0.98658977396347947, 0.5067281983646692, 0.38123799246900225),
            (0.95700115596546842, 0.30871203658627527, 0.22191465613888758),
            (0.83704729921677534, 0.13394848298208384, 0.13079584956753487),
            (0.6663437338436351, 0.063391003889196051, 0.086412920408389146)]


def sns_blues():
    return [(0.85840831293779263, 0.91344867874594293, 0.96456747616038607),
            (0.73094965079251462, 0.83947713375091548, 0.92132257293252384),
            (0.53568628967977039, 0.7460822911823497, 0.86425221877939562),
            (0.32628989885835086, 0.61862362903707169, 0.80279893524506507),
            (0.16696656059985066, 0.48069205132185244, 0.72915034294128422),
            (0.044059979477349451, 0.33388697645243476, 0.62445215617909156)]


def sns_grays():
    return [(0.92950404111076801, 0.92950404111076801, 0.92950404111076801),
            (0.81911573550280403, 0.81911573550280403, 0.81911573550280403),
            (0.677001172654769, 0.677001172654769, 0.677001172654769),
            (0.50857363752290308, 0.50857363752290308, 0.50857363752290308),
            (0.35912342246840978, 0.35912342246840978, 0.35912342246840978),
            (0.16793541627771713, 0.16793541627771713, 0.16793541627771713)]


def sns_greens():
    return [(0.88281430917627668, 0.9546943510279936, 0.86219147303525134),
            (0.73714726672453035, 0.89551711503197162, 0.7108343117377337),
            (0.55732411999328468, 0.81642446307575001, 0.54695887647423092),
            (0.33882353901863099, 0.71172627631355734, 0.40584391180206747),
            (0.17139562607980241, 0.58151482694289269, 0.29790081942782681),
            (0.017762399946942051, 0.44267590116052069, 0.18523645330877864)]


def sns_greenblue():
    return [(0.8682814380701851, 0.94888120258555697, 0.84765860192915976),
            (0.75903115623137529, 0.90563629935769474, 0.75434065285850971),
            (0.58477509874923561, 0.83869282007217405, 0.73448675169664268),
            (0.3799308030044331, 0.74309882346321554, 0.80276817784589882),
            (0.20845829207523198, 0.59340256172067973, 0.76899655356126673),
            (0.049134950339794162, 0.42611304170945108, 0.68364477087469666)]


def sns_greenblue_d():
    return [(0.21697808798621682, 0.32733564601225013, 0.36941176807179171),
            (0.23442778952760632, 0.45820839330261826, 0.54352941859002213),
            (0.25140587751382315, 0.58554403931486831, 0.71294118666181383),
            (0.32480841754308709, 0.68493145540648814, 0.78994746862673293),
            (0.45066770474895151, 0.75099834881576832, 0.77038576275694604),
            (0.58002308326608998, 0.81890043370863974, 0.75028067616855398)]
