import numpy as np
import scipy as sp
from collections import Iterable
import sys


def median_confidence_index(N, confidence):
    if isinstance(N, Iterable) and isinstance(confidence, Iterable):
        if not isinstance(N, np.ndarray):
            N = np.array(N)
        if not isinstance(confidence, np.ndarray):
            confidence = np.array(confidence)
        if N.shape != confidence.shape:
            sys.exit('If N and confidence are Iterables, their shape has to match. {0} and {1}'.format(N.shape, confidence.shape))
    elif isinstance(N, Iterable):
        if not isinstance(N, np.ndarray):
            N = np.array(N)
        confidence = np.array([confidence] * len(N))
    elif isinstance(confidence, Iterable):
        if not isinstance(confidence, np.ndarray):
            confidence = np.array(confidence)
        N = np.array([N] * len(confidence))
    else:
        N = np.array([N])
        confidence = np.array([confidence])
    sum1 = np.ones(N.shape[0])
    k = np.zeros(N.shape[0], dtype=int)
    sum0 = np.ones(N.shape[0])
    while(np.any(sum1 > confidence)):
        msk = (sum1 > confidence)
        sum0[msk] = sum1[msk]
        binom = np.array([sp.special.binom(N_, k_) for N_, k_ in zip(N, k)])
        sum1[msk] = sum1[msk] - 2 * binom[msk] / 2 ** N[msk]
        k[msk] += 1
    return k - 2, sum0

def quantile_confidence_index(N, confidence, quantile):
    N_needed = required_sample_size(confidence, quantile)
    if N < required_sample_size(confidence, quantile):
        raise ValueError('Not enough datapoints to calculate the desired confidence interval (minimal datapoints = '
                         + str(int(np.ceil(N_needed))) + ').')
    sum1 = 1
    k = 0
    while(sum1 > confidence):
        sum0 = sum1
        sum1 = sum1 - sp.special.binom(N, k) * quantile**k * (1 - quantile)**(N - k)
        k += 1
    return k - 2, sum0


def required_sample_size(confidence, quantile):
    return np.log(1 - confidence) / np.log(1 - quantile)


def median_confidence_interval(data, confidence=0.75, axis=-1):
    ordered_data = np.sort(data, axis=axis)
    N = np.sum(np.isfinite(ordered_data), axis=axis)
    calculate_index = median_confidence_index(N=N, confidence=confidence)
    sl1 = [np.arange(len(N))] * data.ndim
    sl2 = [np.arange(len(N))] * data.ndim
    sl1[axis] = calculate_index[0]
    sl2[axis] = N - (calculate_index[0] + 1)
    confidence_interval = [ordered_data[sl1], ordered_data[sl2]]
    return confidence_interval, calculate_index[1]

def quantile_confidence_interval(data, confidence=0.75, quantile=None, sigma=None):
    if quantile is None and sigma is None:
        raise TypeError('Either desired quantile or sigma range must be specified')
    elif quantile is not None and sigma is not None:
        raise TypeError('Only desired quantile or sigma range must be specified, not both')
    elif sigma is not None:
        sigma_dict = {1: 0.6827, 2: 0.9545, 3: 0.9973}
        if sigma not in sigma_dict:
            raise ValueError('Sigma must be 1, 2 or 3.')
        quantile = (1 - sigma_dict[sigma]) / 2
    ordered_data = np.sort(data)
    N = len(data)
    calculate_index = quantile_confidence_index(N, confidence=confidence, quantile=quantile)
    confidence_interval = [ordered_data[calculate_index[0]], ordered_data[N - (calculate_index[0] + 1)]]
    return confidence_interval, calculate_index[1]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sn

    sample_size = 24
    sample_data_list = [np.random.rand(sample_size) * 50, np.random.randn(sample_size) * 50, np.random.triangular(0, 25, 50, sample_size)]

    for sample_data in sample_data_list:
        median_conf_int, containing_probab = median_confidence_interval(sample_data)
        sigma_conf_int, containing_probab = quantile_confidence_interval(sample_data, sigma=1)

        mean, variance, std = sp.stats.bayes_mvs(sample_data, 0.75)

        plt.figure()
        plt.plot(sample_data, linestyle = '', marker = 'o')
        plt.hlines(median_conf_int, xmin=0, xmax=len(sample_data), linestyle ='--', color='red')
        plt.hlines(sigma_conf_int, xmin=0, xmax=len(sample_data), linestyle ='--', color='black')

        plt.hlines(np.hstack((mean.statistic + np.array(std.minmax), mean.statistic - np.array(std.minmax))),
                   xmin=0, xmax=len(sample_data), linestyle = '--', color='green')
        plt.hlines(mean.minmax, xmin=0, xmax=len(sample_data), linestyle = '--', color='orange')

    plt.show()
