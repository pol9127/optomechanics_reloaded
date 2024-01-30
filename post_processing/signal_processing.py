from __future__ import division, print_function, unicode_literals

from numpy import array, sum, sqrt


def weighted_mean(values, standard_deviations, correct=True):
    """Derive weighted arithmetic mean for values with standard
    deviations.

    For the mean the weights $w_i = \frac{1}{\sigma_i^2}$ are used,
    such that

    $\bar{x} = \frac{ \sum_{i=1}^n \left( x_i \sigma_i^{-2} \right)}
                    {\sum_{i=1}^n \sigma_i^{-2}}$

    To correct for under-dispersion due to underestimations of
    errors, the variance is calculated as

    $\sigma_{\bar{x}}^2 = \frac{ 1 }{\sum_{i=1}^n \sigma_i^{-2}}
     \times \frac{1}{(n-1)} \sum_{i=1}^n \frac{ (x_i - \bar{x} )^2}
     {\sigma_i^2}$

    Note
    ----
    More information can be found at
    http://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    #Dealing_with_variance

    Parameters
    ----------
    values : ndarray or array_like
        Values to be averaged.
    standard_deviations : ndarray or array_like
        Standard deviations for the values.
    correct : boolean
        If 'True', the standard deviation of the mean value will be
        corrected for over- or under-dispersion.

    Returns
    -------
    float
        Weighted arithmetic mean of input values.
    float
        Standard deviation of the mean value.
    """

    values = array(values)
    standard_deviations = array(standard_deviations)

    weights = 1/(standard_deviations**2)

    mean = sum(values*weights) / sum(weights)

    if correct and len(values) > 1:
        variance = 1 / sum(weights) * \
                   sum((values - mean)**2 * weights) / (len(values)-1)
    else:
        variance = 1 / sum(weights)

    return mean, sqrt(variance)


def exp_moving_average(new_value, old_average=None, N=10, alpha=None):
    """Derive the exp. moving average from the old average and a new
    value.

    Parameters
    ----------
    new_value : int or float or ndarray
        The new value to be used for averaging.
    old_average : float or ndarray or None
        The previous result of the averaging. If 'None', function
        will return the new value, which is the first iteration step
        of the averaging.
    N : int (optional)
        Number of periods for averaging process. 86% of the averaging
        weight is contained in the last N values. Defaults to 10.
    alpha : float (optional)
        Weight of the new value for the average. If alpha is None,
        this value is calculated from N by $\alpha = 2/(N+1)$.

    Returns
    -------
    float or ndarray
        New value of the exponential moving average.
    """

    if old_average is None:
        return new_value

    if alpha is None:
        alpha = 2/(N+1)

    return alpha * new_value + (1-alpha) * old_average
