from __future__ import division, print_function, unicode_literals

from math import ceil
import numpy
from scipy import signal
from scipy.optimize import curve_fit
from scipy.integrate import quad
from scipy.constants import k

from lmfit import Model
from lmfit.models import update_param_vals, height_expr

from ..theory.damping_heating import gas_damping
from ..theory.particle import particle_mass

def derive_cross_spectrum(x, y, sampling_rate, subdivision_factor, return_onesided=True, normalize=False):
    """ Derives the Cross spectrum between timetraces x and y. Based on simple FFT method. 
    When entered with identical timetraces, the result is dentical to that of "derive_psd".
    
    Parameters
    ----------
    x,y: Input time traces
    sampling_rate, subdivision_factor, return_onesided : like for "derive_psd" method
    normalize: If set to true, the resulting spectrum is devided by the marginals (Sxy/sqrt(Sxx * Syy))
    """
    assert len(x) == len(y), "X and Y need to have the same length."
    part_length = len(x)//subdivision_factor
    #x = [x[i*part_length:(i+1)*part_length]
    #            for i in range(subdivision_factor)]
    #y = [y[i*part_length:(i+1)*part_length]
    #            for i in range(subdivision_factor)]
    #cross_sepctrum = []
    #
    #frequency_step = sampling_rate/part_length
    #
    #for i in range(subdivision_factor):
    #    fft_x = numpy.fft.fftshift(numpy.fft.fft(x[i])/len(x[i]))
    #    fft_y = numpy.fft.fftshift(numpy.fft.fft(y[i])/len(y[i]))
    #    
    #    if return_onesided:
    #        # Drop negative Frequencies and correct at f=0, multiply by sqrt(2) since Sx = 2Sxx
    #        fft_x = numpy.sqrt(2)*fft_x[len(fft_x)//2:]
    #        fft_x[0] /= numpy.sqrt(2)
    #        fft_y = numpy.sqrt(2)*fft_y[len(fft_y)//2:]
    #        fft_y[0] /= numpy.sqrt(2)
    #            
    #    # Derive Cross Spectrum
    #    crsp = numpy.conj(fft_x) * fft_y / frequency_step
    #    if normalize == True:
    #        crsp /= (numpy.abs(fft_x) * numpy.abs(fft_y) / frequency_step)
    #    cross_sepctrum.append(crsp)
    #        
    #cross_sepctrum = numpy.average(cross_sepctrum, axis=0)

    segment_size = int(len(x)/subdivision_factor)

    # Derive single-sided PSD using Welch's method
    frequency, cross_sepctrum = signal.csd(
        x, y, sampling_rate,
        nperseg=segment_size, return_onesided=return_onesided)

    if not return_onesided:
        frequency = numpy.fft.fftshift(frequency)
        cross_sepctrum = numpy.fft.fftshift(cross_sepctrum)
    
    return cross_sepctrum, frequency
    
def derive_psd(data, sampling_rate, method='welch',
               subdivision_factor=1, nfft_factor=None, 
               pad_zeros_pow2=False, return_onesided=True, 
               detrend=None, cut_to_power2 = False):
    # type: (numpy.ndarray, float, str, int, float) -> (numpy.ndarray, numpy.ndarray)
    """Derive the power spectral density from a time trace.

    There are two methods available for deriving the PSD: standard FFT
    and the Welch method that offers windowing to avoid spectral
    leakage.

    Note
    ----
    The Welch method uses the function `scipy.signal.welch and does
    not include the DC component of the signal.

    Parameters
    ----------
    data : numpy.ndarray
        1D array of equally spaced time trace data.
    sampling_rate : float
        sampling rate of the equally spaced time trace data
    method : {'fft', 'welch'}, optional
        Method used for derivation of power spectral density. Either
        'fft' for using standard FFT, or 'welch' for Welch's
        windowing method.
    subdivision_factor : int, optional
        The subdivision factor used for deriving the size of the
        window for the 'welch' method. For the 'fft' method, the time
        trace will be devided into this number of segments.
    nfft_factor : float, optional
        Factor of how much longer the spectrum vector should be
        compared to the data vector. For values >1 the data vector
        is zero-padded. Defaults to None (no change in vector length).
    detrend : optional
        When using the 'welch' method, this argument is used there.
    cut_to_power2: bool, optional
        The required frequency resolution fixes the number of points per segment
        in the Welch estimation of the power spectral density. It is computationally
        advantageous, however, to have the number of points given by a power of two.
        It cut_to_power2 is True, then the segment size is adjusted accordingly.
        Defaults to False.
    Returns
    -------
    numpy.ndarray
        power spectral density for the time trace data
    numpy.ndarray
        frequency array of the PSD

    Example
    -------
    The function may be used as follows:

        >>> from numpy import linspace, random, sin, pi
        >>> from matplotlib.pyplot import plot, yscale
        >>>
        >>> t = linspace(0, 100, 10001)[:-1]
        >>> sampling_rate = 1/(t[1]-t[0])
        >>>
        >>> a0 = 4; a1 = 3; f1 = 10; a2 = 2; f2 = 20
        >>> x = a0 + a1*sin(2*pi*f1*t) + a2*sin(2*pi*f2*t) + \
        >>>     (random.rand(len(t))-0.5)
        >>>
        >>> psd, frequency = derive_psd(
        >>>     x, sampling_rate, method='fft', subdivision_factor=10)
        >>>
        >>> plot(frequency, psd)
        >>> yscale('log')

    To see the effect of the method, try subdivision factor of 10
    and 16 for both methods ('fft' and 'welch'). For 'fft' you will
    see spectral leakage with a subdivision factor of 16, not so for
    'welch'. Thus 'welch' is usually the better option. But note
    that 'welch' does not include the DC component of the signal!
    """

    if pad_zeros_pow2 and subdivision_factor == 1:
        defifict = int(2 ** numpy.ceil(numpy.log2(len(data))) - len(data))
        data = numpy.pad(data, (defifict, 0), mode='constant')
    if method.lower() in 'fft':
        # subdivide time trace
        part_length = len(data)//subdivision_factor
        data = [data[i*part_length:(i+1)*part_length]
                for i in range(subdivision_factor)]
        psd = []

        # Derive Frequency Step
        if nfft_factor is not None:
            n = int(part_length*nfft_factor)
        else:
            n = part_length
        frequency_step = sampling_rate/n

        for i in range(subdivision_factor):
            if nfft_factor is not None:
                fft_data = numpy.fft.fftshift(numpy.abs(
                    numpy.fft.fft(data[i], n=n))/n)
            else:
                fft_data = numpy.fft.fftshift(numpy.abs(
                    numpy.fft.fft(data[i]))/len(data[i]))

            if return_onesided:
                # Drop negative Frequencies and correct at f=0, multiply by sqrt(2) since Sx = 2Sxx
                fft_data = numpy.sqrt(2)*fft_data[len(fft_data)//2:]
                fft_data[0] /= numpy.sqrt(2)


            # Derive Power Spectral Density
            psd.append((fft_data)**2 / frequency_step)

        psd = numpy.average(psd, axis=0)

        if return_onesided:
            frequency = numpy.linspace(0, frequency_step*(len(psd)-1),
                                   len(psd))
        else:
            frequency = numpy.linspace(- frequency_step*(len(psd)-1)/2, frequency_step*(len(psd)-1)/2, len(psd))


    elif method.lower() in 'welch':
        # Derive size of segments
        segment_size = int(len(data)/subdivision_factor)
        
        if cut_to_power2:
            segment_size = int(2**(ceil(numpy.log2(segment_size))))
        
        # Derive single-sided PSD using Welch's method
        if nfft_factor is not None:
            frequency, psd = signal.welch(
                data, sampling_rate,
                nperseg=segment_size, return_onesided=return_onesided,
                nfft=int(len(data)*nfft_factor))
        else:
            frequency, psd = signal.welch(
                data, sampling_rate,
                nperseg=segment_size, return_onesided=return_onesided, detrend = detrend)

        if not return_onesided:
            frequency = numpy.fft.fftshift(frequency)
            psd = numpy.fft.fftshift(psd)

    elif method.lower() in 'rfft':
        # Derive size of segments
        sample_size = len(data) // subdivision_factor
        if sample_size == 0:
            sample_size = 1
            subdivision_factor = len(data)
            print('Subdivision factor bigger than sample size. Reducing subdivision factor to maximal value.')

        # Make Sample Size compatible with division factor
        data = data[:sample_size * subdivision_factor]

        data_split = numpy.array(numpy.hsplit(data, subdivision_factor))

        if nfft_factor is not None:
            n = int(sample_size * nfft_factor)
        else:
            n = sample_size


        if pad_zeros_pow2 and subdivision_factor != 1:
            if nfft_factor is not None:
                n = int(2 ** numpy.ceil(numpy.log2(len(data_split[0]) * nfft_factor )))
            else:
                n = int(2 ** numpy.ceil(numpy.log2(len(data_split[0]))))
            psd = numpy.mean(abs(numpy.fft.rfft(data_split, n=n, axis=1)) ** 2, axis=0)
        else:
            psd = numpy.mean(abs(numpy.fft.rfft(data_split, n=n, axis=1)) ** 2, axis=0)
        psd *= 2 / (sampling_rate * n)
        psd[0] /= 2

        frequency = numpy.fft.rfftfreq(n, 1 / sampling_rate)

    else:
        raise ValueError('invalid method')

    return psd, frequency

def fit_overdamped_psd(psd,frequency,guess_amplitude = None, guess_cut_off_frequency = None, 
                        fit_scaling = 'linear',fit_offset = True, plot_fit = False):
    """Fit an overdamped Lorentzian to the power spectral density.

    Before the fit is performed, guess parameters are derived from
    the PSD (or specified by parameters). The fit can either be performed with linear
    scaling, or with logarithmic scaling. For the latter
    10*log10(data) is taken for the fit.
    
    Parameters:
    ----------
    psd : numpy.ndarray
        1D array of the power spectral density which shall be fitted.
    frequency : numpy.ndarray
        1D array of the frequencies corresponding to the PSD.
    guess_amplitude : float, optional
        Guess value for the amplitude. If not specified,
        it will be estimated from the PSD data.
    guess_cut_off_frequency : float, optional
        Guess value for the cut-off frequency. If not specified,
        it will be estimated from the PSD data.
    fit_scaling : {'log', 'linear'}, optional
        Use logarithmic, or linear scaling of the data for fitting.
        Defaults to 'log'.
    fit_offset : bool, optional
        Include an additional offset term to the fit model to account
        for a noise floor. Defaults to True.

    Returns
    -------
    numpy.ndarray
        Array of the fit parameters ([amplitude, cut_off_frequency,
        offset]).
    numpy.ndarray
        2D array with the covariance matrix.
    """
    
    if guess_amplitude is not None and guess_cut_off_frequency is not None:
        guess = [guess_amplitude,guess_cut_off_frequency]
        
    elif guess_amplitude is not None:
        guess = [guess_amplitude] + guess_damped_lorentzian_fit_parameters(frequency,
                                                                         psd,
                                                                         parameter = 'fc')
    
    elif guess_cut_off_frequency is not None:
        guess_fc = guess_damped_lorentzian_fit_parameters(frequency,
                                                          psd,
                                                          parameter = 'amp')
        guess = [guess_fc,guess_cut_off_frequency]
    
    else:
        guess = guess_damped_lorentzian_fit_parameters(frequency,psd)
        
    if fit_offset:
        guess.append(0)
    try:
        if fit_scaling == 'log':
            params, cov = curve_fit(log_damped_lorentzian,frequency,psd,guess, bounds = (0, numpy.inf))
        elif fit_scaling == 'linear':
            params, cov = curve_fit(damped_lorentzian,frequency,psd,guess, bounds = (0, numpy.inf))
        else:
            print('Invalid fit_scaling parameter. Only choices are \'log\' and \'linear\'')
            params, cov = None, None
    except RuntimeError:
        print('ERROR: fit failed')
        params, cov = None, None
    
    if plot_fit:
    
        import matplotlib.pyplot as plt
        from ..visualization.set_axes import set_ax
        fit_curve = damped_lorentzian(frequency, *params)

        fig = plt.figure(figsize = (8,4))
        ax = fig.add_subplot(111)
        ax.semilogy(frequency/1000, psd, 'C0', label='PSD')
        ax.semilogy(frequency/1000, fit_curve, 'C1', label='Fit')
        xlabel = 'Frequency (kHz)'
        ylabel = 'Power Spectral Density'
        font_size = 18
        set_ax(ax,xlabel = xlabel, ylabel = ylabel, fs = font_size, legend = True)
        plt.show()
    
    return params, cov
    
def fit_psd(psd, frequency, guess_f0=None, guess_fwhm=None,
            guess_amplitude=None, fit_window=None, fit_scaling='log',
            fit_offset=True, plot_fit=False):
    # type: (numpy.ndarray, numpy.ndarray, float, float, float, float, str, bool, bool) -> (numpy.ndarray, numpy.ndarray)
    """Fit a Lorentzian to the power spectral density.

    Before the fit is preformed, quess parameters are derived from
    the PSD (or specified by parameters). A window for fitting can
    be specified. The fit can either be performed with linear
    scaling, or with logarithmic scaling. For the latter
    10*log10(data) is taken for the fit. For monitoring, the fit
    together with the data may be plotted.

    Parameters
    ----------
    psd : numpy.ndarray
        1D array of the power spectral density which shall be fitted.
    frequency : numpy.ndarray
        1D array of the frequencies corresponding to the PSD.
    guess_f0 : float, optional
        Guess value for the center frequency. If not specified,
        it will be estimated from the PSD data.
    guess_fwhm : float, optional
        Guess value for the FWHM of the peak. If not specified,
        it will be estimated from the PSD data.
    guess_amplitude : float, optional
        Guess value for the amplitude of the peak. If not specified,
        it will be estimated from the PSD data.
    fit_window : float, optional
        Width of the window used for fitting. If not specified,
        the entire PSD will be used. The window is centered around
        quess_f0 (specified or calculated).
    fit_scaling : {'log', 'linear'}, optional
        Use logarithmic, or linear scaling of the data for fitting.
        Defaults to 'log'.
    fit_offset : bool, optional
        Include an additional offset term to the fit model to account
        for a noise floor. Defaults to True.
    plot_fit : bool, optional
        Plot the fit together with the data. Defaults to False.

    Returns
    -------
    numpy.ndarray
        Array of the fit parameters ([amplitude, damping, frequency,
        offset]).
    numpy.ndarray
        2D array with the covariance matrix.
    """

    # Guess fit parameters
    if guess_amplitude is not None and guess_f0 is not None and \
            guess_fwhm is not None:
        guess = [guess_amplitude, guess_fwhm, guess_f0]
    elif guess_f0 is not None and guess_fwhm is not None:
        guess_amplitude = psd[numpy.where(
            frequency >= guess_f0)[0][0]] * guess_fwhm**2 * guess_f0**2
        guess = [guess_amplitude, guess_fwhm, guess_f0]
    elif guess_f0 is not None:
        guess = guess_lorentzian_fit_parameters(frequency, psd)
        guess[2] = guess_f0
    else:
        guess = guess_lorentzian_fit_parameters(frequency, psd)

    if fit_offset:
        guess.append(0)

    # Fit window
    if fit_window is not None:
        frequency, psd = get_psd_window(frequency, psd, guess[2],
                                        fit_window)

    # Perform fit
    try:
        if fit_scaling in 'log':
            params, cov = fit_log_lorentzian(frequency, psd, guess)
        elif fit_scaling in 'linear':
            params, cov = fit_lorentzian(frequency, psd, guess)
        else:
            params, cov = None, None
    except RuntimeError:
        print("ERROR: fit failed")
        params, cov = None, None

    # Plot fit
    if plot_fit and params is not None:
        import matplotlib.pyplot as plt
        fit_curve = lorentzian(frequency, *params)

        plt.figure()
        plt.plot(frequency/1000, psd, 'b.', label='PSD')
        plt.plot(frequency/1000, fit_curve, 'r', label='Fit')
        plt.xlim((min(frequency)/1000, max(frequency)/1000))
        plt.yscale('log')
        plt.xlabel('Frequency (kHz)')
        plt.ylabel('Power Spectral Density')
        plt.show()

    return params, cov

def get_psd_window(frequency, psd, center, width):
    frequency_low = center - width/2
    frequency_high = center + width/2

    index_range = numpy.arange(
        numpy.where(frequency >= frequency_low)[0][0],
        numpy.where(frequency <= frequency_high)[0][-1])

    return frequency[index_range], psd[index_range]

def lorentzian(x, amplitude, gamma, center, offset=0.0):
    """Returns the Lorentzian Distribution on a linear scale."""
    return amplitude/((x**2 - center**2)**2 + gamma**2 * x**2) + offset

def damped_lorentzian(f,amplitude,cut_off_frequency,offset = 0.0):
    """Returns the overdamped limit of the Lorentzian Distribution on a linear scale."""   
    return amplitude/(cut_off_frequency**2 + f**2) + offset

def log_lorentzian(f, a1, a2, a3, a4=0):
    """Returns the Lorentzian Distribution on a dB-scale."""
    # return 10*numpy.log10(a1/((f**2 - a3**2)**2 + a2**2 * f**2) + a4)
    return 10*numpy.log10(lorentzian(f, a1, a2, a3, a4))

def log_damped_lorentzian(f,amplitude,cut_off_frequency,offset = 0):
    """Returns the overdamped limit of the Lorentzian Distribution on a dB-scale."""
    return 10*numpy.log10(damped_lorentzian(f,amplitude,cut_off_frequency,offset = offset))
    
def guess_damped_lorentzian_fit_parameters(frequency, psd, parameter = 'all'):
    """Returns a guess for amplitude and cut_off_frequency of a psd.
    ------------------
    Parameters:
        frequency: numpy.ndarray
            1D array of frequencies
        psd: numpy.ndarray
            1D array corresponding to the power spectral density
        parameter: string
            Refers to the parameter of interest for estimation:
            - if 'all' the method estimates both amplitude and cut-off frequency
            - if 'amp' the method estimates only the amplitude
            - if 'fc' the method estimates only the cut-off frequency
    ------------------
    Returns:
        list of:
            - amplitude if parameter == 'amp'
            - cut-off frequency if parameter == 'fc'
            - amplitude and cut-off frequency if parameter == 'all'
    -----------------
    """
    
    output = []
    if parameter == 'all':
        max_value = numpy.max(psd)
        cut_off_index = numpy.where(psd <= max_value/2)[0][0]
        cut_off_frequency = frequency[cut_off_index]
        amplitude = max_value * cut_off_frequency**2
        output = [amplitude,cut_off_frequency]
        
    elif parameter == 'amp':
        max_value = numpy.max(psd)
        cut_off_index = numpy.where(psd <= max_value/2)[0][0]
        cut_off_frequency = frequency[cut_off_index]
        amplitude = max_value * cut_off_frequency**2
        output = [amplitude]
        
    elif parameter == 'fc':
        max_value = numpy.max(psd)
        cut_off_index = numpy.where(psd <= max_value/2)[0][0]
        cut_off_frequency = frequency[cut_off_index]
        amplitude = max_value * cut_off_frequency**2
        output = [cut_off_frequency]
        
    else:
        print('keyword \'parameter\' is not valid. Only options are \'all\',\'amp\' and \'fc\'')
    
    return output

def guess_lorentzian_fit_parameters(frequency, psd, x=0, cutoff_freq=0):
    """
    """

    if cutoff_freq > 0:
        # Cut out 1/f-peak
        cutoff = numpy.where(frequency >= cutoff_freq)[0][0]

        psd = psd[cutoff:]
        frequency = frequency[cutoff:]

    if x is not None:
        # Get indices of sorted psd array
        sorted_list = sorted(psd,reverse=True)
        index_list = sorted(range(psd.size), key=lambda y: psd[y],
                            reverse=True)

        # Take xth element of indices array as estimate for the peak
        # (try to avoid spikes)
        f_0 = frequency[index_list[x]]

        # Find FWHM of peak
        half_max_index = index_list[
            numpy.where(sorted_list <= sorted_list[x]/2)[0][0]]
        fwhm = 2*abs(f_0-frequency[half_max_index])

        amplitude = sorted_list[x] * fwhm**2 * f_0**2
    else:
        psd_max = numpy.max(psd)

        index_omega_0 = numpy.where(psd == numpy.max(psd))[0][0]
        f_0 = frequency[index_omega_0]

        index_omega_1 = index_omega_0 + numpy.where(
            psd[index_omega_0:] <= psd_max/2)[0][0]
        omega_1 = frequency[index_omega_1]
        fwhm = 2*abs(omega_1-f_0)

        amplitude = psd_max * fwhm**2 * f_0**2

    return [amplitude, fwhm, f_0]


def fit_lorentzian(frequency, psd, guess=(1e8, 10000, 80000, 0)):
    """Fit Lorentzian to the Power Spectral Density using
    least-square-fit.

    arguments:
    freq -- array of frequencies
    psd -- power spectral density
    guess -- list with guesses of the 3/4 fit parameters (i.e.
             generated by guess_lorentzian_fit_parameters)

    returns:
    params -- array of the four parameters from the fit
    """

    # Perform the Fit
    params, cov = curve_fit(lorentzian, frequency, psd, guess)

    return params, cov


def fit_log_lorentzian(frequency, psd, guess=(1e20, 10000, 80000, 0)):
    """Fit Lorentzian to the Power Spectral Density using
    least-square-fit.

    arguments:
    freq -- array of frequencies
    psd -- power spectral density in dB scale
    guess -- list with guesses of the 3/4 fit parameters (i.e.
             generated by guess_lorentzian_fit_parameters)

    returns:
    params -- array of the four parameters from the fit
    """

    # Perform the Fit
    params, cov = curve_fit(log_lorentzian, frequency,
                            10*numpy.log10(psd), guess)

    return params, cov


def find_peaks(x, y, peak_SNR=100, peak_isolation=1000,
               noise_level=None, output_index=False):
    """
    Find peaks (e.g. in a spectrum) with a given isolation and SNR.

    Parameters
    ----------
    x : ndarray
        Independent variable (e.g. frequency).
    y : ndarray
        Dependent variable (e.g. PSD).
    peak_SNR : float, optional
        Minimum SNR, to recognize a peak. Defaults to 100.
    peak_isolation : float, optional
        Range in which to only expect a single peak in units of x.
        Defaults to 1000.
    noise_level : float, optional
        Noise level in units of y. Default is None, in which case
        the noise level is calculated as y.min().
    output_index : bool, optional
        Switch for putting out the indices instead of x-values.
        Defaults to False.

    Returns
    -------
    ndarray
        Array containing the x-values at which there are peaks,
        or the corresponding indices, if output_index=True.
    """
    x_step = numpy.mean(numpy.diff(x))

    if noise_level is None:
        noise_level = y.min()

    peaks_idx = signal.argrelmax(y,
                                 order=int(peak_isolation / x_step))[0]
    peaks_idx = peaks_idx[y[peaks_idx] > peak_SNR * noise_level]

    if output_index:
        return peaks_idx
    else:
        return x[peaks_idx]


def linear_power_spectrum(x, amplitude, center, sigma, offset=0.0,
                          angular=False):
    """Returns the power spectrum of a damped harmonic oscillator."""
    if angular:
        prefactor = 1
    else:
        prefactor = 2
    return prefactor * amplitude * center**2 * sigma / numpy.pi / (
        (x**2 - center**2)**2 + sigma**2 * x**2) + offset


class DampedHarmonicOscillatorPSDModel(Model):
    r"""
    A model for the PSD of a damped harmonic oscillator.

    The model is based on lmfit and made to be used with lmfit,
    also in conjunction with other models. It is supposed to be able
    to guess the initial fitting parameters, given the dataset and
    optionally some hints given by the user.

    Examples
    --------
    The model may be used to fit a PSD (given as ''frequency'' and
    ''psd'') in the following way:

    >>> fit_range = numpy.where(
    >>>     (frequency > 20000) & (frequency < 70000))
    >>>
    >>> mod = DampedHarmonicOscillatorPSDModel()
    >>> pars = mod.guess(psd[fit_range], x=frequency[fit_range],
    >>>                  center=40000)
    >>>
    >>> out = mod.fit(psd[fit_range], pars, x=frequency[fit_range])
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.plot(frequency[fit_range]/1000, psd[fit_range], '.',
    >>>          label='data')
    >>> plt.plot(frequency[fit_range]/1000, out.best_fit, label='fit')
    >>> plt.plot(frequency[fit_range]/1000, out.init_fit, '--',
    >>>          color='grey', linewidth=1, label='guess')
    >>> plt.yscale('log')
    >>> plt.xlabel('Frequency (kHz)')
    >>> plt.ylabel('PSD (m$^2$/Hz)')
    >>> plt.legend()

    If one wants to fit multiple peaks, the models can be added:

    >>> from lmfit.models import LinearModel
    >>>
    >>> mod_peak_z = DampedHarmonicOscillatorPSDModel(
    >>>     prefix='z_', fit_offset=False)
    >>> mod_peak_x_FB = DampedHarmonicOscillatorPSDModel(
    >>>     prefix='x_FB_', fit_offset=False)
    >>> mod_peak_y_FB = DampedHarmonicOscillatorPSDModel(
    >>>     prefix='y_FB_', fit_offset=False)
    >>> mod_peak_z_FB = DampedHarmonicOscillatorPSDModel(
    >>>     prefix='z_FB_', fit_offset=False)
    >>> mod_noise = LinearModel()
    >>> mod_noise.set_param_hint('slope', value=0, vary=False)
    >>>
    >>> mod = mod_peak_z + mod_peak_x_FB + mod_peak_y_FB + \
    >>>     mod_peak_z_FB + mod_noise
    >>>
    >>> pars = mod_peak_z.guess(
    >>>     psd[fit_range], x=frequency[fit_range], center=40000)
    >>> pars += mod_peak_x_FB.guess(
    >>>     psd[fit_range], x=frequency[fit_range], center=75000)
    >>> pars += mod_peak_y_FB.guess(
    >>>     psd[fit_range], x=frequency[fit_range], center=230000)
    >>> pars += mod_peak_z_FB.guess(
    >>>     psd[fit_range], x=frequency[fit_range], center=265000)
    >>> pars += mod_noise.make_params(
    >>>     intercept=psd[fit_range].min(), slope=0)
    >>>
    >>> out = mod.fit(psd[fit_range], pars, x=frequency[fit_range])
    """

    log_scaling = False

    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None, fit_offset=True, angular=False, **kwargs):

        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars,
                       'angular': angular})

        super(DampedHarmonicOscillatorPSDModel, self).__init__(
            linear_power_spectrum, **kwargs)

        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('center', min=0)
        self.set_param_hint('sigma', min=0)

        if fit_offset:
            pass
            # self.set_param_hint('offset', min=0)
            # min=0 does not work well.
        else:
            self.set_param_hint('offset', value=0, vary=False)

        self.set_param_hint('fwhm', expr=
            ('sqrt({prefix:s}center**2-{prefix:s}sigma**2/2 + '
             'sqrt({prefix:s}center**2*{prefix:s}sigma**2+'
             '{prefix:s}sigma**4/4)) - '
             'sqrt({prefix:s}center**2-{prefix:s}sigma**2/2 - '
             'sqrt({prefix:s}center**2*{prefix:s}sigma**2+'
             '{prefix:s}sigma**4/4))').format(
                prefix=self.prefix))
        self.set_param_hint('height', expr=height_expr(self))

    @property
    def height_factor(self):
        if self.opts['angular']:
            return 1. / numpy.pi
        else:
            return 2. / numpy.pi

    def guess(self, y, x=None, center=None, sigma=None, offset=None,
              peak_SNR=100, peak_isolation=1000, **kwargs):
        """
        Guess the initial fitting parameters for a damped harmonic
        oscillator model. The guessing can be supported by giving a
        rough estimate of the center frequency. If the linewidth
        gamma or the offset are given, those values will be taken as
        initial values directly.

        Parameters
        ----------
        y : ndarray
        x : ndarray
        center : float, optional
        sigma : float, optional
        offset : float, optional
        peak_SNR : float, optional
        peak_isolation : float, optional
        kwargs

        Returns
        -------
        dict
            Parameters that have been guessed.

        """
        maxy, miny = y.max(), y.min()

        if offset is None:
            offset = miny

        peaks_idx = find_peaks(x, y, peak_SNR=peak_SNR,
                               peak_isolation=peak_isolation,
                               noise_level=offset, output_index=True)

        if center is None:
            imaxy = numpy.abs(y - maxy).argmin()
            center = x[imaxy]
        else:
            center_idx = numpy.abs(x - center).argmin()
            center = x[
                peaks_idx[numpy.abs(peaks_idx - center_idx).argmin()]]

        # if sigma is None:
        #     center_idx = numpy.abs(x - center).argmin()
        #     smaller_than_peak_idx = \
        #     numpy.where(y < (y[center_idx] + offset) / numpy.e)[0]
        #     sigma = (x[smaller_than_peak_idx[
        #         smaller_than_peak_idx - center_idx > 0][0]] -
        #              x[smaller_than_peak_idx[
        #                  smaller_than_peak_idx - center_idx < 0][-1]])

        amplitude = numpy.sum(y) * (x[1]-x[0])

        if sigma is None:
            center_idx = numpy.abs(x - center).argmin()

            if self.opts['angular']:
                sigma = amplitude / (numpy.pi * y[center_idx])
            else:
                sigma = 2 * amplitude / (numpy.pi * y[center_idx])

        pars = self.make_params(amplitude=amplitude, center=center,
                                sigma=sigma, offset=offset)
        return update_param_vals(pars, self.prefix, **kwargs)

    def eval(self, *args, **kwargs):
        if self.log_scaling:
            return 10*numpy.log10(super(
                DampedHarmonicOscillatorPSDModel, self).eval(
                *args, **kwargs))
        else:
            return super(DampedHarmonicOscillatorPSDModel, self).eval(
                *args, **kwargs)

    def fit(self, data, *args, log_fit=False, **kwargs):
        if log_fit:
            self.log_scaling = True
            output = super(DampedHarmonicOscillatorPSDModel, self).fit(
                10 * numpy.log10(data), *args, **kwargs)
            output.init_fit = 10 ** (output.init_fit / 10)
            output.best_fit = 10 ** (output.best_fit / 10)
            self.log_scaling = False
            return output
        else:
            return super(DampedHarmonicOscillatorPSDModel, self).fit(
                data, *args, **kwargs)


def nonlinear_power_spectrum(x, amplitude, center, sigma, duffing,
                             offset=0, angular=False, max_energy=None):
    """Returns the power spectrum of a damped Duffing oscillator.
    
    Parameters
    ----------
    x
    amplitude
    center
    sigma
    duffing
    offset
    angular
    max_energy

    Returns
    -------

    """
    if angular:
        prefactor = 1
    else:
        prefactor = 2

    def center_energy(x, amplitude, center, duffing):
        if duffing != 0:
            return (x - center) * 4 / (3 * amplitude * duffing * center)
        else:
            return 0

    def shifted_frequency(E, amplitude, center, duffing):
        return center * (1 + 3 * duffing * amplitude * E / 4)

    @numpy.vectorize
    def vect_function(x, amplitude, center, sigma, duffing, offset,
                      max_energy):
        delta = numpy.sign(duffing) * (x - center)

        e_center = max(center_energy(x, amplitude, center, duffing), 0)
        e_center_plus = max(
            center_energy(x + sigma, amplitude, center, duffing), 0)
        e_center_minus = max(
            center_energy(x - sigma, amplitude, center, duffing), 0)

        if max_energy is None:
            max_energy = max(10, 2 * e_center)

        if delta > 0:
            break_point = (e_center,
                           min(max_energy, e_center_plus),
                           min(max_energy, max(0, e_center_minus)))
        else:
            break_point = ()

        return prefactor * amplitude*center**2*sigma / numpy.pi * quad(
            lambda E: E * numpy.exp(-E) / ((x ** 2 - shifted_frequency(
                E, amplitude, center, duffing) ** 2) ** 2 +
                x ** 2 * sigma ** 2),
            0, max_energy, points=break_point, epsrel=1e-10)[0] + offset

    return vect_function(x, amplitude, center, sigma, duffing, offset,
                         max_energy)


class DampedNonlinearOscillatorPSDModel(Model):
    r"""
    A model for the PSD of a damped Duffing oscillator.

    The model is based on lmfit and made to be used with lmfit,
    also in conjunction with other models. It is supposed to be able
    to guess the initial fitting parameters, given the dataset and
    optionally some hints given by the user.
    """

    log_scaling = False

    def __init__(self, independent_vars=['x'], prefix='', missing=None,
                 name=None, fit_offset=True, negative_duffing=True,
                 **kwargs):

        kwargs.update({'prefix': prefix, 'missing': missing,
                       'independent_vars': independent_vars})

        super(DampedNonlinearOscillatorPSDModel, self).__init__(
            nonlinear_power_spectrum, **kwargs)

        self.set_param_hint('amplitude', min=0)
        self.set_param_hint('center', min=0)
        self.set_param_hint('sigma', min=0)

        if negative_duffing:
            self.set_param_hint('duffing', max=0)

        if fit_offset:
            pass
            # self.set_param_hint('offset', min=0)
            # min=0 does not work well.
        else:
            self.set_param_hint('offset', value=0, vary=False)

    def guess(self, y, x=None, center=None, sigma=None,
              duffing=None, offset=None, peak_SNR=100,
              peak_isolation=1000, pressure=None, radius=None,
              density=None, real_duffing=None, **kwargs):
        """
        Guess the initial fitting parameters for a damped Duffing
        oscillator model. The guessing can be supported by giving a
        rough estimate of the center frequency. If the linewidth
        gamma or the offset are given, those values will be taken as
        initial values directly.

        Parameters
        ----------
        y : ndarray
        x : ndarray
        center : float, optional
        sigma : float, optional
        duffing : float, optional
        offset : float, optional
        peak_SNR : float, optional
        peak_isolation : float, optional
        pressure : float, optional
        radius : float, optional
        density : float, optional
        real_duffing : float, optional
        kwargs

        Returns
        -------
        dict
            Parameters that have been guessed.

        """
        maxy, miny = y.max(), y.min()

        if offset is None:
            try:
                if self.param_hints['offset']['vary'] is True:
                    offset = miny
                else:
                    offset = self.param_hints['offset']['value']
            except KeyError:
                offset = miny


        peaks_idx = find_peaks(x, y, peak_SNR=peak_SNR,
                               peak_isolation=peak_isolation,
                               noise_level=offset, output_index=True)

        if center is None:
            imaxy = numpy.abs(y - maxy).argmin()
            center = x[imaxy]
        else:
            center_idx = numpy.abs(x - center).argmin()
            center = x[
                peaks_idx[numpy.abs(peaks_idx - center_idx).argmin()]]

        center_idx = numpy.abs(x - center).argmin()
        smaller_than_peak_idx = \
            numpy.where(y < (y[center_idx] + offset) / 2)[0]
        delta_left = x[center_idx] - x[smaller_than_peak_idx[
            smaller_than_peak_idx - center_idx < 0][-1]]
        delta_right = x[smaller_than_peak_idx[
            smaller_than_peak_idx - center_idx > 0][0]] - x[center_idx]

        if sigma is None:
            if pressure is not None:
                sigma = gas_damping(pressure=100 * pressure,
                                    radius=radius, density=density) / (
                        2 * numpy.pi)
            else:
                sigma = 2 * min(delta_left, delta_right)

                # amplitude = y[numpy.where(x >= center)[0][
                #     0]] * sigma * pi/2
        amplitude = sum(y) * (x[1] - x[0])

        # if duffing is None:
        #     duffing_sign = sign(delta_right - delta_left)
        #     duffing = (duffing_sign*(delta_right+delta_left-sigma) *
        #                4*pi/(3*pi-8)/3 / amplitude / center)
        if duffing is None:
            duffing = k * 300 / (2 * particle_mass(radius, density) *
                                 center ** 2) * real_duffing / amplitude

        pars = self.make_params(amplitude=amplitude, center=center,
                                sigma=sigma, duffing=duffing,
                                offset=offset)
        return update_param_vals(pars, self.prefix, **kwargs)

    def eval(self, *args, **kwargs):
        if self.log_scaling:
            return 10 * numpy.log10(super(
                DampedNonlinearOscillatorPSDModel, self).eval(
                *args, **kwargs))
        else:
            return super(DampedNonlinearOscillatorPSDModel, self).eval(
                *args, **kwargs)

    def fit(self, data, *args, log_fit=False, **kwargs):
        if log_fit:
            self.log_scaling = True
            output = super(DampedNonlinearOscillatorPSDModel, self).fit(
                10 * numpy.log10(data), *args, **kwargs)
            output.init_fit = 10 ** (output.init_fit / 10)
            output.best_fit = 10 ** (output.best_fit / 10)
            self.log_scaling = False
            return output
        else:
            return super(DampedNonlinearOscillatorPSDModel, self).fit(
                data, *args, **kwargs)


def cavity_lorentzian(x, height=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function with the height as a parameter instead of the amplitude.

    lorentzian(x, height, center, sigma) =
        (height/(1 + ((1.0*x-center)/sigma)**2))

    """
    return (height/(1 + ((1.0*x-center)/sigma)**2))

from lmfit.models import fwhm_expr, guess_from_peak, COMMON_INIT_DOC, COMMON_GUESS_DOC

class CavityLorentzianModel(Model):
    r"""A model based on a Lorentzian or Cauchy-Lorentz distribution function
    (see https://en.wikipedia.org/wiki/Cauchy_distribution), with three Parameters:
    ``amplitude``, ``center``, and ``sigma``.
    In addition, parameters ``fwhm`` and ``height`` are included as constraints
    to report full width at half maximum and maximum peak height, respectively.

    .. math::

        f(x; A, \mu, \sigma) = h \big[\frac{\sigma^2}{(x - \mu)^2 + \sigma^2}\big]

    where the parameter ``height`` corresponds to :math:`A/(pi * sigma)`, ``center`` to
    :math:`\mu`, and ``sigma`` to :math:`\sigma`.  The full width at
    half maximum is :math:`2\sigma`.

    """

    fwhm_factor = 2.0
    height_factor = 1./numpy.pi

    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        super(CavityLorentzianModel, self).__init__(cavity_lorentzian, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        self.set_param_hint('sigma', min=0)
        self.set_param_hint('fwhm', expr=fwhm_expr(self))

    # def guess(self, data, x=None, negative=False, **kwargs):
    #     """Estimate initial model parameter values from data."""
    #     pars = guess_from_peak(self, data, x, negative, ampscale=1.25)
    #     return update_param_vals(pars, self.prefix, **kwargs)

    __init__.__doc__ = COMMON_INIT_DOC
    # guess.__doc__ = COMMON_GUESS_DOC
