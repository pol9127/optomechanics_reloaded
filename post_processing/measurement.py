"""
@author: Andrei Militaru
@date: 23rd January 2019
@description: 
    A measurement class for post-processing timetraces.
    The default settings are specific for csv files saved with Lecroy Waverunner scopes.
"""

import numpy as np
import pandas as pd
from ..post_processing import spectrum, calibration
import matplotlib.pyplot as plt
from warnings import warn

class timetrace():

    def __init__(self,
                 name,
                 path = None,
                 underdamped = True,
                 channels = ['C3'],
                 skiprows = 4,
                 fs = None,
                 time = 'Time',
                 amp = 'Ampl',
                 pressure = None,
                 temperature = (297.5,1),
                 **kwargs):
                 
        """
        Parameters:
        -----------
        path: string
            location of the measurement file
        name: string, optional
            name of the measurement file without channel prefix.
            If None, an instance can be instantiated at a later time.
                Example: C1xchannel0000.csv, name = "xchannel0000.csv"
        underdamped: boolean
            information used for the fit during the calibration.
        channels: list of channels
            file name is "path/" + channels[i] + "name" for i in channels.
        skiprows: integer
            number of lines from which the meaningful data start
        fs: float, optional
            sampling frequency. If None, it can be reconstructed from self.fs()
        time: string
            name of the time column. For multiple channels, the time is assumed
            to be the same for all.
        amp: string
            name of the second column
        pressure: tuple of floats
            pressure in mbar at which the timetrace has been recorded.
            pressure[0] corresponds to the estimated value, error bar is pressure[1].
            If not given, the error bar is assumed to be 0.
        temperature: tuple of floats
            temperature at which the timetrace has been recorded.
            temperature[0] is the estimated value, temperatur[1] is the error bar.
        kwargs: key arguments for pandas.read_csv()
        -----------
        """
        
        self.path = '' if path is None else path
        self.name = name
        self.underdamped = underdamped
        self.channels = channels
        if pressure is not None:
            if type(pressure) is not tuple:
                pressure = (pressure,0)
            self.pressure = pressure
        else:
            self.pressure = None
        if temperature is not None:
            if type(temperature) is not tuple:
                temperature = (temperature,0)
            self.temperature = temperature
        self.timetraces = []
        time_initialized = False
            
        for attribute in channels:
        
            filename = self.path + attribute + self.name
            scope = pd.read_csv(filename, skiprows = skiprows,**kwargs)
            if time is not None and not time_initialized:
                self.time = np.array(scope[time])
                time_initialized = True
            self.timetraces.append(np.array(scope[amp]))
            
        if not time_initialized and fs is not None:
            self.fs = fs
            dt = 1/self.fs
            N = len(self.timetraces[0])
            T = dt*N
            self.time = np.linspace(0,T,N)
        elif time_initialized and fs is None:
            self.fs = 1/(self.time[1]-self.time[0])
        elif time_initialized and fs is not None:
            print('Warning: both sampling frequency and time axis have been provided.')
            print('The value calculated from the time axis will be kept to avoid conflicts.')
            self.fs = 1/(self.time[1]-self.time[0])
        else:
            print('Warning: the sampling frequency is not defined at this point.')
            
        self.calibrations = {}
        parameters = {'is_calibrated'  : False,
                      'c_calibrated'   : None,
                      'R_factor'       : None,
                      'particle_size'  : None,
                      'particle_mass'  : None,
                      'damping'        : None,      # measured in [rad s^-1]
                      'fits'           : None}      #[amplitude, damping, frequency,offset]
        
        for items in channels:
            self.calibrations[items] = parameters.copy()
    
    def get_attribute(self,quantity = 'C3'):
        """ Retrieves timetrace corresponding to quantity.
        --------------------
        Parameters:
            quantity: string
                quantity to be retrieved
        -------------------
        Returns:
            numpy.ndarray: 1D timetrace corresponding to quantity
        """
        
        if quantity in self.channels:
            indx = self.channels.index(quantity)
            return self.timetraces[indx]
        else:
            print('Error: the desired quantity does not belong to this object.')
            return None
    
    def truncate_to_power2(self, quantity = 'C3', adjust_time = True):
        if len(self.channels) > 1:
            warn('More channels are present, might need to recalibrate the time.')
        trace = self.get_attribute(quantity = quantity)
        N = len(trace)
        new_N = 2**int(np.log2(N))
        trace = trace[:new_N]
        indx = self.channels.index(quantity)
        self.timetraces[indx] = trace
        if adjust_time:
            self.time = self.time[:new_N]
        return self
    
    def join(self,other):
        self.channels += other.channels
        for traces in other.timetraces:
            self.timetraces.append(traces)
     
    @staticmethod
    def get_slice(vec,
                  ext1,
                  ext2):
        """
        -------------------
        Parameters:
            vec: 1D numpy.ndarray of ordered values
                vector that needs to be sliced
            ext1: float
                first extreme of the slice
            ext2: float
                second extreme of the slice
        --------------------
        Returns:
            output: numpy array
                arg1 index where vec[arg1] == ext1
                arg2 index where vec[arg2] == ext2
                output = vec[arg1:arg2]
        -------------------
        """
        
        output = []
        for i in range(0,len(vec)):
            if ((vec[i] > ext1) and (vec[i] < ext2)):
                output.append(vec[i])
            else:
                continue
        return np.array(output)
    
    @staticmethod
    def ext_indexes(vec,
                    ext1,
                    ext2):
        """
        -------------------
        Parameters:
            vec: 1D numpy.ndarray of ordered values
                vector whose indexes must be found
            ext1: float
                first value of interest
            ext2: float
                second value of interest
        --------------------
        Returns:
            out1,out2: float numbers
                out1 index where vec[out1] == ext1
                out2 index where vec[out2] == ext2
        -------------------
        """
        
        out1 = 0
        out2 = 0
        
        for i in range(0,len(vec)):
            if i == 0:
                if vec[i] > ext1:
                    out1 = i
                else:
                    continue
            elif i == (len(vec)-1):
                if vec[-1] < ext2:
                    out2 = i
                else:
                    continue
            else:
                if (vec[i] >= ext1 and vec[i-1] < ext1):
                    out1 = i
                elif (vec[i] >= ext2 and vec[i-1] < ext2):
                    out2 = i
                else:
                    continue
        return out1,out2
            
    def fourier(self, 
                quantity = None):
        """
        Function that calculates the Fourier transform of quantit.
        It returns both the frequency and the bilateral transform.
        --------------
        Parameters:
            quantity: string or None
                which timetrace of self needs to be transformed.
                If None, the first timetrace from self.timetraces is taken
        --------------
        Returns:
            f,fftx: numpy arrays
                f is the bilateral frequency array
                fftx is the Fourier transform of self.quantity
        --------------
        """
        
        t = self.time
        x = self.get_attribute(quantity) if quantity is not None else self.timetraces[0]
        dt = t[1]-t[0]
        f = np.linspace(-1/(2*dt),1/(2*dt),len(t))
        fftx = np.fft.fftshift(np.fft.fft(x))
        return f,fftx  

    def num(self,tmm):
        """
        ----------
        Returns 
            out: integer
                number of elements from self.time that cover a time interval of tmm.
                Units are assumed to be miliseconds.
        ------------
        """
        dt = self.time[1]-self.time[0]
        return int(tmm*1e-3//dt)    

    def get_subdivision_factor(self,
                               frequency_resolution,
                               mode = 'subdivision_factor'):
        '''
        --------------
        Parameters:
            frequency_resolution: float
                for the estimation of the power spectral density,
                given the length of a trace calculates the sibdivision factor 
                or the segment length needed for the welch method to have such
                resolution.
            mode: 'subdivision_factor' or 'nperseg', optional
                if subdivision_factor calculates in how many segments should
                the timetrace be split
                if nperseg calculates the number of samples per segment.
                Defaults to 'subdivision_factor'
        -------------
        '''
        
        fs = self.fs
        T = self.time[-1]-self.time[0]
        if mode == 'subdivision_factor':
            return frequency_resolution//(1/T)
        elif mode == 'nperseg':
            L = len(self.time)
            segments = frequency_resolution//(1/T)
            return L//segments
        else:
            raise ValueError('Available modes are \'subdivision_factor\' and \'nperseg\'')

    def PSD(self,
            quantity = None,
            fs = None,
            **kwargs):
        """
        Wrapper to Erik Hebestreit's spectrum.derive_psd method
        ------------
        Parameters:
            quantity: string
                timetrace whose power spectral density needs to be estimated
            fs: float, optional
                sampling frequency. If None, extracted with self.fs()
            kwargs: key arguments as in Erik's spectrum.derive_psd
        ------------
        Returns:
            psd: numpy.ndarray
                the power spectral density of quantity
            frequency: numpy.ndarray
                frequency vector corresponding to psd
        ------------
        """
        if fs is None:
            fs = self.fs
        trace = self.get_attribute(quantity) if quantity is not None else self.timetraces[0]
        return spectrum.derive_psd(trace,
                                   fs,
                                   **kwargs)

    def cross_PSD(self,
                  quantities = ['C1','C2'],
                  fs = None, 
                  subdivision_factor = 1,
                  **kwargs):
        """
        Wrapper to Erik Hebestreit's spectrum.derive_cross_spectrum
        -----------
        Parameters:
            quantities: list of strings
                the x,y from spectrum.derive_cross_spectrum
            fs: float, optional
                sampling frequency. If None, extracted with self.fs()
            subdivision_factor: integer, optional
                number of intervals that are averaged for the psd estimation
            kwargs: key arguments as in Erik's spectrum.derive_cross_spectrum
        -----------
        Returns:
            cross_spectrum: numpy.ndarray
                cross spectrum between quantities[0] and quantities[1]
            frequency: numpy.ndarray
                frequency vector corresponding to cross_spectrum
        -----------
        """

        if fs is None:
            fs = self.fs()
        X = self.get_attribute(quantities[0])
        Y = self.get_attribute(quantities[1])
        return spectrum.derive_cross_psd(X, Y, fs, **kwargs)

    def calibration(self,
                    quantity = 'C3',
                    fs = None,
                    subdivision_factor = 1,
                    method = 'welch',
                    window = None,
                    particle_size = None,
                    **kwargs):
        """
        Extracts calibration factor and the lorentzian fit parameters.
        It relies on Erik Hebestreit's calibration for underdamped regime.
        -----------
        Parameters:
            quantity: string
                mode whose calibration needs to be estimated
            fs: float, optional
                sampling frequency. If None, self.fs() will be used.
            subdivision_factor: integer, optional
                number of intervals that are averaged for the psd estimation.
            method: string
                available methods are 'welch', with windowing to avoid
                spectral leakage, and 'fft'. See spectrum for more documentation
            window: list of two integers, optional
                interval of frequencies for the fit
                    Example: if fit is desider from 30kHz to 50kHz then
                        window = [3e5,5e5]
            kwargs: key arguments for fitting functions from spectrum
        -----------
        Returns:
            calibration: float
                calibration factor expressed in nm/mV
            calibration_variance: float
                error bar of the calibration factor
            params: numpy.ndarray
                1D array of fitting parameters
            cov: numpy.ndarray
                2D array of covariance matrix of fitting paramers
        ----------
        """
        self.calibrations[quantity]['is_calibrated'] = False
        
        if self.pressure is None or self.temperature is None:
            print('Pressure and temperature are needed for the calibration')
        
        if subdivision_factor is None:
            fs = self.fs
        psd, frequency = self.PSD(quantity = quantity,
                                  fs = fs,
                                  subdivision_factor = subdivision_factor,
                                  method = method)
        if window is not None:
            ind0,ind1 = self.ext_indexes(frequency,window[0],window[1])
            psd = psd[ind0:ind1]
            frequency = frequency[ind0:ind1]

        if self.underdamped: # method valid when resonance is visible
            params, cov = spectrum.fit_psd(psd,frequency, **kwargs)
            fit = (params,cov)
            estimations = calibration.calibrate(fit,self.temperature,self.pressure)
            
            self.calibrations[quantity]['c_calibrated'] = estimations[0]
            self.calibrations[quantity]['R_factor'] = estimations[1]
            self.calibrations[quantity]['particle_size'] = estimations[2]
            self.calibrations[quantity]['particle_mass'] = estimations[3]
            self.calibrations[quantity]['damping'] = (2*np.pi * params[1],2*np.pi * np.sqrt(cov[1,1]))
            self.calibrations[quantity]['fits'] = fit
            self.calibrations[quantity]['is_calibrated'] = True

        else:
            params, cov = spectrum.fit_overdamped_psd(psd,frequency,**kwargs)
            fit = (params,cov)
            if particle_size is None:
                particle_size = (1.36e-7/2,1e-8)
            estimations = calibration.calibrate_overdamped(fit,
                                                           self.temperature,
                                                           self.pressure,
                                                           particle_size = particle_size)
                                                           
            self.calibrations[quantity]['c_calibrated'] = estimations[0]
            self.calibrations[quantity]['particle_size'] = particle_size
            self.calibrations[quantity]['particle_mass'] = estimations[1]
            self.calibrations[quantity]['damping'] = (2*np.pi*estimations[2][0],2*np.pi*estimations[2][1])
            self.calibrations[quantity]['fits'] = fit
            self.calibrations[quantity]['is_calibrated'] = True
            
    def electrostatic_force(self,
                            modulation_frequency,
                            modulation_amplitude,
                            quantity = 'C3',
                            fs = None,
                            subdivision_factor = 1,
                            method = 'welch',
                            starting_frequency = None,
                            nfft_factor = 1,
                            **kwargs):
        """The method assumes a constant electrostatic modulation.
        It retrieves the electrostatic force that acts on the particle.
        Formulas used are from Francesco Ricci's absolute mass measurement paper.
        -----------------
        Parameters:
            modulation_frequency: float
                frequency of the electrostatic modulation
            modulation_amplitude: float
                amplitude of the electrostatic modulation [V]
            quantity: string, optional
                quantity on which the modulation signal is applied.
                Defaults to 'C3'
            fs: float, optional
                sampling frequency. If None, self.fs will be used.
            subdivision_factor: integer, optional
                Number of intervals used for the psd estimation.
                Defaults to 1.
            method: string, optional
                method used for the psd estimation. Available options are
                'welch' and 'fft', default is 'welch'
            starting_frequency: float, optional
                lowest frequency to consider in the lorentzian fit
            nfft_factor: float
                factor of zero_pads, equivalent to interpolation of spectrum
                for more frequencies
            kwargs: keyword arguments for spectrum.derive_psd
        -------------------
        Returns:
            float: electrostatic force per modulation amplitude
                Example: if with 10V an electrostatic force of amplitude 1pN is observed,
                then the method returns 1pN/10V = 1e-13
        -------------------
        """
        
        if not self.calibrations[quantity]['is_calibrated']:
            print('Error: the particle has not been calibrated')
        
        if fs is None:
            fs = self.fs
            
        psd, frequency = self.PSD(quantity = quantity,
                                  fs = fs,
                                  subdivision_factor = subdivision_factor,
                                  method = method,
                                  nfft_factor = nfft_factor)
                                  
        tau = 1/fs * len(frequency)/nfft_factor
                                  
        ind0,ind1 = self.ext_indexes(frequency,starting_frequency, modulation_frequency)
        psd_th = psd[ind0:ind1]
        frequency_th = frequency[ind0:ind1]
        
        neutral_fit, neutral_cov = spectrum.fit_psd(psd_th,
                                                    frequency_th, 
                                                    **kwargs)
        f_resonance = neutral_fit[2]
        damping = 2*np.pi*neutral_fit[1]
        
        fitted_psd = spectrum.lorentzian(frequency,*neutral_fit)
        detrended_psd = psd - fitted_psd
        resonance_index = np.where(frequencies >= modulation_frequency)[0][0]
        
        electrostatic_force = np.sqrt(detrended_psd[resonance_index]*4*damping**2/tau)
        electrostatic_force /= self.calibrations[quantity]['c_calibrated']
        
        return electrostatic_force/modulation_amplitude
        
            
"""
TODO:
    -debug the charge method
    -implement plotting methods
    -implement a renormalization method after the calibration
    
    -implement a histogram method
    -implement a filtering method
"""