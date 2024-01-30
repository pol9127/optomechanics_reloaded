'''
@ Author: Andrei Militaru
@ date: 27th of May 2019
'''

import math
import numpy as np
from ..post_processing import spectrum, calibration, measurement
from warnings import warn
from scipy.constants import Boltzmann as kB
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from ..theory.stochastics.function import diffusion, function, get_D_nb
import matplotlib.pyplot as plt
from ..visualization.set_axes import set_ax
import threading
import subprocess
from ..post_processing import cpp_extension as cpp
import os
import time
import pandas as pd
from numba import njit


class cpp_thread (threading.Thread):

    def __init__(self,
                 executable, 
                 str_input_file, 
                 str_output_file, 
                 number_of_elements, 
                 tolerance,
                 patience,
                 *coefficients):
        '''
        Thread class that runs the C++ script for nonlinear inversion.
        The performance of this method is limited only by python's speed in reading
        the file with the results.
        -----------------------------
        Parameters:
            executable: str
                the C++ executable that needs to be run. It is machine-dependent, so it should 
                be compile from the script optomechanics.post_processing.cpp_extention.nonlinearity_inversion.
            input_file: str
                file from which to read the vector to be inverted.
            output_file: str
                file where to save to corrected vector.
            number_of_elements: int
                number of elements that need to be inverted from input_file.
            tolerance: float
                precision of the Newton-Raphson inversion algorithm.
            patience: int
                maximum number of iterations
            coefficients: array of floats
                coefficients of nonlinear terms.
        -----------------------------
        '''
        
        threading.Thread.__init__(self)
        self.tolerance = tolerance
        self.patience = patience
        self.executable = executable
        self.input_file = str_input_file
        self.output_file = str_output_file
        self.number_of_elements = number_of_elements
        self.coefficients = coefficients
        
    def run(self):
        executable = '.' + self.executable
        cpp_process = executable + ' ' + self.input_file + ' '
        cpp_process += self.output_file + ' ' + str(self.number_of_elements) + ' '
        cpp_process += str(self.tolerance) + ' ' + str(self.patience)
        for coefficient in self.coefficients:
            cpp_process += ' ' + str(coefficient)
        subprocess.run(cpp_process, shell = True)


class LinearCalibrator():

    def __init__(self):
        
        self.underdamped = True
        self.parameters = {'is_calibrated'  : False,
                           'c_calibrated'   : None,      # bits/m
                           'R_factor'       : None,
                           'particle_size'  : None,      # radius
                           'particle_mass'  : None,      # kg
                           'damping'        : None,      # measured in [rad s^-1]
                           'fits'           : None,      # [amplitude, damping, frequency,offset]
                           'pressure'       : None,      # mbar
                           'temperature'    : None}      # K
    
    def calibrate(self,
                  trace, 
                  fs,
                  underdamped=True,
                  pressure=(980, 5),
                  temperature=(297.5, 1),
                  freq_resolution=100,
                  cut_to_power2=True,
                  window=None,
                  particle_size=None,
                  nm=False,
                  **kwargs):
        """
        Extracts calibration factor and the lorentzian fit parameters.
        It relies on Erik Hebestreit's calibration script for underdamped regime.
        -----------
        Parameters:
            trace: numpy.ndarray
                trace used to estimate the calibration parameters
            fs: float
                sampling frequency
            underdamped: bool, optional
                whether the timetrace has been taken in the underdamped regime or not.
                Defaults to True
            pressure: tuple of floats, optional
                pressure in mbar at which the timetrace has been recorded.
                pressure[0] corresponds to the estimated value, error bar is pressure[1].
                If not given, the error bar is assumed to be 0.
            temperature: tuple of floats, optional
                temperature at which the timetrace has been recorded.
                temperature[0] is the estimated value, temperatur[1] is the error bar.
            method: string
                available methods are 'welch', with windowing to avoid
                spectral leakage, and 'fft'. See spectrum for more documentation
            window: list of two integers, optional
                interval of frequencies for the fit
                    Example: if fit is desider from 30kHz to 50kHz then
                        window = [3e5,5e5]
            freq_resolution: float, optional
                when estimating the power spectral density, frequency resolution
                required in Hz. Defaults to 100
            cut_to_power2: bool, optional
                The required frequency resolution fixes the number of points per segment
                in the Welch estimation of the power spectral density. It is computationally
                advantageous, however, to have the number of points given by a power of two.
                It cut_to_power2 is True, then the segment size is adjusted accordingly.
                Defaults to False.
            particle_size: float, optional
                Radius of the particle in meters. If None, kinetic gas theory is used instead.
                Used to estimate the damping coefficient given the pressure. 
                Defaults to 68 nm.
            nm: bool, optional
                whether the returned trace should be measured in nanometers or not.
                Defaults to True
            kwargs: key arguments for fitting functions from spectrum
        -----------
        Returns:
            calibrated_trace: numpy.ndarray
                trace measured in nanomaters (meters) depending whether nm is True (False).
        ----------
        """          
        self.parameters['is_calibrated'] = False
        
        if type(temperature) is not tuple:
            temperature = (temperature, 0)
        if type(pressure) is not tuple:
            pressure = (pressure, 0)
        
        self.parameters['pressure'] = pressure
        self.parameters['temperature'] = temperature
        
        N = len(trace)
        T = N/fs
        subdivision_factor = int(freq_resolution/(1/T))
        
        psd, frequency = spectrum.derive_psd(trace,
                                             fs,
                                             subdivision_factor = subdivision_factor,
                                             cut_to_power2 = cut_to_power2)
                                             
        if window is not None:
            ind0,ind1 = measurement.timetrace.ext_indexes(frequency,window[0],window[1])
            psd = psd[ind0:ind1]
            frequency = frequency[ind0:ind1]
        
        self.underdamped = underdamped
        
        if underdamped: # method valid when resonance is visible
            params, cov = spectrum.fit_psd(psd,frequency, **kwargs)
            fit = (params,cov)
            estimations = calibration.calibrate(fit,
                                                temperature,
                                                pressure,
                                                particle_size=particle_size)
            
            self.parameters['c_calibrated'] = estimations[0]
            self.parameters['R_factor'] = estimations[1]
            self.parameters['particle_size'] = estimations[2]
            self.parameters['particle_mass'] = estimations[3]
            self.parameters['damping'] = (2*np.pi * params[1],2*np.pi * np.sqrt(cov[1,1]))
            self.parameters['fits'] = fit
            self.parameters['is_calibrated'] = True
        
        else:
            params, cov = spectrum.fit_overdamped_psd(psd,frequency,**kwargs)
            fit = (params,cov)
            if particle_size is None:
                particle_size = (68e-9,10e-9)
            estimations = calibration.calibrate_overdamped(fit,
                                                           temperature,
                                                           pressure,
                                                           particle_size = particle_size)
                                                           
            self.parameters['c_calibrated'] = estimations[0]
            self.parameters['particle_size'] = particle_size
            self.parameters['particle_mass'] = estimations[1]
            self.parameters['damping'] = (2*np.pi*estimations[2][0],2*np.pi*estimations[2][1])
            self.parameters['fits'] = fit
            self.parameters['is_calibrated'] = True
        
        return self.calibrate_trace(trace, nm)
        
    def return_spectrum(self, frequency):
        '''
        Given a frequency numpy.array, the method returns the 
        theoretical power spectral density as estimated from the calibration.
        -------------------------
        Parameters:
            frequency: numpy.ndarray
                frequency array over which the theoretical power spectral density is constructed.
                The values are interpreted in Hz.
        ------------------------
        Returns:
            fit_curve: numpy.ndarray
                theoretical power spectral density as from calibration parameters.
        ------------------------
        '''
        
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet.')
        
        if self.underdamped:
            fit_curve = spectrum.lorentzian(frequency, *self.parameters['fits'][0])
        else: 
            fit_curve = spectrum.damped_lorentzian(frequency, *self.parameters['fits'][0])
            
        return fit_curve
        
    def calibrate_trace(self, trace, nm=False):
        '''
        Given a trace in bits (or Volts) the method returns the trace
        in meters.
        ------------------------
        Parameters:
            trace: numpy.ndarray 
                the timetrace to be calibrated. It needs to be a 
                numpy array such that division by scalar is possible.
            nm: bool, optional
                whether to return the trace in nanometers or in meters.
                Defaults to True.
        -----------------------
        Returns:
            transformed trace
        -----------------------
        '''
        
        if not self.parameters['is_calibrated']:
            raise Exception('Calibrator not initialized yet.')
            
        c = self.parameters['c_calibrated'][0]
        position = trace/c
        return position/1e-9 if nm else position
    
    def c(self):
        '''
        Returns the calibration factor measured in bits/m.
        '''
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet.')
            
        return self.parameters['c_calibrated'][0]
        
    def fc(self):
        '''
        Returns the characteristic frequency [Hz] for the overdamped regime.
        '''
        
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet')
        elif self.underdamped == True:
            str1 = 'Calibration performed in the underdamped regime, there is no characteristic frequency fc. '
            str2 = 'Maybe you meant the resonance frequency (f0)?'
            raise Exception(str1 + str2)
            
        return self.parameters['fits'][0][1]

    def f0(self):
        """
        Returns the resonance frequency [Hz] for the underdamped regime.
        """

        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet.')

        return self.parameters['fits'][0][2]

    def m(self):
        '''
        Returns the mass of the particle [kg] as obtained from calibration.
        '''
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet')
        
        return self.parameters['particle_mass'][0]
        
    def gamma(self):
        '''
        Returns the damping coefficient of the particle [rad/s] as obtained from calibration.
        '''
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet')
            
        return self.parameters['damping'][0]
        
    def sigma(self):
        '''
        Returns an estimation of the thermal noise strength: sqrt(2kT/mGamma)
        '''
        
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet')
            
        m = self.m()
        gamma = self.gamma()
        kT = self.parameters['temperature'][0]*kB
        
        return np.sqrt(2*kT/(m*gamma))
        
    def print_parameters(self):
        '''
        Prints the relevant estimated parameters.
        '''
        if self.parameters['is_calibrated'] == False:
            raise Exception('Calibrator not initialized yet.')
        
        if self.underdamped:
            c = self.parameters['c_calibrated']
            radius = self.parameters['particle_size'] 
            mass = self.parameters['particle_mass']
            damping = self.parameters['damping']
            resonance = (self.parameters['fits'][0][2],
                         self.parameters['fits'][1][2,2])
            offset = (self.parameters['fits'][0][3],
                      self.parameters['fits'][1][3,3])
                      
            offset_pm = (offset[0]/(c[0]*1e-12)**2, offset[1]/(c[0]*1e-12)**2)
                      
            print('\n------------------------------------\n')
            print('Calibration parameter: {:.2f} ({:.2f}) bits/nm'.format(
                                                        c[0]*1e-9, c[1]*1e-9))
            print('Particle radius: {:.2f} ({:.2f}) nm'.format(
                                                        radius[0]/1e-9, radius[1]/1e-9))
            print('Particle mass: {:.2e} ({:.2e}) kg'.format(
                                                        mass[0], mass[1]))
            print('Damping: 2pi {:.2f} ({:.2f}) krad/s'.format(
                                              damping[0]/(2*np.pi*1e3), damping[1]/(2*np.pi*1e3)))
            print('Angular resonance frequency:  2pi {:.2f} ({:.2f}) krad/s'.format(
                                              resonance[0]/1e3, resonance[1]/1e3))
            print('Estimated measurement error:  {:.2f} ({:.2f}) pm^2/Hz'.format(
                                                        offset_pm[0], offset_pm[1]))
            print('\n------------------------------------\n')
            
        else:
            c = self.parameters['c_calibrated']
            radius = self.parameters['particle_size'] 
            mass = self.parameters['particle_mass']
            damping = self.parameters['damping']
            cutoff = (self.parameters['fits'][0][1],
                         np.sqrt(self.parameters['fits'][1][1,1]))
            offset = (self.parameters['fits'][0][2],
                      self.parameters['fits'][1][2,2])
                      
            offset_pm = (offset[0]/(c[0]*1e-12)**2, np.sqrt(offset[1])/(c[0]*1e-12)**2)
            
            print('\n------------------------------------\n')
            print('Calibration parameter: {:.2f} ({:.2f}) bits/nm'.format(
                                                        c[0]*1e-9, c[1]*1e-9))
            print('Particle radius: {:.2f} ({:.2f}) nm'.format(
                                                        radius[0]/1e-9, radius[1]/1e-9))
            print('Particle mass: {:.2e} ({:.2e}) kg'.format(
                                                        mass[0], mass[1]))
            print('Damping: 2pi {:.2f} ({:.2f}) krad/s'.format(
                                                damping[0]/(2*np.pi*1e3), damping[1]/(2*np.pi*1e3)))
            print('Characteristic frequency:  2pi {:.2f} ({:.2f}) krad/s'.format(
                                                cutoff[0]/1e3, cutoff[1]/1e3))
            print('Estimated measurement error:  {:.2f} ({:.2f}) pm$^2$/Hz'.format(
                                                        offset_pm[0], offset_pm[1]))
            print('\n------------------------------------\n')
            
        return 0


class NonlinearCalibrator():

    def __init__(self):
        
        self.calibrated = False
        self.calibrators = []
        self.regr_drift = []
        self.regr_sigma = []
        self.processes = []
        self.degree = 0
        self.iterations = 0
        self.artificial_calibration = 1
        self.tolerance = None
        self.method = None

    @property
    def fc(self):
        if self.calibrated:
            return self.calibrators[-1].fc()
        else:
            raise Exception('Calibrator not yet calibrated.')

    def nonlinearity(self, y, *args):
        '''
        This method returns an optomechanics.stochastics.function instance 
        corresponding to y - x - sum([args[i]*x**i+2 for i in range(len(args))]).
        -----------------------
        Parameters:
            y: float
                value detected for the particle's position
            args: array of floats
                coefficients of higher order distorsions in the detection
        ----------------------
        Returns:
            instance of class optomechanics.stochastics.function
        ----------------------
        '''

        self.degree = len(args)
        x = function.x()
        output = y - x
        for i in range(self.degree):
            output -= args[i]*x**(i+2)
        
        return output
    
    @staticmethod
    def polynomial_feature_transformation(pos, degree, intercept=True):
        '''
        Given a vector interpreted as position, creates a matrix made of 
        different powers of the position. The output is ready to be used as
        input for sklearn.linear_model.LinearRegression class instances.
        ------------------
        Parameters:
            pos: numpy.ndarray
                position vector
            degree: int
                maximum degree of the feature transformation
            intercept: bool, optional
                whether to include a last column made of ones, in order to allow
                for affine transformations of the features.
        -------------------
        Returns:
            numpy.ndarray of shape (len(pos), degree) if not intercept
            numpy.ndarray of shape (len(pos), degree + 1) if intercept
        -------------------
        '''

        if intercept:
            phi = np.zeros((len(pos),degree + 1))
        else: 
            phi = np.zeros((len(pos),degree))
        for idx in range(degree):
            phi[:,idx] = pos**(idx+1)
        if intercept:
            phi[:,-1] = np.ones(pos.shape[0])

        return phi
    
    @staticmethod
    def plot_nonlinearity(*coefficients, plot_range=(-500e-9, 500e-9)):
        '''
        Given the estimated nonlinear coefficients, a plot of the true position vs the
        detected position is shown.
        ----------------------
        Parameters:
            coefficients: array
                array of coefficients of higher order, i.e. starting from the square term.
            plot_range: tuple, optional,
                range of positions of interest, (xmin, xmax), measured in meters.
                Defaults to (-500e-9, 500e-9).
        ---------------------
        Returns:
            Instance of class optomechanics.stochastics.function.function corresponding 
            to the nonlinearity.
        --------------------
        '''

        print('Coefficients are:',coefficients)
        x = function.x()
        output = function.x()
        for i in range(len(coefficients)):
            coefficient = coefficients[i]
            power = i + 2
            output += coefficient * x**power
        position = np.linspace(plot_range[0], plot_range[1], 1000)
        detected = np.array([output.eval(position[i]) for i in range(len(position))])

        xlabel = 'True position [nm]'
        ylabel = 'Detected [nm]'
        label1 = 'Linear'
        label2 = 'Nonlinear'
        font = 18
        fignon = plt.figure(figsize = (8,6))
        axnon = fignon.add_subplot(211)

        axnon.plot(position/1e-9,position/1e-9, linewidth = 2, label = label1)
        axnon.plot(position/1e-9,detected/1e-9, linewidth = 2, label = label2)

        set_ax(axnon, xlabel = xlabel, ylabel = ylabel, fs = font, legend = True)
        plt.tight_layout()
        plt.show()
        
        return output

    def calibrate(self,
                  trace,
                  fs,
                  grid=np.linspace(-100e-9,100e-9,250),
                  degree=3,
                  iterations=2,
                  tolerance=1e-12,
                  artificial_calibration=1e10,
                  inversion_method='numba',
                  nm=False,
                  method='newton',
                  monitor_estimations=False,
                  plot_nonlinearity=False,
                  patience=100,
                  number_of_threads=15,
                  file_path=None,
                  executable=None,
                  verbose=False,
                  **kwargs):
        """
        Nonlinear calibration of an overdamped trace.
        The method consists of an iterative scheme that alternates linear calibration
        (as described in the PhD Thesis of Erik Hebestreit) and polynomial fit of the drift
        and of the diffusion of the particle. The nonlinear calibration method that both
        the equipartition theorem and the fluctuation-dissipation theorem apply on the particle.
        The linear calibration makes use of kinetic theory to find a proportionality factor with which 
        the equipartition theorem is fulfilled. The nonlinear steps of the iteration make use of
        the fact that the thermal noise is white and additive. 
        ----------------------
        Parameters:
            trace: numpy.ndarray
                timetrace of the particle measured with a nonlinear detection.
                For the correct nonlinear calibration, it is useful to have a few tens of seconds long
                trace.
            fs: float
                sampling frequency at which the timetrace has been measured or simulated
            grid: numpy.ndarray, optional
                1D array of the points where the drift and the diffusion need to be calculated.
                Defaults to np.arange(-100e-9,100e-9,250).
            degree: int, optional
                Taylor order to which the detection nonlinearity should be corrected.
                Defaults to 3.
            iterations: int, optional
                number of repetitions of linear and nonlinear iterations.
                Defaults to 2.
            tolerance: float, optional
                used for NonlinearCalibrator.inv_nonlinearity() method.
                Represents the precision required for the numerical algorithm in the inversion
                of the equation y = x + c1*x**2 + c2*x**3.
                Defaults to 1e-12.
            artificial_calibration: float, optional
                after nonlinear calibration, a last linear calibration might be necessary.
                If the values of the trace are too small, however, fitting a lorentzian to the
                power spectral density can fail. For this reason, before the last linear calibration
                the whole timetrace is multiplied by artificial_calibration.
                Defaults to 1e10.
            inversion_method: str, optional
                Method used to invert the nonlinearity. Possible options are:
                - 'numba': just in time compilation with numba.njit() is used. This is the default
                  and suggested method.
                - 'function_class': a distortion function is built with an optomechanics.stochastics.function.function
                  instance. The inversion is inverted through the find_zero method of such class.
                  This method tends to be slow due to the class overhead and the python interpreter.
                - 'cpp_threads': a C++ script is called from the python interpreter. The distorted trace is
                  printed to a file and different C++ threads are called to invert it. Bottleneck of this method
                  is the file printing, otherwise the fastest one.
            nm: bool, optional
                Whether the returned trace should be measured in nanometers (nm = True) or meters (nm = False).
                Defaults to False
            method: str, optional
                Used if inversion_method is function_class. Method use for the numerical inversion of the nonlinearity.
                Options are 'recursive' and 'newton', the latter being the Newton-Raphson algorithm.
                Defaults to 'newton'.
            monitor_estimations: bool, optional
                If True, after estimating the drift and the diffusion, the two quantities are plot.
                Defaults to False.
            plot_nonlinearity: bool, optional
                If True, the 'true position'-'detected position' plot will be shown.
                Defaults to False.
            patience: int, optional
                Number of maximum iterations for numerical inversion before returning
                the current value of the variable. Useful when algorithm freezes.
                Defaults to 100.
            number_of_threads: int, optional
                Used if inversion_method is 'cpp_threads'. The vector is split into number_of_threads subvectors
                and separate threads are called for inversion. The optimal performance is
                when number_of_threads equals the number of CPUs of the computer. Defaults to 15, good for Sisyphous.
            file_path: str, optional
                Used if inversion_method is 'cpp_threads'. The path of the intermediate files used for the inversion.
                Defaults to current directory.
            executable: str, optional
                Used if inversion_method is 'cpp_threads'. C++ executable that performs the nonlinearity inversion.
                Defaults to optomechanics.post_processing.cpp_extension.cpp_inversion
            verbose: bool, optional
                If True, it provides updates about the iterations currently running. 
                Defaults to False
            kwargs: keyword arguments used for LinearCalibrator.calibrate() method.
        -----------------------
        Returns:
            calibrated_trace: 
                trace of the particle measured in meters (nm = False) or in nanometers (nm = True).
                The trace is corrected both from bit-to-volts proportionality factors and from nonlinear
                terms that come from the interferometric position measurement.
        ----------------------
        """
        
        self.__init__()
        if verbose:
            print('Calibrator cleared.')
        self.artificial_calibration = artificial_calibration
        self.iterations = iterations
        self.degree = degree
        self.tolerance = tolerance
        self.method = method

        self.calibrators.append(LinearCalibrator())
        if verbose:
            print('Pre-calibration...')
        xc = self.calibrators[0].calibrate(trace, fs, **kwargs)
        if verbose:
            print('Pre-calibration completed.')
        
        for i in range(iterations):
            if verbose:
                print('\n------------------------')
                print('Iteration ' + str(i+1) + ' running...')
                print('------------------------\n')
            
            fc_calib = self.calibrators[i].fc()
            sigma_calib = self.calibrators[i].sigma()

            def mu(x):
                return -2*np.pi*fc_calib*x

            def sigma(x):
                return sigma_calib

            process = diffusion(mu,sigma)

            if verbose:
                print('Estimating drift and diffusion...')
            estimations = get_D_nb(xc, grid, fs)
            (used_positions, drifts, diffs) = estimations
            if verbose:
                print('Drift and diffusion estimated.')

            if monitor_estimations:
                process.plot_nb_D_estimations(estimations, superpose_theory=True)

            phi = self.polynomial_feature_transformation(np.array(used_positions), degree-1)

            self.regr_drift.append(LinearRegression(fit_intercept=False))
            self.regr_sigma.append(LinearRegression(fit_intercept=False))

            self.regr_drift[i].fit(phi, drifts)
            self.regr_sigma[i].fit(phi, np.sqrt(2*diffs))

            weights = self.regr_sigma[i].coef_
            nonlinearity_coefficients = np.zeros(degree-1)
            for idx in range(degree-1):
                power = idx+2
                nonlinearity_coefficients[idx] = weights[idx]/(power*weights[-1])
            
            if plot_nonlinearity:
                self.plot_nonlinearity(*nonlinearity_coefficients)
            if verbose:
                print('\nInverting the nonlinearity...')

            if inversion_method == 'numba':
                @njit()
                def inv_nonlinearity(y,
                                     int_tolerance,
                                     int_patience,
                                     c,
                                     int_length):

                    def f(x, y, cc, internal_length):
                        out = y - x
                        for int_idx in range(internal_length):
                            out -= cc[int_idx]*x**(int_idx+2)
                        return out

                    def f_der(x, cc, internal_length):
                        out = -1
                        for int_idx in range(internal_length):
                            out -= (int_idx+2)*cc[int_idx]*x**(int_idx+1)
                        return out
                    delta = np.inf
                    iteration = 0
                    z0 = 0
                    while delta > int_tolerance and iteration < int_patience:
                        prev = z0
                        z0 = z0 - f(z0, y, c, int_length)/f_der(z0, c, int_length)
                        delta = np.abs(prev - z0)
                        iteration += 1
                    return z0

                @njit()
                def inv_trace(xcc,
                              int_tolerance,
                              int_patience,
                              int_nonlinearity_coefficients,
                              int_length):

                    for idx in range(len(xc)):
                        xcc[idx] = inv_nonlinearity(xcc[idx],
                                                    int_tolerance,
                                                    int_patience,
                                                    int_nonlinearity_coefficients,
                                                    int_length)
                    return xc
                if verbose:
                    print('Inverting nonlinearity...')
                xc = inv_trace(xc,
                               tolerance,
                               patience,
                               nonlinearity_coefficients,
                               len(nonlinearity_coefficients))

            elif inversion_method == 'function_class':
                nonlinearity_constructor = self.nonlinearity
                for idx in tqdm(range(len(xc))):
                    nonlinear_function = nonlinearity_constructor(xc[idx],*nonlinearity_coefficients)
                    xc[idx] = nonlinear_function.find_zero(tolerance = tolerance)

            elif inversion_method == 'cpp_threads':
                threads = []
                sublength = int(len(xc)/number_of_threads)
                
                print('Passing vector to C++ extension.')
                
                for fold_idx in tqdm(range(number_of_threads)):
                    if fold_idx != (number_of_threads-1):
                        fold = xc[sublength*fold_idx : sublength*(fold_idx+1)]
                    else:
                        fold = xc[sublength*fold_idx :]
                    
                    if file_path is None:
                        file_path = os.getcwd() + '/'
                        
                    if executable is None:
                        path_steps = cpp.__file__.split('/')[:-1]
                        executable = ''
                        for folder in path_steps:
                            executable += folder + '/'
                        executable += 'cpp_inversion'
                    
                    input_file = file_path + 'vector' + str(fold_idx) + '.txt'
                    output_file = file_path + 'out_vector' + str(fold_idx) + '.txt'
                    
                    with open(input_file, 'w') as write_file:
                        for j in range(len(fold)):
                            write_file.write('{:.12f}\n'.format(fold[j]))
                            
                    threads.append(cpp_thread(executable,
                                              input_file,
                                              output_file,
                                              len(fold),
                                              tolerance,
                                              patience,
                                              *nonlinearity_coefficients))
                    threads[fold_idx].start()
                    
                print('Executing C++ extension...')
                print('Reloading vector...')
                
                for fold_idx in tqdm(range(number_of_threads)):
                
                    while threads[fold_idx].is_alive():
                        time.sleep(0.1)
                    
                    input_file = file_path + 'vector' + str(fold_idx) + '.txt'
                    output_file = file_path + 'out_vector' + str(fold_idx) + '.txt'
                    rm_input = 'rm ' + input_file
                    rm_output = 'rm ' + output_file
                    
                    if fold_idx != (number_of_threads-1):
                        xc[sublength*fold_idx : sublength*(fold_idx+1)] = np.array(pd.read_csv(output_file, header = None)[0])
                    else:
                        xc[sublength*fold_idx :] = np.array(pd.read_csv(output_file, header = None)[0])
                    
                    subprocess.run(rm_input, shell = True)
                    subprocess.run(rm_output, shell = True)

            else:
                raise Exception('Inversion method chosen is invalid. Only options are \'numba\', ' + (
                                 '\'function_class\' and \'cpp_threads\'.'))

            if verbose:        
                print('Nonlinearity inverted.\n')

            self.calibrators.append(LinearCalibrator())
            if verbose:
                print('Calibrating linearly...')
            xc = self.calibrators[-1].calibrate(xc*artificial_calibration, fs, **kwargs)
            if verbose:
                print('Linear calibration completed.')

        self.calibrated = True
                                           
        return xc if not nm else xc/1e-9

    def calibrate_trace(self,
                        y, 
                        patience=100,
                        inversion_method='numba',
                        number_of_threads=15,
                        file_path=None,
                        executable=None,
                        verbose=False):
        '''
        Applies the nonlinear calibration to the input trace y.
        ---------------------
        Parameters:
            y: numpy.ndarray
                trace to be calibrated.
            patience: int, optional
                number of iterations for numerical inversion before returning the current value.
                Useful for cases where the algorithm might be frozen.
                Defaults to 100.
            inversion_method: str, optional
                Method used to invert the nonlinearity. Possible options are:
                - 'numba': just in time compilation with numba.njit() is used. This is the default
                  and suggested method.
                - 'function_class': a distortion function is built with an optomechanics.stochastics.function.function
                  instance. The inversion is inverted through the find_zero method of such class.
                  This method tends to be slow due to the class overhead and the python interpreter.
                - 'cpp_threads': a C++ script is called from the python interpreter. The distorted trace is
                  printed to a file and different C++ threads are called to invert it. Bottleneck of this method
                  is the file printing, otherwise the fastest one.
            number_of_threads: int, optional
                Used if inversion_method is 'cpp_threads'. The vector to be inverted is split
                into number_of_threads subvectors and separate threads are called for inversion.
                The optimal performance is when number_of_threads equals the number of CPUs of the computer.
                Defaults to 15, good for Sisyphous.
            file_path: str, optional
                Used if inversion_method is 'cpp_threads'. The path of the intermediate files used for the inversion.
                Defaults to current directory.
            executable: str, optional
                Used if inversion_method is 'cpp_threads'. C++ executable that performs the nonlinearity inversion.
                Defaults to optomechanics.post_processing.cpp_extension.cpp_inversion
            verbose: bool, optional
                If True, it provides updates about the iterations currently running. 
                Defaults to False
        --------------------
        Returns:
            numpy.ndarray: calibrated trace in meters.
        ---------------------
        '''
        
        if not self.calibrated:
            raise Exception('Nonlinear calibrator not initialized yet.')

        xc = self.calibrators[0].calibrate_trace(y)

        for i in range(self.iterations):
            if verbose:
                print('\n------------------------')
                print('Iteration ' + str(i+1) + ' running...')
                print('------------------------\n')

            weights = self.regr_sigma[i].coef_
            nonlinearity_coefficients = np.zeros(self.degree-1)
            for idx in range(self.degree-1):
                power = idx+2
                nonlinearity_coefficients[idx] = weights[idx]/(power*weights[-1])
            
            if verbose:
                print('\nInverting the nonlinearity...')

            if inversion_method == 'numba':
                @njit()
                def inv_nonlinearity(y,
                                     int_tolerance,
                                     int_patience,
                                     c,
                                     int_length):

                    def f(x, y, cc, internal_length):
                        out = y - x
                        for int_idx in range(internal_length):
                            out -= cc[int_idx] * x ** (int_idx + 2)
                        return out

                    def f_der(x, cc, internal_length):
                        out = -1
                        for int_idx in range(internal_length):
                            out -= (int_idx + 2) * cc[int_idx] * x ** (int_idx + 1)
                        return out

                    delta = np.inf
                    iteration = 0
                    z0 = 0
                    while delta > int_tolerance and iteration < int_patience:
                        prev = z0
                        z0 = z0 - f(z0, y, c, int_length) / f_der(z0, c, int_length)
                        delta = np.abs(prev - z0)
                        iteration += 1
                    return z0

                @njit()
                def inv_trace(xcc,
                              int_tolerance,
                              int_patience,
                              int_nonlinearity_coefficients,
                              int_length):

                    for idx in range(len(xc)):
                        xcc[idx] = inv_nonlinearity(xcc[idx],
                                                    int_tolerance,
                                                    int_patience,
                                                    int_nonlinearity_coefficients,
                                                    int_length)
                    return xc

                if verbose:
                    print('Inverting nonlinearity...')
                xc = inv_trace(xc,
                               self.tolerance,
                               patience,
                               nonlinearity_coefficients,
                               len(nonlinearity_coefficients))
            elif inversion_method == 'function_class':
                nonlinearity_constructor = self.nonlinearity
                for idx in tqdm(range(len(xc))):
                    nonlinear_function = nonlinearity_constructor(xc[idx],*nonlinearity_coefficients)
                    xc[idx] = nonlinear_function.find_zero(tolerance = self.tolerance)

            elif inversion_method == 'cpp_threads':
                threads = []
                sublength = int(len(xc)/number_of_threads)
                
                print('Passing vector to C++ extension.')
                
                for fold_idx in tqdm(range(number_of_threads)):
                    if fold_idx != (number_of_threads-1):
                        fold = xc[sublength*fold_idx : sublength*(fold_idx+1)]
                    else:
                        fold = xc[sublength*fold_idx :]
                    
                    if file_path is None:
                        file_path = os.getcwd() + '/'
                        
                    if executable is None:
                        path_steps = cpp.__file__.split('/')[:-1]
                        executable = ''
                        for folder in path_steps:
                            executable += folder + '/'
                        executable += 'cpp_inversion'
                    
                    input_file = file_path + 'vector' + str(fold_idx) + '.txt'
                    output_file = file_path + 'out_vector' + str(fold_idx) + '.txt'
                    
                    with open(input_file, 'w') as write_file:
                        for j in range(len(fold)):
                            write_file.write('{:.12f}\n'.format(fold[j]))
                            
                    threads.append(cpp_thread(executable,
                                              input_file,
                                              output_file,
                                              len(fold),
                                              self.tolerance,
                                              patience,
                                              *nonlinearity_coefficients))
                    threads[fold_idx].start()
                
                if verbose:
                    print('Executing C++ extension...')
                    print('Reloading vector...')
                
                for fold_idx in tqdm(range(number_of_threads)):
                
                    while threads[fold_idx].is_alive():
                        time.sleep(0.1)
                    
                    input_file = file_path + 'vector' + str(fold_idx) + '.txt'
                    output_file = file_path + 'out_vector' + str(fold_idx) + '.txt'
                    rm_input = 'rm ' + input_file
                    rm_output = 'rm ' + output_file
                    
                    if fold_idx != (number_of_threads-1):
                        xc[sublength*fold_idx : sublength*(fold_idx+1)] = np.array(pd.read_csv(output_file, header = None)[0])
                    else:
                        xc[sublength*fold_idx :] = np.array(pd.read_csv(output_file, header = None)[0])
                    
                    subprocess.run(rm_input, shell = True)
                    subprocess.run(rm_output, shell = True)

            else:
                raise Exception('Inversion method chosen is invalid. Only options are \'numba\', ' + (
                                 '\'function_class\' and \'cpp_threads\'.'))

            if verbose:
                print('Nonlinearity inverted.\n')

            xc = self.calibrators[i+1].calibrate_trace(xc*self.artificial_calibration)

            if verbose:
                print('Linear calibration completed.')

        return xc

