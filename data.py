from __future__ import division, print_function, unicode_literals

from time import time

import numpy
from .processing import derive_psd, fit_psd, calibrate

import h5py


class MeasurementSet(object):
    """Class representing a set of measurements stored in an HDF5 file.

    The structure of the HDF5 files is as follows:
        * one file contains data from one experiment
        * data that belongs together (e.g. acquired using the same parameter
            set) is grouped in a Measurement with a measurement ID, that is
            represented by an HDF5 Group with the name /measurement_#id#
        * data within one Measurement that is acquired at the same time
            (identical time stamp) has the same data ID
        * IDs in file names should be a four digit integer with leading zeros
            but don't need to be if there are more IDs necessary
        * data spanning a longer time (e.g. pressure, temperature recordings)
            are stored outside a Measurement (in file root) and without an ID
        * the data is stored in an HDF5 Dataset with name 'name[_#id#]'
        * metadata is stored in the attributes of the datasets or groups

    Data can for example be stored in:
        * /pressure
        * /temperature
        * /measurement_0000/timetrace_0000
        * /measurement_0321/timetrace_0002

    Standard data that can be stored and its metadata:
        * timetrace
        * psd
        * pressure
        * temperature
    """

    def __init__(self, file=None, filename=None):
        # type: (h5py.File, str)
        """Initialize new MeasurementSet from file.

        :param file: File object of the HDF5 file.
        :param filename: File name of the HDF5 file.
        :return: New MeasurementSet.
        """
        self.measurement = dict()
        self.additional_data = dict()

        if file is None:
            file = h5py.File(filename, mode='a')

        self.file = file

        for name in file.keys():
            data = file[name]

            if name.lower().startswith('measurement') and \
                    name.split('_')[-1].isdecimal():
                self.measurement[int(name.split('_')[-1])] = Measurement(data)
            elif name.lower() is 'pressure':
                self.pressure = TimeSeries(data, name='pressure')
            elif name.lower() is 'temperature':
                self.temperature = TimeSeries(data, name='temperature')
            elif name.startswith('/'):
                if 'time_step' in data.attrs.keys() or \
                        'sampling_rate' in data.attrs.keys() or \
                        'time_vector' in data.attrs.keys():
                    self.additional_data[name] = TimeSeries(data, name=name)
                else:
                    self.additional_data[name] = data
            else:
                pass

        self.attributes = file.attrs

    def plot_pressure(self):
        """Plot the pressure data in the measurement set."""
        # TODO
        pass

    def plot_temperature(self):
        """Plot the temperature data in the measurement set."""
        # TODO
        pass

    def populate_pressure_attributes(self):
        """Write pressure attribute to all measurements based on data in the
        pressure dataset."""
        # TODO: this should write the pressure attribute to all the
        # measurements and time_traces, based on the time_stamp and the
        # pressure data
        pass

    def populate_temperature_attributes(self):
        """Write temperature attribute to all measurements based on data in the
        temperature dataset."""
        # TODO: this should write the temperature attribute to all the
        # measurements and time_traces, based on the time_stamp and the
        # temperature data
        pass


class Measurement(object):
    """Class representing a measurement that is stored in an HDF5 file. A
    measurement contains data where the environmental parameters (e.g.
    pressure, or temperature) remain constant."""

    def __init__(self, measurement):
        # type: (h5py.Group)
        """Initialize a new Measurement object from a group of an HDF5 file.

        :param measurement: Group of an HDF5 file.
        :return: Measurement object containing time traces, PSDs, additional
            time series data, and metadata as attributes.
        """
        self.time_trace = dict()
        self.psd = dict()
        self.additional_data = dict()
        self.measurement = measurement
        self.attributes = measurement.attrs

        for name in measurement.keys():
            data = measurement[name]

            if name.lower().startswith('timetrace') or \
                    name.lower().startswith('tt') and \
                    name.split('_')[1].isdecimal():
                self.time_trace[int(name.split('_')[1])] = TimeTrace(data)
            elif name.lower().startswith('psd') and \
                    name.split('_')[1].isdecimal():
                self.psd[int(name.split('_')[1])] = PowerSpectralDensity(data)
            else:
                self.additional_data[name] = data

    @property
    def time_stamp(self):
        try:
            return self.attributes['time_stamp']
        except KeyError:
            return None

    @time_stamp.setter
    def time_stamp(self, value):
        self.attributes['time_stamp'] = value


class Data(object):
    """Base class for data that is stored in an HDF5 dataset."""

    def __init__(self, data, unit=None, time_stamp=None, name=None,
                 data_id=None, measurement_id=None):
        # type: (h5py.Dataset or numpy.ndarray or list or tuple, str, float,
        #        str, int, int)
        """Initialize the Data object either by giving an HDF5 Dataset, or by
        specifying data and metadata of the dataset directly.

        Name, data id, and measurement id specify the storage location of the
        Dataset in the HDF5 file:
            * all three parameters set: /measurement_#id#/name_#id#
            * no data id: /measurement_#id#/name
            * no data id and no measurement if: /name

        :param data: HDF5 dataset or 1D/2D ndarray containing the data. In case
            of a 2D ndarray, second dimension contains columns (e.g. axes)
        :param unit: Unit of the data. Can also be specified through attributes
            of HDF5 Dataset.
        :param time_stamp: Time stamp as seconds since epoche (1.1.1970)
        :param name: Short name of the type of data to be used as a name for
            the HDF5 Dataset.
        :param data_id: Integer ID for the data. Has to be unique within one
            measurement.
        :param measurement_id: Integer ID for the measurement. Has to be unique
            within one measurement set/HDF5 file.
        """

        if isinstance(data, (numpy.ndarray, list, tuple)):
            self._data = numpy.array(data)
            self.attributes = dict()
            self.name = name
            self.id = data_id
            self.measurement_id = measurement_id
        elif type(data) is h5py.Dataset:
            self._data = data
            self.attributes = data.attrs

            try:
                self.name = data.name.split('/')[-1].split('_')[0]
            except IndexError:
                self.name = None

            try:
                self.id = int(data.name.split('/')[-1].split('_')[1])
            except IndexError:
                self.id = None

            try:
                self.measurement_id = \
                    int(data.name.split('/')[-2].split('_')[1])
            except IndexError:
                self.measurement_id = None
        elif type(data) is h5py.File:
            path_name = '/'

            if measurement_id is not None:
                path_name += 'measurement_{0:04d}/'.format(measurement_id)
            path_name += name
            if data_id is not None:
                path_name += '_{0:04d}'.format(data_id)

            self._data = data.create_dataset(name, shape=(0, 0),
                                             maxshape=(None, None))
            self.attributes = self._data.attrs
            self.name = name
            self.id = data_id
            self.measurement_id = measurement_id
        else:
            raise TypeError('data of unsupported type {0}'.format(type(data)))

        if time_stamp is not None:
            self.attributes['time_stamp'] = time_stamp

        if unit is not None:
            self.unit = unit

    def __len__(self):
        return self.shape[0]

    @property
    def shape(self):
        return self._data.shape

    @property
    def number_of_columns(self):
        if len(self.shape) == 1:
            return 1
        else:
            return self._data.shape[1]

    @property
    def data(self):
        return self._data[...]

    @data.setter
    def data(self, value):
        self._data[...] = value
        # TODO: Slicing does not work yet for h5py.Dataset. Only writing the
        # entire array.

    def data_column(self, column):
        assert 0 <= column < self.number_of_columns, 'column out of range'
        if self.number_of_columns == 1:
            return self._data
        else:
            return self._data[..., column]

    @property
    def time_stamp(self):
        try:
            return self.attributes['time_stamp']
        except KeyError:
            return None

    @time_stamp.setter
    def time_stamp(self, value):
        self.attributes['time_stamp'] = value

    @property
    def unit(self):
        try:
            return self.attributes['unit']
        except KeyError:
            return None

    @unit.setter
    def unit(self, value):
        assert isinstance(value, str), 'unit has to be a string value'
        self.attributes['unit'] = value

    @property
    def labels(self):
        labels = list([None for i in range(self.number_of_columns)])

        for i in range(self.number_of_columns):
            try:
                labels[i] = self.get_label(i)
            except KeyError:
                pass

        return labels

    @labels.setter
    def labels(self, values):
        for i in range(len(values)):
            self.set_label(values[i], i)

    def get_label(self, number):
        return self.attributes['label_{0}'.format(number)]

    def set_label(self, name, number):
        self.attributes['label_{0}'.format(number)] = name

    def get_column(self, name):
        labels = self.labels
        assert name in labels, 'name unknown'
        return labels.index(name)

    def append_data(self, new_data, axis=0):
        if isinstance(self._data, numpy.ndarray):
            self._data = numpy.append(self._data, numpy.array(new_data),
                                      axis=axis)
        elif type(self._data) is h5py.Dataset:
            original_shape = self.shape

            self._data.resize(self.shape[axis]+new_data.shape[axis], axis)

            if axis == 0:
                #self._data.resize((self.shape[0]+new_data[0], new_data[1]))
                self._data[original_shape[axis]:, ...] = new_data
            elif axis == 1:
                #self._data.resize((new_data[0], self.shape[1]+new_data[1]))
                self._data[:, original_shape[axis]:, ...] = new_data

    def append_data_by_label(self, new_data):
        # type: (dict)
        max_length = 0
        for key in new_data.keys():
            length = len(new_data[key])
            if length > max_length:
                max_length = length

        data_array = numpy.zeros((max_length, self.number_of_columns))

        for i in range(self.number_of_columns):
            name = self.get_label(i)

            if name in new_data.keys():
                data_array[:, i] = new_data[name]

        self.append_data(data_array, axis=0)

    def save_to_hdf(self, loc=None, name=None, overwrite=False):
        if type(loc) is h5py.File or type(loc) is h5py.Group:
            file = loc
        elif isinstance(loc, str):
            file = h5py.File(loc, mode='a')
        else:
            return None

        if name in file.keys():
            dset = file[name]
            if overwrite:
                dset.resize(self.shape)
                dset[...] = self.data
        else:
            dset = file.create_dataset(name=name, data=self.data,
                                       dtype=self.data.dtype,
                                       maxshape=
                                       tuple([None for i in
                                              range(len(self.shape))]))

        for name in self.attributes.keys():
            dset.attrs[name] = self.attributes[name]
        
        if isinstance(loc, str):
            file.close()

    def save_to_csv(self, loc=None):
        pass


class TimeSeries(Data):

    def __init__(self, data, unit=None, time_vector=None, time_step=None,
                 sampling_rate=None, time_stamp=None, name=None, data_id=None,
                 measurement_id=None):

        super(TimeSeries, self).__init__(data, unit=unit,
                                         time_stamp=time_stamp, name=name,
                                         data_id=data_id,
                                         measurement_id=measurement_id)

        # priority order: time_vector, time_step, sampling_rate
        if time_vector is not None:
            self.time = time_vector
        elif 'time_vector' not in self.attributes.keys():
            if time_step is None and sampling_rate is not None:
                self.sampling_rate = sampling_rate
            elif time_step is not None:
                self.time_step = time_step
            elif 'time_step' not in self.attributes.keys():
                raise ValueError("time_vector, time_step or sampling_rate "
                                 "have to be specified in either dataset or "
                                 "parameters")

    @property
    def time(self):
        """
        :return:Time vector of the time trace.
        """

        if 'time_vector' in self.attributes.keys():
            return self.attributes['time_vector']
        else:
            if 'time_step' in self.attributes.keys():
                time_step = self.attributes['time_step']
            elif 'sampling_rate' in self.attributes.keys():
                time_step = 1/self.attributes['sampling_rate']
            else:
                return None
        return numpy.linspace(0, (len(self)-1)*time_step, len(self))

    @time.setter
    def time(self, time_vector):
        assert len(time_vector) is len(self), 'time_vector has to be of ' \
                                              'length {0}'.format(len(self))
        self.attributes['time_vector'] = time_vector

    @property
    def absolute_time(self):
        """
        :return:Time vector of the time trace.
        """

        if self.time_stamp is not None:
            return self.time + self.time_stamp
        else:
            return self.time

    @property
    def relative_time(self):
        """
        :return:Time vector of the time trace.
        """

        time_vector = self.time
        return time_vector - time_vector[0]

    @property
    def time_step(self):
        if 'time_step' in self.attributes.keys():
            return self.attributes['time_step']
        elif 'sampling_rate' in self.attributes.keys():
            return 1/self.attributes['sampling_rate']
        else:
            return None

    @time_step.setter
    def time_step(self, value):
        assert value > 0, 'time_step has to be a positive numeric value'
        self.attributes['time_step'] = value

    @property
    def sampling_rate(self):
        if 'sampling_rate' in self.attributes.keys():
            return self.attributes['sampling_rate']
        elif 'time_step' in self.attributes.keys():
            return 1/self.attributes['time_step']
        else:
            return None

    @sampling_rate.setter
    def sampling_rate(self, value):
        assert value > 0, 'sampling_rate has to be a positive numeric value'
        self.attributes['sampling_rate'] = value

    @staticmethod
    def concatenate(traces):
        time_step = traces[0].time_step
        data = traces[0].data

        for trace in traces[1:]:
            assert trace.time_step == traces[0].time_step
            assert trace.number_of_columns == traces[0].number_of_columns
            data = numpy.concatenate((data, trace.data))

        new_tt = TimeTrace(data, time_step=time_step,
                           time_stamp=traces[0].time_stamp)
        new_tt.attributes = traces[0].attributes
        # FIXME: does this make sense?
        # TODO: also make it work for time_vector

        return new_tt

    def divide(self, parts=None, num_elements=None):
        """

        :param parts:
        :param num_elements:
        :return:
        """
        if parts is not None:
            if num_elements is not None:
                dist = int((len(self)-num_elements)/(parts-1))
            else:
                num_elements = int(len(self) / parts)
                dist = num_elements
        elif num_elements is not None:
            parts = int(len(self)/num_elements)
            dist = num_elements
        else:
            raise ValueError('Either parts or elements have to be given.')

        for i in range(parts):
            # TODO: create new TimeTraces for each part with indices
            # [(dist*i):(dist*(i+1), ...]
            pass

    def plot(self):
        """
        Plot the time series data.
        """
        import matplotlib.pyplot as plt

        plt.figure()
        for i in range(self.number_of_columns):
            try:
                label = self.attributes['label_{0}'.format(i)]
            except KeyError:
                label = None

            plt.plot(self.time, self.data_column(i), label=label)

        plt.xlabel("Time (s)")
        plt.ylabel("{0} ({1})".format(self.name, self.unit))
        plt.legend()


class TimeTrace(TimeSeries):

    def __init__(self, data, **kwargs):

        super(TimeTrace, self).__init__(data, name='timetrace', **kwargs)

    def to_psd(self, method=None, subdivision_factor=None, save=False):
        """Derive the power spectral density from the time trace.

        Parameters
        ----------
        method : str
            Method used for derivation of power spectral density. Either 'fft'
            for using standard FFT, or 'welch' for Welch's windowing method.
        subdivision_factor : int
            Parameter for 'welch'-method, giving the subdivision factor used
            for deriving the size of the window.

        Returns
        -------
        PowerSpectralDensity
            The PSD for the time trace data.

        """

        psd = [None for i in range(self.number_of_columns)]
        frequency_step = None
        for i in range(self.number_of_columns):
            psd[i], frequency_step = \
                derive_psd(self.data_column(i), self.sampling_rate,
                           method=method,
                           subdivision_factor=subdivision_factor)
        psd = numpy.squeeze(numpy.array(psd).transpose())

        if self.unit is '' or None:
            unit = '1/Hz'
        else:
            unit = '{0}^2/Hz'.format(self.unit)

        psd_object = PowerSpectralDensity(psd, unit=unit,
                                          frequency_step=frequency_step,
                                          time_stamp=self.time_stamp)

        for i in range(self.number_of_columns):
            psd_object.attributes['label_{0}'.format(i)] = \
                self.attributes['label_{0}'.format(i)]

        if type(self._data) is h5py.Dataset:
            psd_object.attributes['source_file'] = self._data.file.filename
            psd_object.attributes['source_dset'] = self._data.name

        if save is True and type(self._data) is h5py.Dataset:
            psd_object.save_to_hdf(loc=self._data.parent,
                                   name='psd_{0:04d}'.format(self.id))
        elif type(save) is h5py.Group:
            psd_object.save_to_hdf(loc=save,
                                   name='psd_{0:04d}'.format(self.id))
        elif type(save) is h5py.File:
            psd_object.save_to_hdf(loc=save,
                                   name='/measurement_{0:04d}/psd_{1:04d}'
                                   .format(self.measurement_id, self.id))
        elif isinstance(save, str):
            psd_object.save_to_hdf(loc=save,
                                   name='/measurement_{0:04d}/psd_{1:04d}'
                                   .format(self.measurement_id, self.id))

        return psd_object


class PowerSpectralDensity(Data):

    def __init__(self, data, unit=None, frequency_step=None,
                 time_stamp=time(), data_id=None,
                 measurement_id=None):

        super(PowerSpectralDensity, self).__init__(
                data, unit=unit, time_stamp=time_stamp, name='psd',
                data_id=data_id, measurement_id=measurement_id)

        if frequency_step is not None:
            self.attributes['frequency_step'] = frequency_step
        elif 'frequency_step' not in self.attributes.keys():
            raise ValueError("frequency_step has to be specified in either "
                             "dataset or parameters")

    @property
    def frequency(self):
        # type: () -> numpy.ndarray
        return numpy.linspace(0, (len(self)-1)*self.frequency_step, len(self))

    @property
    def frequency_step(self):
        try:
            return self.attributes['frequency_step']
        except KeyError:
            return None

    @frequency_step.setter
    def frequency_step(self, value):
        assert value > 0, 'frequency_step has to be a positive numeric value'
        self.attributes['frequency_step'] = value

    @staticmethod
    def average(psds):
        frequency_step = psds[0].frequency_step
        data = psds[0].data

        for psd in psds[1:]:
            assert psd.frequency_step == frequency_step
            data = numpy.vstack((data, psd.data))

        data = numpy.sum(data, axis=0)/data.shape[0]

        new_psd = PowerSpectralDensity(data, frequency_step=frequency_step,
                                       time_stamp=psds[0].time_stamp)
        new_psd.attributes = psds[0].attributes
        # FIXME: does this make sense?
        # TODO: make sure it works for multi-column data
        new_psd.attributes['averaged'] = True

        return new_psd

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()

        for i in range(self.number_of_columns):
            try:
                label = self.attributes['label_{0}'.format(i)]
            except KeyError:
                label = None

            if max(self.frequency) > 5000:
                plt.plot(self.frequency/1000, self.data_column(i), label=label)
                plt.xlabel("Frequency (kHz)")
            else:
                plt.plot(self.frequency, self.data_column(i), label=label)
                plt.xlabel("Frequency (Hz)")

        plt.ylabel("PSD ({0})".format(self.unit))
        plt.yscale('log')
        plt.legend()

    def fit(self, **kwargs):
        fit_parameters = None
        covariance_matrix = None

        for i in range(self.number_of_columns):
            param, cov = fit_psd(self.frequency, self.data_column(i), **kwargs)

            param = param
            cov = cov

            if i is 0:
                fit_parameters = numpy.zeros((len(param),
                                              self.number_of_columns))
                covariance_matrix = numpy.zeros((len(param), len(param),
                                                 self.number_of_columns))

            fit_parameters[:, i] = param
            covariance_matrix[:, :, i] = cov

        self.attributes['fit_parameters'] = fit_parameters

        self.attributes['fit_covariance_matrix'] = covariance_matrix

        return fit_parameters

    def calibrate(self, T0=300, gas_pressure=1, T0_error=0,
                  gas_pressure_error=0):
        if 'fit_parameters' not in self.attributes.keys():
            self.fit()

        R = numpy.zeros((self.number_of_columns, ))
        R_error = numpy.zeros((self.number_of_columns, ))
        c = numpy.zeros((self.number_of_columns, ))
        c_error = numpy.zeros((self.number_of_columns, ))
        particle_size = numpy.zeros((self.number_of_columns, ))
        particle_size_error = numpy.zeros((self.number_of_columns, ))
        particle_mass = numpy.zeros((self.number_of_columns, ))
        particle_mass_error = numpy.zeros((self.number_of_columns, ))

        for i in range(self.number_of_columns):
            fit_parameters = self.attributes['fit_parameters'][..., i]

            if 'fit_covariance_matrix' in self.attributes.keys():
                fit_parameters_cov = \
                    self.attributes['fit_covariance_matrix'][..., i]
            else:
                fit_parameters_cov = None

            calibration = \
                calibrate((fit_parameters, fit_parameters_cov),
                          (T0, T0_error), (gas_pressure, gas_pressure_error))

            c[i] = calibration[0][0]
            c_error[i] = calibration[0][1]
            R[i] = calibration[1][0]
            R_error[i] = calibration[1][1]
            particle_size[i] = calibration[2][0]
            particle_size_error[i] = calibration[2][1]
            particle_mass[i] = calibration[3][0]
            particle_mass_error[i] = calibration[3][1]

        calibration_parameters = {'R': R,
                                  'R_error': R_error,
                                  'c': c,
                                  'c_error': c_error,
                                  'particle_size': particle_size,
                                  'particle_size_error': particle_size_error,
                                  'particle_mass': particle_mass,
                                  'particle_mass_error': particle_mass_error}

        calibration = Calibration(calibration_parameters,
                                  time_stamp=self.time_stamp)

        # TODO: set a source attribute
        # calibration.attributes['data_source'] = self...

        return calibration_parameters


class Calibration(Data):
    def __init__(self, calibration_data, data=None,  time_stamp=time(),
                 data_id=0):
        # type: (dict, h5py.Dataset, float, int)
        if data is None:
            data = []

        super(Calibration, self).__init__(data, time_stamp=time_stamp,
                                          name='calibration',
                                          data_id=data_id)

        for key in calibration_data.keys():
            self.attributes[key] = calibration_data[key]

    @property
    def temperature(self):
        try:
            return self.attributes['temperature']
        except KeyError:
            return None

    @property
    def pressure(self):
        try:
            return self.attributes['pressure']
        except KeyError:
            return None

    @property
    def R(self):
        try:
            return self.attributes['R']
        except KeyError:
            return None

    @R.setter
    def R(self, value):
        # type: (float)
        self.attributes['R'] = value

    @property
    def R_error(self):
        try:
            return self.attributes['R_error']
        except KeyError:
            return None

    @R_error.setter
    def R_error(self, value):
        # type: (float)
        self.attributes['R_error'] = value

    @property
    def c(self):
        try:
            return self.attributes['c']
        except KeyError:
            return None

    @c.setter
    def c(self, value):
        # type: (float)
        self.attributes['c'] = value

    @property
    def c_error(self):
        try:
            return self.attributes['c_error']
        except KeyError:
            return None

    @c_error.setter
    def c_error(self, value):
        # type: (float)
        self.attributes['c_error'] = value

    @property
    def particle_size(self):
        try:
            return self.attributes['particle_size']
        except KeyError:
            return None

    @property
    def particle_size_error(self):
        try:
            return self.attributes['particle_size_error']
        except KeyError:
            return None

    @property
    def particle_mass(self):
        try:
            return self.attributes['particle_mass']
        except KeyError:
            return None

    @property
    def particle_mass_error(self):
        try:
            return self.attributes['particle_mass_error']
        except KeyError:
            return None