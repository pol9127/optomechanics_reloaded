import numpy as np
from scipy.special import hermite
from scipy.optimize import curve_fit
from optomechanics.theory.gaussian_beam import rayleigh_length, width, wavefront_radius
import scipy as sc
from math import factorial, inf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
from mpl_toolkits.mplot3d import Axes3D
import itertools
from timeit import default_timer
import h5py
import pandas as pd

class Cavity:
    def __init__(self, wavelength, x_bounds=None, y_bounds=None, x_resolution=None, y_resolution=None):
        self.wavelength = wavelength
        if x_bounds is not None and x_resolution is not None:
            self.x = np.linspace(*x_bounds, x_resolution)
        else:
            self.x = None
        if y_bounds is not None and y_resolution is not None:
            self.y = np.linspace(*y_bounds, y_resolution)
        else:
            self.y = None
        if self.x is not None and self.y is not None:
            self.X, self.Y = np.meshgrid(self.x, self.y)

        self.profile = {}
        self.profile_x = {}
        self.profile_y = {}
        self.curvature_x = {}
        self.curvature_y = {}
        self.mm_matrices = {'A' : None, 'B' : None, 'M' : None}
        self.lengths = None
        self.waists = {}
        self.centers = {}
        self.modes = None
        self.mirrors = {'A' : None, 'B' : None}
        self.eigenvalues = None
        self.eigenvectors = None
        self.model_profile_params = {'A' : None, 'B' : None}
        self.dummy_flat = {'A' : False, 'B' : False}

    def reinitialize_frame(self, x_bounds, y_bounds, x_resolution, y_resolution):
        self.x = np.linspace(*x_bounds, x_resolution)
        self.y = np.linspace(*y_bounds, y_resolution)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def create_parabola(self, curvature_x, curvature_y, mirror='B'):
        self.curvature_x[mirror] = curvature_x
        self.curvature_y[mirror] = curvature_y

        a = np.sqrt(2 * self.curvature_x[mirror])
        b = np.sqrt(2 * self.curvature_y[mirror])

        self.profile[mirror] = self.X ** 2 / a ** 2 + self.Y ** 2 / b ** 2
        self.profile_x[mirror] = self.x ** 2 / a ** 2
        self.profile_y[mirror] = self.y ** 2 / b ** 2
        self.mirrors[mirror] = ['ideal', 'parabolic']

    def create_spherical(self, curvature_x, curvature_y, mirror='B'):
        self.curvature_x[mirror] = curvature_x
        self.curvature_y[mirror] = curvature_y

        self.profile_x[mirror] = (abs(self.curvature_x[mirror]) - np.sqrt(self.curvature_x[mirror] ** 2 - self.x ** 2)) * np.sign(self.curvature_x[mirror])
        self.profile_y[mirror] = (abs(self.curvature_y[mirror]) - np.sqrt(self.curvature_y[mirror] ** 2 - self.y ** 2)) * np.sign(self.curvature_x[mirror])

        if self.curvature_x[mirror] == self.curvature_y[mirror]:
            self.profile[mirror] = (abs(self.curvature_x[mirror]) - np.sqrt(self.curvature_x[mirror] ** 2 - self.X ** 2 - self.Y ** 2)) * np.sign(self.curvature_x[mirror])

        self.mirrors[mirror] = ['ideal', 'spherical']

    def create_flat(self, mirror='A'):
        self.dummy_flat[mirror] = True
        self.curvature_x[mirror] = inf
        self.curvature_y[mirror] = inf

        self.profile[mirror] = np.zeros(self.X.shape)
        self.profile_x[mirror] = np.zeros(len(self.x))
        self.profile_y[mirror] = np.zeros(len(self.y))
        self.mirrors[mirror] = ['ideal', 'dummy_flat']

    def create_gaussian(self, A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0, mirror='B'):
        self.profile[mirror] = self.elliptical_gaussian([self.X, self.Y], A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0)
        self.profile_x[mirror] = self.elliptical_gaussian([self.x, 0], A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0)
        self.profile_y[mirror] = self.elliptical_gaussian([0, self.y], A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0)
        self.curvature_x[mirror] = 0
        self.curvature_y[mirror] = 0

        self.mirrors[mirror] = ['ideal', 'gaussian']

    @staticmethod
    def herm_gauss_mode_1d(m, w_0, wavelength, mirror_points, z):
        rayleigh = rayleigh_length(w_0, wavelength=wavelength)
        waist = width(z, w_0, wavelength=wavelength, rayleigh_len=rayleigh)
        curvature = wavefront_radius(z, rayleigh_len=rayleigh, width_x=w_0, wavelength=wavelength)
        phase = (2 * m + 1) / 2 * np.arctan(z / rayleigh)
        normalization = np.sqrt(np.sqrt(2) / (2 ** m * float(factorial(m)) * np.sqrt(np.pi))) / np.sqrt(waist)
        herm_poly = hermite(m)(np.sqrt(2) * mirror_points / waist)
        exponential = np.exp(-mirror_points ** 2 / waist ** 2 + 1j * ((np.pi * mirror_points ** 2) / (wavelength * curvature) - phase))
        return normalization * herm_poly * exponential

    @staticmethod
    def herm_gauss_mode_2d(m, n, w_0_x, w_0_y, wavelength,  mirror_points_2d, z_x, z_y=None):
        if z_y is None:
            z_y = z_x
        herm_gauss_mode_x = Cavity.herm_gauss_mode_1d(m, w_0_x, wavelength, mirror_points_2d[0], z_x)
        herm_gauss_mode_y = Cavity.herm_gauss_mode_1d(n, w_0_y, wavelength, mirror_points_2d[1], z_y)
        return herm_gauss_mode_x * herm_gauss_mode_y

    def mode_overlap_1d(self, w_0, length, center, m_1, m_2=None, axis='x', mirror='B'):
        if axis == 'x':
            delta_2 = self.profile_x[mirror]
            mirror_points = self.x
        elif axis == 'y':
            delta_2 = self.profile_y[mirror]
            mirror_points = self.y

        if mirror == 'A':
            length_AB = -center
        elif mirror == 'B':
            length_AB = length - center

        if delta_2 is None:
            sys.exit('Error: Please specify mirror geometry before projecting wave functions.')

        tem_1 = self.herm_gauss_mode_1d(m_1, w_0, self.wavelength, mirror_points, length_AB)
        if m_2 is None:
            tem_2 = self.herm_gauss_mode_1d(m_1, w_0, self.wavelength, mirror_points, length_AB)
        else:
            tem_2 = self.herm_gauss_mode_1d(m_2, w_0, self.wavelength, mirror_points, length_AB)
        return np.exp(-1j * 4 * np.pi * delta_2 / self.wavelength) * tem_1 * tem_2

    def mode_overlap_2d(self, w_0_x, w_0_y, length, center_x, center_y, mn_1, mn_2, mirror='B'):
        mirror_points_2d = [self.X, self.Y]
        delta_2 = self.profile[mirror]

        if mirror == 'A':
            length_AB_x = -center_x
            length_AB_y = -center_y
        elif mirror == 'B':
            length_AB_x = length - center_x
            length_AB_y = length - center_y

        if delta_2 is None:
            sys.exit('Error: Please specify mirror geometry before projecting wave functions.')

        tem_1 = self.herm_gauss_mode_2d(*mn_1, w_0_x, w_0_y, self.wavelength, mirror_points_2d, length_AB_x, length_AB_y)
        tem_2 = self.herm_gauss_mode_2d(*mn_2, w_0_x, w_0_y, self.wavelength, mirror_points_2d, length_AB_x, length_AB_y)
        return np.exp(-1j * 4 * np.pi * delta_2 / self.wavelength) * tem_1 * tem_2

    def best_waists(self, waist_guess, center_guess, lengths, axis='x'):
        self.lengths = lengths
        param_guess = np.array([waist_guess, center_guess])

        if not self.dummy_flat['A'] and not self.dummy_flat['B']:
            waists = []
            centers = []
            for length in lengths:
                def mode_overlap_wrapped(x):
                    overlap_A = -1 * abs(np.mean(self.mode_overlap_1d(x[0], length, x[1], 0, 0, axis, mirror='A')))
                    overlap_B = -1 * abs(np.mean(self.mode_overlap_1d(x[0], length, x[1], 0, 0, axis, mirror='B')))
                    return overlap_A + overlap_B

                minimization = minimize(mode_overlap_wrapped, param_guess,
                                        bounds=[(1e-9, 1), (0, length)])
                centers.append(minimization.x[1])
                waists.append(minimization.x[0])
            self.waists[axis] = np.array(waists)
            self.centers[axis] = np.array(centers)

        elif self.dummy_flat['A']:
            waists = []
            for length in lengths:
                def mode_overlap_wrapped(x):
                    overlap = -1 * abs(np.mean(self.mode_overlap_1d(x, length, 0, 0, 0, axis, mirror='B')))
                    return overlap

                minimization = minimize(mode_overlap_wrapped, param_guess[:1],
                                        bounds=[(1e-9, 1)])
                waists.append(minimization.x[0])
            self.waists[axis] = np.array(waists)
            self.centers[axis] = np.zeros(len(self.waists[axis]))

        elif self.dummy_flat['B']:
            waists = []
            for length in lengths:
                def mode_overlap_wrapped(x):
                    overlap = -1 * abs(np.mean(self.mode_overlap_1d(x, length, 0, 0, 0, axis, mirror='A')))
                    return overlap

                minimization = minimize(mode_overlap_wrapped, param_guess[:1],
                                        bounds=[(1e-9, 1)])
                waists.append(minimization.x[0])
            self.waists[axis] = np.array(waists)
            self.centers[axis] = np.zeros(len(self.waists[axis]))

        return self.waists[axis], self.centers[axis]

    def mode_mixing_matrices(self, n_modes, w_0_xs=None, w_0_ys=None, lengths=None, debug=False):
        if not self.dummy_flat['A']:
            self._mode_mixing_matrices(n_modes, w_0_xs=w_0_xs, w_0_ys=w_0_ys, lengths=lengths, debug=debug, mirror='A')
        if not self.dummy_flat['B']:
            self._mode_mixing_matrices(n_modes, w_0_xs=w_0_xs, w_0_ys=w_0_ys, lengths=lengths, debug=debug, mirror='B')

        if not self.dummy_flat['A'] and not self.dummy_flat['B']:
            self.mm_matrices['M'] = [A * B * np.exp(4j * np.pi * length / self.wavelength) for A, B, length in zip(self.mm_matrices['A'], self.mm_matrices['B'], self.lengths)]
        elif self.dummy_flat['A']:
            self.mm_matrices['M'] = [B * np.exp(4j * np.pi * length / self.wavelength) for B, length in zip(self.mm_matrices['B'], self.lengths)]
        elif self.dummy_flat['B']:
            self.mm_matrices['M'] = [A * np.exp(4j * np.pi * length / self.wavelength) for A, length in zip(self.mm_matrices['A'], self.lengths)]


    def _mode_mixing_matrices(self, n_modes, w_0_xs=None, w_0_ys=None, lengths=None, debug=False, mirror='B'):
        if lengths is not None:
            self.lengths = lengths
        if w_0_xs is not None:
            self.waists['x'] = w_0_xs
        if w_0_ys is not None:
            self.waists['y'] = w_0_ys

        if mirror == 'A':
            lengths_AB_x = self.centers['x']
            lengths_AB_y = self.centers['y']
        elif mirror == 'B':
            lengths_AB_x = self.lengths - self.centers['x']
            lengths_AB_y = self.lengths - self.centers['y']

        if debug:
            t_0 = default_timer()

        complete_range = np.arange(n_modes)
        nested_list = [list(zip(np.arange(n_modes - cr), cr * np.ones(n_modes - cr, dtype=int))) for cr in complete_range]
        unsorted_list = list(itertools.chain(*nested_list))
        self.modes = sorted(unsorted_list, key=lambda x: (np.mean(x), np.std(x), x[0]))
        if debug:
            t_1 = default_timer()
            print(t_1 - t_0, 'prepared mn list')

        mirror_profile = np.exp(-1j * 4 * np.pi * self.profile[mirror] / self.wavelength)
        self.mm_matrices[mirror] = []
        for w_0_x, w_0_y, length_x, length_y in zip(self.waists['x'], self.waists['y'], lengths_AB_x, lengths_AB_y):
            if debug:
                t_1 = default_timer()
            mm_matrix = np.zeros((len(self.modes), len(self.modes)), dtype=np.complex64)
            mode_dict_x = {key : self.herm_gauss_mode_1d(key, w_0_x, self.wavelength, self.X, length_x) for key in complete_range}
            mode_dict_y = {key: self.herm_gauss_mode_1d(key, w_0_y, self.wavelength, self.Y, length_y) for key in complete_range}
            mode_dict_2d = {key : mode_dict_x[key[0]] * mode_dict_y[key[1]] for key in self.modes}

            if debug:
                t_2 = default_timer()
                print (t_2 - t_1, 'prepared dicts', w_0_x, w_0_y, length_x, length_y)

            for i in range(len(self.modes)):
                for j in range(i, len(self.modes)):
                    mm_matrix[i, j] = np.mean(mode_dict_2d[self.modes[i]] * mode_dict_2d[self.modes[j]] * mirror_profile)

            if debug:
                t_3 = default_timer()
                print (t_3 - t_2, 'filled matrix', w_0_x, w_0_y, length_x, length_y)

            mm_matrix = mm_matrix + mm_matrix.T - np.diag(mm_matrix.diagonal())
            if debug:
                t_4 = default_timer()
                print (t_4 - t_3, 'symmetrized matrix', w_0_x, w_0_y, length_x, length_y)

            self.mm_matrices[mirror].append(mm_matrix)

    def mode_losses(self):
        if self.mm_matrices['M'] is None:
            sys.exit('Error: No mode mixing matrices have been calculated so far!')

        self.eigenvalues = []
        self.eigenvectors = []
        for mm_matrix, length in zip(self.mm_matrices['M'], self.lengths):
            eigenvals, eigenvecs = np.linalg.eig(mm_matrix)
            self.eigenvalues.append(eigenvals)
            self.eigenvectors.append(eigenvecs)

    def plot_mode_overlaps_1d(self, w_0s, lengths, centers, m_1, m_2=None, axis='x', mirror='B', xlabel=r'mirror position [$\mu$m]', ylabel='Overlap', xscale=1e6, yscale=1e-3, fontsize=15, show=True, plot=True):
        if not isinstance(w_0s, (np.ndarray, list)):
            w_0s = np.array([w_0s])
        if not isinstance(lengths, (np.ndarray, list)):
            lengths = np.array([lengths])
        if not isinstance(centers, (np.ndarray, list)):
            w_0s = np.array([centers])

        if axis == 'x':
            mirror_points = self.x
        elif axis == 'y':
            mirror_points = self.y

        if plot:
            fig, ax = plt.subplots(3, sharex=True)
            ax[0].grid(True)
            ax[1].grid(True)
            ax[2].grid(True)
            ax[2].set_xlabel(xlabel, fontsize=fontsize)
            ax[0].set_ylabel(r'$\Re$ ' + ylabel, fontsize=fontsize)
            ax[1].set_ylabel(r'$\Im$ ' + ylabel, fontsize=fontsize)
            ax[2].set_ylabel(r'$abs$ ' + ylabel, fontsize=fontsize)

            plt.setp(ax[0].get_yticklabels(), fontsize=fontsize)
            plt.setp(ax[1].get_yticklabels(), fontsize=fontsize)
            plt.setp(ax[2].get_yticklabels(), fontsize=fontsize)
            plt.setp(ax[0].get_xticklabels(), fontsize=fontsize)
            plt.setp(ax[1].get_xticklabels(), fontsize=fontsize)
            plt.setp(ax[2].get_xticklabels(), fontsize=fontsize)

        if len(w_0s) == len(lengths) and len(w_0s) == len(centers):
            overlaps = np.array([self.mode_overlap_1d(w_0, length, center, m_1, m_2, axis, mirror=mirror) for w_0, length, center in zip(w_0s, lengths, centers)]).T
        elif len(w_0s) == 1 and len(centers) == 1:
            overlaps = np.array([self.mode_overlap_1d(w_0s[0], length, centers[0], m_1, m_2, axis, mirror=mirror) for length in lengths]).T
        elif len(lengths) == 1 and len(centers) == 1:
            overlaps = np.array([self.mode_overlap_1d(w_0, lengths[0], centers[0], m_1, m_2, axis, mirror=mirror) for w_0 in w_0s]).T
        elif len(w_0s) == 1 and len(lengths) == 1:
            overlaps = np.array([self.mode_overlap_1d(w_0s[0], lengths[0], center, m_1, m_2, axis, mirror=mirror) for center in centers]).T

        else:
            sys.exit('Error: Length of waists and cavity lengths are not matching and neither length is equal to one.')

        if plot:
            ax[0].plot(mirror_points * xscale, np.real(overlaps) * yscale)
            ax[1].plot(mirror_points * xscale, np.imag(overlaps) * yscale)
            ax[2].plot(mirror_points * xscale, abs(overlaps) * yscale)

            plt.tight_layout()

            if show:
                plt.show()

        return overlaps

    def plot_profile(self, axis='x', kind='3d', mirror='B', show=True):
        mirror_points = None
        mirror_points_2d = None
        if axis == 'x':
            mirror_points = self.x
            delta = self.profile_x[mirror]
            if delta is None:
                sys.exit('Error: Please specify mirror geometry before trying to plot.')
        elif axis == 'y':
            mirror_points = self.y
            delta = self.profile_y[mirror]
            if delta is None:
                sys.exit('Error: Please specify mirror geometry before trying to plot')
        elif axis == 'both':
            mirror_points_2d = [self.X, self.Y]
            delta_2d = self.profile[mirror]
            if delta_2d is None:
                sys.exit('Error: Please specify mirror geometry before trying to plot')

        if mirror_points is not None:
            fig, ax = plt.subplots()
            ax.plot(mirror_points, delta, marker='o', linestyle='--')
        else:
            if kind == '3d':
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.plot_surface(*mirror_points_2d, delta_2d)
            elif kind == '2d':
                fig, ax = plt.subplots()
                ax.pcolormesh(*mirror_points_2d, delta_2d)

        ax.grid(True)
        if show:
            plt.show()

    def plot_integrated_mode_overlaps_1d(self, waists, lengths, centers, m_1, m_2=None, plot_against=['lengths'], axis='x', mirror='B', xlabel=r'length [$\mu$m]', ylabel='Overlap', xscale=1e6, yscale=1e-3, fontsize=15, show=True):
        quantities = {'lengths' : lengths, 'waists' : waists, 'centers' : centers}
        if not isinstance(plot_against, (np.ndarray, list)):
            plot_against = np.array([plot_against])
        if not isinstance(waists, (np.ndarray, list)):
            waists = np.array([waists])
        if not isinstance(lengths, (np.ndarray, list)):
            lengths = np.array([lengths])
        if not isinstance(centers, (np.ndarray, list)):
            waists = np.array([centers])


        if mirror in ['A', 'B']:
            if len(waists) == len(lengths) and len(waists) == len(centers) and len(plot_against) == 1:
                overlaps = np.array([abs(np.mean(self.mode_overlap_1d(w_0, length, center, m_1, m_2, axis, mirror=mirror))) for w_0, length, center in zip(waists, lengths, centers)]).T
            elif len(waists) == 1 and len(centers) == 1:
                overlaps = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], length, centers[0], m_1, m_2, axis, mirror=mirror))) for length in lengths]).T
            elif len(lengths) == 1 and len(centers) == 1:
                overlaps = np.array([abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], centers[0], m_1, m_2, axis, mirror=mirror))) for w_0 in waists]).T
            elif len(waists) == 1 and len(lengths) == 1:
                overlaps = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], lengths[0], center, m_1, m_2, axis, mirror=mirror))) for center in centers]).T
            elif len(plot_against) == 2 and 'waists' not in plot_against:
                overlaps = np.array([[abs(np.mean(self.mode_overlap_1d(waists[0], length, center, m_1, m_2, axis, mirror=mirror))) for length in lengths] for center in centers]).T
                ax_labels = ['centers', 'lengths']
            elif len(plot_against) == 2 and 'lengths' not in plot_against:
                overlaps = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], center, m_1, m_2, axis, mirror=mirror))) for w_0 in waists] for center in centers]).T
                ax_labels = ['centers', 'waists']
            elif len(plot_against) == 2 and 'centers' not in plot_against:
                overlaps = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, length, centers[0], m_1, m_2, axis, mirror=mirror))) for w_0 in waists] for length in lengths]).T
                ax_labels = ['lengths', 'waists']
            else:
                sys.exit('Error: Length of waists and cavity lengths are not matching and neither length is equal to one.')
        elif mirror == 'both':
            if len(waists) == len(lengths) and len(waists) == len(centers) and len(plot_against) == 1:
                overlaps_A = np.array([abs(np.mean(self.mode_overlap_1d(w_0, length, center, m_1, m_2, axis, mirror='A'))) for w_0, length, center in zip(waists, lengths, centers)]).T
                overlaps_B = np.array([abs(np.mean(self.mode_overlap_1d(w_0, length, center, m_1, m_2, axis, mirror='B'))) for w_0, length, center in zip(waists, lengths, centers)]).T
                overlaps = overlaps_A + overlaps_B
            elif len(waists) == 1 and len(centers) == 1:
                overlaps_A = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], length, centers[0], m_1, m_2, axis, mirror='A'))) for length in lengths]).T
                overlaps_B = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], length, centers[0], m_1, m_2, axis, mirror='B'))) for length in lengths]).T
                overlaps = overlaps_A + overlaps_B
            elif len(lengths) == 1 and len(centers) == 1:
                overlaps_A = np.array([abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], centers[0], m_1, m_2, axis, mirror='A'))) for w_0 in waists]).T
                overlaps_B = np.array([abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], centers[0], m_1, m_2, axis, mirror='B'))) for w_0 in waists]).T
                overlaps = overlaps_A + overlaps_B
            elif len(waists) == 1 and len(lengths) == 1:
                overlaps_A = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], lengths[0], center, m_1, m_2, axis, mirror='A'))) for center in centers]).T
                overlaps_B = np.array([abs(np.mean(self.mode_overlap_1d(waists[0], lengths[0], center, m_1, m_2, axis, mirror='B'))) for center in centers]).T
                overlaps = overlaps_A + overlaps_B
            elif len(plot_against) == 2 and 'waists' not in plot_against:
                overlaps_A = np.array([[abs(np.mean(self.mode_overlap_1d(waists[0], length, center, m_1, m_2, axis, mirror='A'))) for length in lengths] for center in centers]).T
                overlaps_B = np.array([[abs(np.mean(self.mode_overlap_1d(waists[0], length, center, m_1, m_2, axis, mirror='B'))) for length in lengths] for center in centers]).T
                overlaps = overlaps_A + overlaps_B
                ax_labels = ['centers', 'lengths']
            elif len(plot_against) == 2 and 'lengths' not in plot_against:
                overlaps_A = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], center, m_1, m_2, axis, mirror='A'))) for w_0 in waists] for center in centers]).T
                overlaps_B = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, lengths[0], center, m_1, m_2, axis, mirror='B'))) for w_0 in waists] for center in centers]).T
                overlaps = overlaps_A + overlaps_B
                ax_labels = ['centers', 'waists']
            elif len(plot_against) == 2 and 'centers' not in plot_against:
                overlaps_A = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, length, centers[0], m_1, m_2, axis, mirror='A'))) for w_0 in waists] for length in lengths]).T
                overlaps_B = np.array([[abs(np.mean(self.mode_overlap_1d(w_0, length, centers[0], m_1, m_2, axis, mirror='B'))) for w_0 in waists] for length in lengths]).T
                overlaps = overlaps_A + overlaps_B
                ax_labels = ['lengths', 'waists']

            else:
                sys.exit('Error: Length of waists and cavity lengths are not matching and neither length is equal to one.')

        if len(plot_against) == 1:
            fig, ax = plt.subplots()
            ax.set_xlabel(xlabel, fontsize=fontsize)
            ax.set_ylabel(ylabel, fontsize=fontsize)

            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            ax.plot(quantities[plot_against[0]] * xscale, overlaps * yscale)

        elif len(plot_against) == 2:
            X, Y = np.meshgrid(quantities[ax_labels[0]], quantities[ax_labels[1]])

            fig, ax = plt.subplots()
            ax.pcolormesh(X, Y, overlaps)

        ax.grid(True)
        plt.tight_layout()

        if show:
            plt.show()
        return overlaps

    def plot_herm_gauss_mode_1d(self, m, w_0, length, axis='x'):
        if axis == 'x':
            mirror_points = self.x
        elif axis == 'y':
            mirror_points = self.y

        fig, ax = plt.subplots()
        ax.plot(mirror_points, abs(self.herm_gauss_mode_1d(m, w_0, self.wavelength, mirror_points, length)))
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_herm_gauss_mode_2d(self, m, n, w_0_x, w_0_y, length, kind='3d', show=True, xlabel='', ylabel='', xscale=1, yscale=1, fontsize=15, colorbar=True):
        mirror_points_2d = np.meshgrid(self.x, self.y)

        if kind == '3d':
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(mirror_points_2d[0] * xscale, mirror_points_2d[1] * yscale, abs(self.herm_gauss_mode_2d(m, n, w_0_x, w_0_y, self.wavelength, mirror_points_2d, length)))
        elif kind == '2d':
            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(mirror_points_2d[0] * xscale, mirror_points_2d[1] * yscale, abs(self.herm_gauss_mode_2d(m, n, w_0_x, w_0_y, self.wavelength, mirror_points_2d, length)))
            if colorbar:
                fig.colorbar(mesh)

        ax.grid(True)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=15)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.tight_layout()
        if show:
            plt.show()
        
    def plot_and_calc_mode_mixing_matrix(self, n_modes, w_0_x, w_0_y, length, debug=False):
        if debug:
            t_0 = default_timer()

        complete_range = np.arange(n_modes)
        nested_list = [list(zip(np.arange(n_modes - cr), cr * np.ones(n_modes - cr, dtype=int))) for cr in complete_range]
        unsorted_list = list(itertools.chain(*nested_list))
        sorted_list = sorted(unsorted_list, key=lambda x: (np.mean(x), np.std(x), x[0]))
        mm_matrix = np.zeros((len(sorted_list), len(sorted_list)))
        if debug:
            t_1 = default_timer()
            print (t_1 - t_0, 'prepared mn list')

        mode_dict_x = {key : self.herm_gauss_mode_1d(key, w_0_x, self.wavelength, self.X, length) for key in complete_range}
        mode_dict_y = {key: self.herm_gauss_mode_1d(key, w_0_y, self.wavelength, self.Y, length) for key in complete_range}
        mode_dict_2d = {key : mode_dict_x[key[0]] * mode_dict_y[key[1]] for key in sorted_list}

        mirror = np.exp(-1j * 4 * np.pi * self.profile / self.wavelength)
        if debug:
            t_2 = default_timer()
            print (t_2 - t_1, 'prepared dicts')

        for i in range(len(sorted_list)):
            for j in range(i, len(sorted_list)):
                mm_matrix[i, j] = abs(np.mean(mode_dict_2d[sorted_list[i]] * mode_dict_2d[sorted_list[j]] * mirror))

        if debug:
            t_3 = default_timer()
            print (t_3 - t_2, 'filled matrix')

        mm_matrix = mm_matrix + mm_matrix.T - np.diag(mm_matrix.diagonal())
        if debug:
            t_4 = default_timer()
            print (t_4 - t_3, 'symmetrized matrix')

        fig, ax = plt.subplots()
        ax.pcolormesh(mm_matrix)
        ax.grid(True)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mode_mixing_matrix(mm_matrix):
        fig, ax = plt.subplots()
        ax.pcolormesh(abs(mm_matrix))
        ax.grid(True)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_best_waists(self, waist_guess, center_guess, lengths, axis='x',show=True):
        waists, centers = self.best_waists(waist_guess, center_guess, lengths, axis=axis)
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot(lengths * 1e6, waists * 1e6, marker='o', label='waists')
        ax2.plot(lengths * 1e6, centers * 1e6, color='red', marker='o', label='centers')
        ax.set_xlabel(r'Length [$\mu$m]', fontsize=15)
        ax.set_ylabel(r'Waists [$\mu$m]', fontsize=15)
        ax2.set_ylabel(r'Centers [$\mu$m]', fontsize=15)
        ax.legend(loc='upper left', fontsize=15)
        ax2.legend(loc='upper right', fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=15)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax2.get_yticklabels(), fontsize=15)
        plt.grid()
        plt.tight_layout()
        if show:
            plt.show()

    def plot_hybrid_modes(self, ratios, w_0_x, w_0_y, length, show=True, save_file=None):
        if self.modes is None:
            sys.exit('Error: No modes have been calculated so far.')

        mode_dict_2d = {key: self.herm_gauss_mode_2d(*key, w_0_x, w_0_y, self.wavelength, [self.X, self.Y], length) for key in self.modes}

        hybrid_mode = np.sum([ratio * mode_dict_2d[mode] for ratio, mode in zip(ratios, self.modes)], axis=0)

        fig, ax = plt.subplots()
        ax.pcolormesh(self.X, self.Y, abs(hybrid_mode))
        ax.grid(True)
        ax.invert_yaxis()
        plt.tight_layout()
        if save_file is not None:
            plt.savefig(save_file, dpi=300)
        if show:
            plt.show()

    def export_hdf5(self, filename):
        hdf5_dump = h5py.File(filename, 'w')

        hdf5_dump.attrs['wavelength'] = self.wavelength
        mirror_profiles = hdf5_dump.create_group('mirror_profiles')
        mm_matrix_group = hdf5_dump.create_group('mode_mixing_matrices')
        mm_matrices = {}
        for mirror in self.mirrors:
            if self.mirrors[mirror] is not None:
                if self.mirrors[mirror][0] == 'ideal':
                    hdf5_dump.attrs[mirror + ' Curvatures'] = [self.curvature_x[mirror], self.curvature_y[mirror]]
                    hdf5_dump.attrs[mirror + ' Kind'] = self.mirrors[mirror][0]
                    hdf5_dump.attrs[mirror + ' Geometry'] = self.mirrors[mirror][1]

                mirror_profiles.create_dataset(mirror, data=self.profile[mirror])
                mirror_profiles.create_dataset(mirror + '_x', data=self.profile_x[mirror])
                mirror_profiles.create_dataset(mirror + '_y', data=self.profile_y[mirror])
                mm_matrices[mirror] = mm_matrix_group.create_group(mirror)

                if self.mm_matrices[mirror] is not None:
                    for idx, mm_matrix in zip(np.arange(len(self.mm_matrices[mirror])), self.mm_matrices[mirror]):
                        mm_matrices[mirror].create_dataset(str(idx).zfill(4), data=mm_matrix)

        if self.mm_matrices['M'] is not None:
            mm_matrices['M'] = mm_matrix_group.create_group('M')
            for idx, mm_matrix in zip(np.arange(len(self.mm_matrices['M'])), self.mm_matrices['M']):
                mm_matrices['M'].create_dataset(str(idx).zfill(4), data=mm_matrix)

        if self.lengths is not None:
            cavity_data = hdf5_dump.create_group('cavity_data')
            cavity_data.create_dataset('cavity_lengths', data=self.lengths)

        if self.modes is not None:
            cavity_data.create_dataset('modes', data=self.modes)

        cavity_data.create_dataset('grid_X', data=self.X)
        cavity_data.create_dataset('grid_Y', data=self.Y)
        cavity_data.create_dataset('grid_x', data=self.x)
        cavity_data.create_dataset('grid_y', data=self.y)

        for axis in self.waists:
            cavity_data.create_dataset('waists_' + axis, data=self.waists[axis])

        for axis in self.centers:
            cavity_data.create_dataset('centers_' + axis, data=self.waists[axis])

        if self.eigenvalues is not None:
            eigen_data = hdf5_dump.create_group('eigen_data')
            eigen_values = eigen_data.create_group('eigen_values')
            eigen_vectors = eigen_data.create_group('eigen_vectors')
            for idx, eigenvals, eigenvecs in zip(range(len(self.eigenvalues)), self.eigenvalues, self.eigenvectors):
                eigen_values.create_dataset(str(idx).zfill(4), data=eigenvals)
                eigen_vectors.create_dataset(str(idx).zfill(4), data=eigenvecs)

        hdf5_dump.close()

    def import_hdf5(self, filename):
        hdf5_dump = h5py.File(filename, 'r')
        self.wavelength = hdf5_dump.attrs['wavelength']

        for mirror in ['A', 'B']:
            if mirror in hdf5_dump['mirror_profiles'].keys():
                kind = hdf5_dump.attrs[mirror + ' Kind']
                self.curvature_x[mirror], self.curvature_y[mirror] = hdf5_dump.attrs[mirror + ' Curvatures']

                if kind == 'ideal':
                    geom = hdf5_dump.attrs[mirror + ' Geometry']
                    self.mirrors[mirror] = [kind, geom]
                else:
                    self.mirrors[mirror] = kind

                self.profile[mirror] = hdf5_dump['mirror_profiles/' + mirror][:]
                self.profile_x[mirror] = hdf5_dump['mirror_profiles/' + mirror + '_x'][:]
                self.profile_y[mirror] = hdf5_dump['mirror_profiles/' + mirror + '_y'][:]

                if 'mode_mixing_matrices' in hdf5_dump.keys():
                    if mirror in hdf5_dump['mode_mixing_matrices'].keys():
                        self.mm_matrices[mirror] = [x[:] for x in hdf5_dump['mode_mixing_matrices/'+ mirror].values()]

        if 'M' in hdf5_dump['mode_mixing_matrices'].keys():
            self.mm_matrices['M'] = [x[:] for x in hdf5_dump['mode_mixing_matrices/M'].values()]

        if 'cavity_data' in hdf5_dump.keys():
            self.lengths = hdf5_dump['cavity_data/cavity_lengths'][:]
            if 'centers_x' in hdf5_dump['cavity_data'].keys():
                self.centers['x'] = hdf5_dump['cavity_data/centers_x'][:]
            if 'centers_y' in hdf5_dump['cavity_data'].keys():
                self.centers['y'] = hdf5_dump['cavity_data/centers_y'][:]
            if 'waists_x' in hdf5_dump['cavity_data'].keys():
                self.waists['x'] = hdf5_dump['cavity_data/waists_x'][:]
            if 'waists_y' in hdf5_dump['cavity_data'].keys():
                self.waists['y'] = hdf5_dump['cavity_data/waists_y'][:]
            if 'modes' in hdf5_dump['cavity_data'].keys():
                self.modes = [tuple(x) for x in hdf5_dump['cavity_data/modes'][:]]

        self.X = hdf5_dump['cavity_data/grid_X'][:]
        self.Y = hdf5_dump['cavity_data/grid_Y'][:]
        self.x = hdf5_dump['cavity_data/grid_x'][:]
        self.y = hdf5_dump['cavity_data/grid_y'][:]

        if 'eigen_data' in hdf5_dump.keys():
            if 'eigen_values' in hdf5_dump['eigen_data'].keys():
                self.eigenvalues = [x[:] for x in hdf5_dump['eigen_data/eigen_values'].values()]
            if 'eigen_vectors' in hdf5_dump['eigen_data'].keys():
                self.eigenvectors = [x[:] for x in hdf5_dump['eigen_data/eigen_vectors'].values()]

    def read_profile(self, filename, mirror='B'):
        with open(filename) as f:
            header = {}
            f.readline()
            f.readline()
            f.readline()
            for i in range(45):
                line =f.readline().strip()
                if line != '':
                    line_split = line.replace('\"', '').split(',')

                    header_val = line_split[1]
                    header_key = line_split[0]
                    for el in line_split[2:]:
                        header_key +=  ', ' + el
                    header[header_key] = header_val

        self.profile[mirror] = pd.read_csv(filename, skiprows=52, delimiter=',', header=None).values * float(header['Z calibration, nm/digit']) * 1e-9
        self.profile[mirror] -= np.min(self.profile[mirror])
        self.x = (np.arange(self.profile[mirror].shape[1]) - 0.5 * self.profile[mirror].shape[1]) * float(header['XY calibration, nm/pixel']) * 1e-9
        self.y = (np.arange(self.profile[mirror].shape[0]) - 0.5 * self.profile[mirror].shape[0]) * float(header['XY calibration, nm/pixel']) * 1e-9
        self.X, self.Y = np.meshgrid(self.x, self.y)

        def elliptical_gaussian_raveled(xy, A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0):
            return self.elliptical_gaussian(xy, A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0).ravel()

        params = curve_fit(elliptical_gaussian_raveled, [self.X, self.Y], self.profile[mirror].ravel(), p0=[-10e-6, 0, 0, 20e-6, 20e-6, 1, 1, 0])[0]
        self.model_profile_params[mirror] = {'A' : params[0],
                                             'x_0' : params[1],
                                             'y_0' : params[2],
                                             'sigma_x' : params[3],
                                             'sigma_y' : params[4],
                                             'p_x' : params[5],
                                             'p_y' : params[6],
                                             'z_0': params[7]}

        self.x -= self.model_profile_params[mirror]['x_0']
        self.y -= self.model_profile_params[mirror]['y_0']
        self.X -= self.model_profile_params[mirror]['x_0']
        self.Y -= self.model_profile_params[mirror]['y_0']

        x_center = np.argmin(abs(self.x))
        y_center = np.argmin(abs(self.y))

        self.profile_x[mirror] = self.profile[mirror][y_center, :]
        self.profile_y[mirror] = self.profile[mirror][:, x_center]


    def plot_fitted_real_mirror(self, axis='x', mirror='B', show=True):
        if self.model_profile_params[mirror] is None:
            sys.exit('Error: No Profile was imported and fitted!')

        if axis=='x':
            args = self.x
            model = self.elliptical_gaussian([self.x, 0], **self.model_profile_params[mirror])
            real = self.profile_x[mirror]
        elif axis=='y':
            args = self.y
            model = self.elliptical_gaussian([0, self.y], **self.model_profile_params[mirror])
            real = self.profile_y[mirror]


        fig, ax = plt.subplots()
        ax.plot(args, model, '--')
        ax.plot(args, real)
        ax.grid(True)
        plt.tight_layout()
        if show:
            plt.show()

    @staticmethod
    def elliptical_gaussian(xy, A, x_0, y_0, sigma_x, sigma_y, p_x, p_y, z_0):
        x, y = xy
        return A * np.exp(-1 * ((x - x_0) ** 2 / (2 * sigma_x ** 2)) ** p_x - ((y - y_0) ** 2 / (2 * sigma_y ** 2)) ** p_y) + z_0






# laser = 1550e-9
# curvature = 800e-6
# x_bounds = [-100 * 1e-6, 100 * 1e-6]
# y_bounds = [-100 * 1e-6, 100 * 1e-6]
# x_resolution = 1536
# y_resolution = 2048
# # mode_orders = 5
# lengths = np.linspace(100, 700, 20) * 1e-6
# # #
# cav = Cavity(laser)
# cav.read_profile('/home/dwindey/gitwork/optomechanics/theory/cavity_physics/5,10.csv', mirror='B')
#
# print(cav.model_profile_params['B'])

# cav.reinitialize_frame(x_bounds, y_bounds, x_resolution, y_resolution)
# cav.model_profile_params['x_0'] = 0
# cav.model_profile_params['y_0'] = 0
# cav.create_spherical(curvature, curvature, mirror='B')
# # cav.create_gaussian(**cav.model_profile_params['B'], mirror='B')
# cav.create_flat(mirror='A')
# cav.plot_best_waists(10e-6, 0, lengths)
#
# cav.create_spherical(-curvature - 1000, -curvature - 1000, mirror='A')
#
# cav.plot_profile(axis='both', mirror='B')
#
# # waists = np.linspace(5, 12, 15) * 1e-6
# # lengths = np.array([300e-6])
# # centers = np.linspace(0, 500, 30) * 1e-6
# # cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='y', mirror='B', show=False)
# # cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='y', mirror='A', show=False)
# # cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='y', mirror='both')
#
# # plt.figure()
# # plt.pcolormesh(cav.X * 1e6, cav.Y * 1e6, data * 1e6)
# # plt.grid()
# # plt.colorbar()
# # plt.tight_layout()
# # plt.show()
#
# # #
# # cav.create_spherical(curvature, curvature, mirror='B')
# # cav.create_spherical(-curvature - 1000, -curvature - 1000, mirror='A')
# # #
# lengths = np.linspace(100, 700, 3) * 1e-6
# #
# cav.plot_best_waists(10e-6, 50e-6, lengths, axis='x', show=False)
# cav.plot_best_waists(10e-6, 50e-6, lengths, axis='y', show=False)
#
# cav.plot_integrated_mode_overlaps_1d(cav.waists['x'], lengths, cav.centers['x'], 0, plot_against=['lengths'], axis='x', mirror='both', show=False)
# cav.plot_integrated_mode_overlaps_1d(cav.waists['y'], lengths, cav.centers['y'], 0, plot_against=['lengths'], axis='y', mirror='both')
#
# # cav.best_waists(15e-6, 50e-6, lengths, axis='x')
# # cav.best_waists(15e-6, 50e-6, lengths, axis='y')
# #
# cav.mode_mixing_matrices(mode_orders, debug=True)
# cav.mode_losses()
#
# first_vector_components = np.array([[abs(vec[0]) for vec in vecs] for vecs in cav.eigenvectors])
# order = np.argsort(first_vector_components, axis=1)[:,::-1]
# eigen_vectors_ordered = [vecs[od] for vecs, od in zip(cav.eigenvectors, order)]
# eigen_values_ordered = [vals[od] for vals, od in zip(cav.eigenvalues, order)]
#
# idx = 1
# cav.plot_hybrid_modes(eigen_vectors_ordered[idx][0], cav.waists['x'][idx], cav.waists['y'][idx], cav.lengths[idx] - cav.centers['x'][idx])
# cav.export_hdf5('big_dump.hdf5')

#cav.import_hdf5('second_dump.hdf5')
#cav.plot_hybrid_modes(cav.eigenvectors[4][54], cav.waists['x'][4], cav.waists['y'][4], cav.lengths[4] - cav.centers['x'][4])



# cav.plot_best_waists(15e-6, 50e-6, lengths)
# lengths = 200e-6
# waists, centers = cav.best_waists(12e-6, 120e-6, lengths, 0)
# waists = np.linspace(5,20, 20) * 1e-6
# centers = np.linspace(0, 20, 10) * 1e-6
# cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='x', mirror='A', show=False)
# cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='x', mirror='B', show=False)
# cav.plot_integrated_mode_overlaps_1d(waists, lengths, centers, 0, plot_against=['waists', 'centers'], axis='x', mirror='both', show=True)


# fig, ax = plt.subplots()
# ax2 = ax.twinx()
# ax.plot(lengths, waists, marker='o')
# ax2.plot(lengths, centers, color='red', marker='o')
# plt.grid()
# plt.show()

# lengths = np.array([200]) * 1e-6
# waists = np.array([7]) * 1e-6
# waists = np.linspace(5, 20, 18) * 1e-6
#
#
# centers = np.linspace(0, 500, 20) * 1e-6
# centers = np.array([0]) * 1e-6

# ov_As = []
# ov_Bs = []
# for waist in waists:
#     overlaps_A = cav.plot_mode_overlaps_1d(waist, lengths, centers, 0, mirror='A', show=False, plot=False)
#     overlaps_B = cav.plot_mode_overlaps_1d(waist, lengths, centers, 0, mirror='B', show=False, plot=False)
#     ov_As.append(abs(np.mean(overlaps_A, axis=0)))
#     ov_Bs.append(abs(np.mean(overlaps_B, axis=0)))
#
# ov_As = np.array(ov_As)
# ov_Bs = np.array(ov_Bs)
# ovs = ov_As + ov_Bs


# fig, ax = plt.subplots()
# ax.plot(waists, abs(np.mean(overlaps_A, axis=0)))
# ax.plot(waists, abs(np.mean(overlaps_B, axis=0)))
# ax2 = ax.twinx()
# ax2.plot(waists, abs(np.mean(overlaps_B, axis=0)) + abs(np.mean(overlaps_A, axis=0)))

# ax.plot(lengths, waists, marker='o')
# ax2.plot(lengths, centers, color='red', marker='o')
# plt.grid()
# plt.show()

# X, Y = np.meshgrid(centers, waists)
#
# fig, ax = plt.subplots()
# ax.pcolormesh(X, Y, ovs)
# ax.grid(True)
# ax.invert_yaxis()
# plt.tight_layout()
# plt.show()





# cav.mode_mixing_matrices(mode_orders, waists, waists, lengths, debug=True)
#
# cav.plot_mode_mixing_matrix(cav.mm_matrices[0])
# cav.plot_mode_mixing_matrix(cav.mm_matrices[1])
#
# cav.mode_losses()




#
# length = np.array([100e-6])
# waist = cav.best_waists(7e-6, length, 0)[0]
# cav.mode_mixing_matrix(mode_orders, waist[0], waist[0], length[0])
# cav.plot_mode_mixing_matrix()



# cav.plot_profile('both', kind='2d')

# cav.plot_herm_gauss_mode_1d(0, 10e-6, 300e-6,axis='y')
# cav.plot_herm_gauss_mode_2d(1, 2, 10e-6, 10e-6, 300e-6, kind='2d')

# lengths = np.linspace(10, 790, 100) * 1e-6
# lengths = np.array([100e-6])
# waists = cav.best_waists(7e-6, lengths, 0)[0]
# overlaps_integrated = cav.plot_integrated_mode_overlaps_1d(waists, lengths, 0, axis='x')
#
# overlaps = cav.plot_mode_overlaps_1d(waists, lengths, 0)


