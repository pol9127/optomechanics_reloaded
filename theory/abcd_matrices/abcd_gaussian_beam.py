import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

from collections import OrderedDict
from optomechanics.theory.gaussian_beam import rayleigh_length, width
from copy import copy
from optomechanics.theory.abcd_matrices.matrices import *

class GaussianBeam:
    def __init__(self, waist, wavelength):
        self.rayleigh = rayleigh_length(waist, waist, wavelength)
        self.waist = waist
        self.q = 1j * self.rayleigh
        self.wavelength = wavelength
        self.z = 0

    def __mul__(self, other):
        matrix = other.matrices['transmission']
        self.q = (self.q * matrix[0, 0] + matrix[0, 1]) / (self.q * matrix[1, 0] + matrix[1, 1])
        self.rayleigh = np.imag(self.q)
        self.waist = np.sqrt(self.rayleigh * self.wavelength / (other.tags['n_final'] * np.pi))
        self.z = np.real(self.q)

    def __truediv__(self, other):
        matrix = other.matrices['reflection']
        self.q = (self.q * matrix[0, 0] + matrix[0, 1]) / (self.q * matrix[1, 0] + matrix[1, 1])
        self.rayleigh = np.imag(self.q)
        self.waist = np.sqrt(self.rayleigh * self.wavelength / (other.tags['n_final'] * np.pi))
        self.z = np.real(self.q)

    def __floordiv__(self, other):
        matrix = other.matrices['reverse_transmission']
        self.q = (self.q * matrix[0, 0] + matrix[0, 1]) / (self.q * matrix[1, 0] + matrix[1, 1])
        self.rayleigh = np.imag(self.q)
        if 'n_initial' in other.tags:
            n = other.tags['n_initial']
        else:
            n = other.tags['n_final']
        self.waist = np.sqrt(self.rayleigh * self.wavelength / (n * np.pi))
        self.z = np.real(self.q)


class LensSystem:
    def __init__(self, path_height=20e-3, n_surrounding=1, tail=10e-3, n_range=[1,5]):
        self.__elements = {}
        self.reference_pos = 0
        self.refractive_indicies = [n_surrounding]
        self.element_counter = 0
        self.tail = tail
        self.height = path_height
        self.n_range = n_range
        self.offset = 0

    def __add_medium(self, pos, distance, refractive_index=1, patch_offset=0):
        self.__elements[(pos, 'm')] = Medium(distance, refractive_index, height=self.height, pos=pos, patch_offset=patch_offset, n_range=self.n_range)

    def add_flat_surface(self, n_final, relative_pos=None, absolute_pos=None):
        if relative_pos is None and absolute_pos is None:
            print('either relative or absolute position of element must be passed')
            return
        elif absolute_pos is not None:
            pos = absolute_pos
        else:
            pos = self.reference_pos + relative_pos
        n_initial = self.refractive_indicies[self.element_counter]
        self.__elements[(pos, 'c')] = FlatSurface(n_initial, n_final, height=self.height, pos=pos, n_range=self.n_range)
        self.refractive_indicies.append(n_final)
        self.element_counter += 1

    def add_curved_surface(self, n_final, curvature, relative_pos=None, absolute_pos=None, height_custom=None):
        if relative_pos is None and absolute_pos is None:
            print('either relative or absolute position of element must be passed')
            return
        elif absolute_pos is not None:
            pos = absolute_pos
        else:
            pos = self.reference_pos + relative_pos

        if height_custom is not None:
            height = height_custom
        else:
            height = self.height
        n_initial = self.refractive_indicies[self.element_counter]
        self.__elements[(pos, 'c')] = CurvedSurface(n_initial, n_final, curvature, height=height, pos=pos, n_range=self.n_range)
        self.refractive_indicies.append(n_final)
        self.element_counter += 1

    def fill_media(self):
        positions = [key[0] for key in self.components]

        if positions != []:
            comps = list(self.components.values())
            positions.append(positions[-1] + self.tail)
            for i in range(len(comps)):
                pos0 = positions[i]
                distance = positions[i + 1] - pos0
                self.__add_medium(pos0, distance, self.refractive_indicies[i + 1], comps[i].width)

            if positions[0] > 0:
                self.__add_medium(0, positions[0], self.refractive_indicies[0], comps[i].width)

    def initialize_beam(self, gaussian_beam_):
        self.gaussian_beam = copy(gaussian_beam_)
        self.gaussian_beam_transmission = [copy(self.gaussian_beam)]
        self.gaussian_beam_reflection = []
        self.beam_position = list(self.components.keys())[0]

    def propagate_beam(self, steps=None):
        if steps is None:
            components = list(self.components.keys())
        else:
            components = list(self.components.keys())[:steps]
        for comp in components:
            self.gaussian_beam * self.components[comp]
            self.gaussian_beam_transmission.append(copy(self.gaussian_beam))
            self.beam_position = comp

        # self.gaussian_beam_transmission = self.gaussian_beam_transmission[:-1]

    def reflect_beam(self, steps=2, offset=0):
        self.offset = offset
        comp_keys = list(self.components.keys())
        self.beam_position = comp_keys[comp_keys.index(self.beam_position) - self.offset]
        if self.beam_position == list(self.components.keys())[-1]:
            print('end of system reached, nowhere to reflect')
        elif self.beam_position[1] != 'm':
            print('cannot reflect from medium, please propagate until surface')
        else:
            last_component_idx = list(self.components.keys()).index(self.beam_position) + 1
            components = list(self.components.keys())[last_component_idx - steps + 1 : last_component_idx + 1][::-1]
            self.gaussian_beam = copy(self.gaussian_beam_transmission[-1 - self.offset])
            self.gaussian_beam / self.components[components[0]]
            self.gaussian_beam_reflection.append(copy(self.gaussian_beam))

            for comp in components[1:]:
                self.gaussian_beam // self.components[comp]
                self.gaussian_beam_reflection.append(copy(self.gaussian_beam))

    @property
    def components(self):
        return OrderedDict(sorted(self.__elements.items(), key=lambda t: t[0]))

    def show_setup(self, show=True, figure=None):
        patches = []
        for comp in self.components.values():
            comp_p = comp.patch
            if isinstance(comp_p, tuple):
                patches += [*comp_p]
            else:
                patches.append(comp_p)

        if patches != []:
            patch_col = PatchCollection(patches, alpha=1, facecolors=[p.get_facecolor() for p in patches],
                                        edgecolors=[p.get_edgecolor() for p in patches])
            if figure is None:
                fig, ax = plt.subplots()
            else:
                ax = figure.axes[0]
            ax.add_collection(patch_col)
            x_min = list(self.components.keys())[0][0]
            x_max = list(self.components.keys())[-1][0] + list(self.components.values())[-1].tags['distance']
            plt.axhline(y=0, xmin=0, xmax=1, color='k', linestyle='--')
            plt.xlim(x_min, x_max)
            plt.ylim(-0.6 * self.height, 0.6 * self.height)
            if show:
                plt.show()

    def show_beam(self, savefile=None):
        resolution = 500
        positions = np.array([key[0] for key in self.components])[:len(self.gaussian_beam_transmission)]
        if 'distance' in list(self.components.values())[:len(self.gaussian_beam_transmission)][-1].tags:
            tail = list(self.components.values())[:len(self.gaussian_beam_transmission)][-1].tags['distance']
            positions = np.hstack((positions, positions[-1] + tail))
        else:
            tail = 0
        # positions = np.hstack((positions, positions[-1] + tail))
        wave_radii_list = []
        wave_z_list = []
        for i in range(len(positions) - 1):
            w_0 = self.gaussian_beam_transmission[i].waist
            z_R = self.gaussian_beam_transmission[i].rayleigh
            z_0 = self.gaussian_beam_transmission[i].z
            if positions[i + 1] != positions[i]:
                z = np.linspace(positions[i], positions[i + 1], resolution)
            else:
                z = np.array([positions[i]])

            wave_radii = width(z - positions[i] + z_0, w_0, wavelength=self.gaussian_beam_transmission[i].wavelength,
                               rayleigh_len=z_R)
            wave_radii_list.append(wave_radii)
            wave_z_list.append(z)

        wave_radii = np.hstack(wave_radii_list)
        wave_z = np.hstack(wave_z_list)
        fig, ax = plt.subplots()
        ax.plot(wave_z, wave_radii,'r-')
        ax.plot(wave_z, -1 * wave_radii,'r-')

        reverse_positions = np.array([key[0] for key in self.components])[len(self.gaussian_beam_transmission) - self.offset - len(self.gaussian_beam_reflection):len(self.gaussian_beam_transmission) - self.offset][::-1]
        if reverse_positions != []:

            reverse_wave_radii_list = []
            reverse_wave_z_list = []
            for i in range(len(reverse_positions) - 1):
                w_0 = self.gaussian_beam_reflection[i + 1].waist
                z_R = self.gaussian_beam_reflection[i + 1].rayleigh
                z_0 = self.gaussian_beam_reflection[i + 1].z
                if reverse_positions[i + 1] != reverse_positions[i]:
                    z = np.linspace(reverse_positions[i], reverse_positions[i + 1], resolution)
                else:
                    z = np.array([reverse_positions[i]])
                reverse_wave_radii = width(z - reverse_positions[i] + z_0, w_0, wavelength=self.gaussian_beam_reflection[i].wavelength,
                                   rayleigh_len=z_R)[::-1]
                reverse_wave_radii_list.append(reverse_wave_radii)
                reverse_wave_z_list.append(z)

            reverse_wave_radii = np.hstack(reverse_wave_radii_list)
            reverse_wave_z = np.hstack(reverse_wave_z_list)

            ax.plot(reverse_wave_z, reverse_wave_radii,'k-')
            ax.plot(reverse_wave_z, -1 * reverse_wave_radii,'k-')

        self.show_setup(show=False, figure=fig)
        # ylim0 = ax.get_ylim()
        # wave_radii_max = np.max(wave_radii)
        # if ylim0[1] < wave_radii_max:
        #     ax.set_ylim(-1.1 * wave_radii_max, 1.1 * wave_radii_max)

        if savefile is not None:
            plt.savefig(savefile, dpi=300)
        else:
            plt.show()