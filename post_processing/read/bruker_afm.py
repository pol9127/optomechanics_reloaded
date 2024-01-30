import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

class AFMScan(object):
    def __init__(self, filename):
        files = glob(filename + '*.flt')
        self.names = [os.path.split(f)[-1][:os.path.split(f)[-1].rfind('.')] for f in files]

        self.meta_data = {}
        self.data = {}

        for fn, nm in zip(files, self.names):
            mt, dt = self.read_flt(fn)
            self.meta_data[nm] = mt
            self.data[nm] = dt


    def read_flt(self, fname):
        with open(fname, 'br') as f:
            file_buffer = f.read()
        delimiter = file_buffer.rfind(b'[Data]') + 8
        header_buffer = file_buffer[:delimiter].replace(b'\xb5', b'u').replace(b'\xb0', b'deg').decode('utf-8',
                                                                                                       'ignore')
        data_buffer = file_buffer[delimiter:]
        header = dict([h.split('=') for h in header_buffer.split('\r\n') if len(h.split('=')) == 2])
        z_scaling, z_unit = header['ZTransferCoefficient'].split(' ')
        data = np.frombuffer(data_buffer, dtype=np.float32).reshape(int(header['ResolutionX']),
                                                                    int(header['ResolutionY'])) * float(z_scaling)
        return header, data

    @property
    def xy(self):
        meta_data = list(self.meta_data.values())[0]
        x_range, x_unit =  meta_data['ScanRangeX'].split(' ')
        x_range = float(x_range)
        x = np.linspace(0, x_range, int(meta_data['ResolutionX']))
        y_range, y_unit =  meta_data['ScanRangeY'].split(' ')
        y_range = float(y_range)
        y = np.linspace(0, y_range, int(meta_data['ResolutionY']))
        X, Y = np.meshgrid(x, y)
        return X, Y, x_unit, y_unit

    def plot_data(self, name, show=True, subtract_2d_poly=False):
        X, Y, x_unit, y_unit = self.xy
        fig, ax = plt.subplots()
        if not subtract_2d_poly:
            data = self.data[name]
            m = ax.pcolormesh(X, Y, data)
        else:
            data = self.data[name] - np.polynomial.polynomial.polyval2d(X, Y, self.polyfit2d(name))
            m = ax.pcolormesh(X, Y, data)
        ax.set_ylabel(y_unit, fontsize=15)
        ax.set_xlabel(x_unit, fontsize=15)
        plt.colorbar(m)
        plt.tight_layout()
        if show:
            plt.show()
        return data

    def polyfit2d(self, name):
        X, Y, x_unit, y_unit = self.xy
        X = X.flatten()
        Y = Y.flatten()

        A = np.array([X * 0 + 1, Y, Y ** 2, X, X * Y, X * Y ** 2, X ** 2, X ** 2 * Y, X ** 2 * Y ** 2]).T
        B = self.data[name].flatten()

        coeff, r, rank, s = np.linalg.lstsq(A, B)
        coeff = coeff.reshape((3, 3))
        return coeff


if __name__ == '__main__':
    name = 'fine_1'
    afm = AFMScan(filename='/media/cavitydata/Cavity Lab/data/projects/particle laser/afm_scan/Y2O3_05Er_136nm_IPA/' + name)
    afm.plot_data(name + '.SIG_TOPO_FRW', show=False)
    afm.plot_data(name + '.SIG_TOPO_BKW', show=False)
    plt.show()

    # names = ['big_scan', 'second_scan', 'first_scan', 'second_scan']
    #
    # afm_data = [AFMScan(filename='/media/cavitydata/Cavity Lab/data/projects/particle laser/afm_scan/Y2O3_136nm/' + name) for name in names]
    #
    # vmins = np.array([np.min(afm.data[name + '.SIG_TOPO_FRW']) for afm, name in zip(afm_data, names)])
    # vmaxs = np.array([np.max(afm.data[name + '.SIG_TOPO_FRW']) for afm, name in zip(afm_data, names)])
    # vmin = np.min(vmins)
    # vmax = np.max(vmaxs)
    #
    # show_y_label = [True, False, True, False]
    # show_x_label = [False, False, True, True]
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # for afm, name, ax, show_y_lab, show_x_lab in zip(afm_data, names, axes.flat, show_y_label, show_x_label):
    #     X, Y, x_unit, y_unit = afm.xy
    #     m = ax.pcolormesh(X, Y, afm.data[name + '.SIG_TOPO_FRW'])
    #     ax.set_title(name)
    #     if show_y_lab:
    #         ax.set_ylabel(y_unit, fontsize=15)
    #     if show_x_lab:
    #         ax.set_xlabel(x_unit, fontsize=15)
    # plt.tight_layout()
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # plt.colorbar(m, cax=cbar_ax)
    # plt.show()

