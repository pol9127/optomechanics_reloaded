import pandas as pd
import tables
import numpy as np
import matplotlib.pyplot as plt
import warnings

class DataFrame(pd.DataFrame):
    def idxmax(self, axis=0, skipna=True, n=1, col=None, interval=None):
        if interval is None:
            interval = [0, self.shape[axis]]
        argmaxima = []
        tmp = self.copy()
        for i in range(n):
            idx_tmp = pd.DataFrame.idxmax(tmp.iloc[interval[0]: interval[1]-i, :], axis, skipna)[col]
            tmp = tmp.drop(idx_tmp)
            argmaxima.append(idx_tmp)
        return argmaxima

class OsciData:
    def __init__(self, filename_):
        self.channel_metadata = {}
        self.channel_data = {}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.data = tables.open_file(filename_, driver='H5FD_CORE')
            for nd in self.data.iter_nodes('/Waveforms'):
                self.channel_metadata[nd._v_name] =  {key :
                                                          nd._v_attrs[key].decode() if isinstance(nd._v_attrs[key],
                                                                                                  np.bytes_)
                                                                                        and key != 'Label'
                                                          else nd._v_attrs[key] for key in nd._v_attrs._f_list()}
                for dt in nd._f_iter_nodes():
                    self.channel_data[nd._v_name] = dt.read()
            self.build_dataframes()

    def build_dataframes(self):
        some_meta_data = list(self.channel_metadata.values())[0]
        runtime = np.arange(0, some_meta_data['NumPoints'])*some_meta_data['XInc'] + some_meta_data['XOrg']
        header = np.sort(list(self.channel_metadata.keys()))
        channel_header = [h for h in header if h.startswith('Channel')]
        math_header = [h for h in header if h.startswith('Math')]
        data = np.vstack((runtime, np.vstack((self.channel_data[key] for key in channel_header)))).T
        channel_header = ['Runtime [s]'] + [hd + ' [V]' for hd in channel_header]
        self.DataFrame = DataFrame(data=data, columns=channel_header)
        self.MetaFrame = pd.DataFrame.from_dict(self.channel_metadata).drop(['Label'], axis=0)




if __name__ == '__main__':
    filename = r'/home/dominik/Desktop/noise-floor.h5'
    data = OsciData(filename)
    # data.DataFrame.columns = ['Runtime [s]', 'Piezo [V]', 'Piezo Amp [V]', 'PDH Error [V]', 'Intensity Refl. [V]']
    # data.DataFrame.plot(x='Runtime [s]', y = ['PDH Error [V]'], grid=True, legend=False, xlim=(0.003,0.00315))
    # plt.savefig('PDH_Error.png',dpi=300)
    # plt.show()