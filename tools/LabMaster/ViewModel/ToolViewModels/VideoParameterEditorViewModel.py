from QtModularUiPack.ViewModels import BaseViewModel
from QtModularUiPack.Framework import ObservableList
import os
import json


class VideoParameterEditorViewModel(BaseViewModel):

    name = 'video_parameter_editor'

    @property
    def parameter_file_name(self):
        return self._parameter_file_name

    @parameter_file_name.setter
    def parameter_file_name(self, value):
        self._parameter_file_name = value
        self.notify_change('parameter_file_name')

    @property
    def data_source(self):
        return self._data_source

    @data_source.setter
    def data_source(self, value):
        self._data_source = value
        self.notify_change('data_source')

    def __init__(self):
        super().__init__()
        self._search_directory = 'C:/Users/Dominik Werner/polybox/Master Thesis/code/VideoAnalyzerTestProjects'
        self._parameter_file_name = 'new_metric_run_parameters.json'
        self._data_source = ObservableList()
        self.collect_data()

    def collect_data(self):
        self.data_source.clear()
        for sub_dir_list in os.walk(self._search_directory):
            for file in sub_dir_list[2]:
                if file == self.parameter_file_name:
                    path = sub_dir_list[0]+'/'+file
                    with open(path, 'r') as fp:
                        parameters = json.load(fp)
                        observable_parameters = ObservableParameters()
                        observable_parameters.file = path
                        observable_parameters.bottom_px = parameters['bottom_px']
                        observable_parameters.roi_x = parameters['roi'][0]
                        observable_parameters.roi_y = parameters['roi'][1]
                        observable_parameters.roi_width = parameters['roi'][2]
                        observable_parameters.roi_height = parameters['roi'][3]
                        observable_parameters.metric = parameters['metric']
                        observable_parameters.working_distance = parameters['working_distance']
                        self.data_source.append(observable_parameters)

    def save_data(self):
        for entry in self.data_source:
            with open(entry.file, 'w') as fp:
                parameters = {'bottom_px': entry.bottom_px,
                              'roi': [entry.roi_x, entry.roi_y, entry.roi_width, entry.roi_height],
                              'metric': entry.metric,
                              'working_distance': entry.working_distance}
                fp.write(json.dumps(parameters))


class ObservableParameters(BaseViewModel):

    @property
    def bottom_px(self):
        return self._bottom_px

    @bottom_px.setter
    def bottom_px(self, value):
        self._bottom_px = value
        self.notify_change('bottom_px')

    @property
    def roi_x(self):
        return self._roi_x

    @roi_x.setter
    def roi_x(self, value):
        self._roi_x = value
        self.notify_change('roi_x')

    @property
    def roi_y(self):
        return self._roi_y

    @roi_y.setter
    def roi_y(self, value):
        self._roi_y = value
        self.notify_change('roi_y')

    @property
    def roi_width(self):
        return self._roi_width

    @roi_width.setter
    def roi_width(self, value):
        self._roi_width = value
        self.notify_change('roi_width')

    @property
    def roi_height(self):
        return self._roi_height

    @roi_height.setter
    def roi_height(self, value):
        self._roi_height = value
        self.notify_change('roi_height')

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value
        self.notify_change('metric')

    @property
    def working_distance(self):
        return self._working_distance

    @working_distance.setter
    def working_distance(self, value):
        self._working_distance = value
        self.notify_change('working_distance')

    @property
    def file(self):
        return self._file

    @file.setter
    def file(self, value):
        self._file = value
        self.notify_change('file')

    def __init__(self):
        super().__init__()
        self._bottom_px = 0
        self._roi_x = 0
        self._roi_y = 0
        self._roi_width = 0
        self._roi_height = 0
        self._metric = 1
        self._working_distance = 0
        self._file = ''
