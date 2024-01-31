from PyQt5.QtWidgets import QTableWidget, QMenuBar, QVBoxLayout
from QtModularUiPack.Widgets import EmptyFrame
from ViewModel.ToolViewModels.VideoParameterEditorViewModel import VideoParameterEditorViewModel


class VideoParameterEditor(EmptyFrame):

    name = 'Video Parameter Editor'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_context = VideoParameterEditorViewModel()
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)
        self._setup_()

    def _setup_(self):
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu('File')
        save_action = file_menu.addAction('Save')
        save_action.triggered.connect(self.data_context.save_data)
        self._layout.addWidget(menu_bar)

        table = QTableWidget()
        self.bindings.set_source('data_source', table)
        self._layout.addWidget(table)


if __name__ == '__main__':
    VideoParameterEditor.standalone_application()
