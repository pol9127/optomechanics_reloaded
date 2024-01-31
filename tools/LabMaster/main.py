from QtModularUiPack.Widgets import ModularApplication
from View.MainWindow import MainWindow
from ViewModel.MainWindowViewModel import MainWindowViewModel
from PyQt5.Qt import QApplication
import sys

# run the Lab Master from here
ModularApplication.standalone_application(title='Lab Master',
                                          window_size=(1280, 720),
                                          frame_search_path='./View/Controls/ToolFrames',
                                          configuration_path='LabMasterSettings.json')
"""app = QApplication(sys.argv)
view_model = MainWindowViewModel()
main_window = MainWindow(view_model)
main_window.show()
sys.exit(app.exec_())"""
