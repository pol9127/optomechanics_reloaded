from PyQt5.QtWidgets import *
from QtModularUiPack.Widgets import ModularFrameHost
from QtModularUiPack.Widgets.DataBinding import BindingEnabledWidget
from QtModularUiPack.ViewModels import BaseContextAwareViewModel
from QtModularUiPack.Framework import is_non_strict_type, is_non_strict_subclass


class MainWindow(QMainWindow, BindingEnabledWidget):
    """
    This is the main window of the Lab Master application
    """

    _count = 0
    _reload_tools = 'Reload Tools'

    def __init__(self, data_context, parent=None):
        super(MainWindow, self).__init__(parent)

        self.data_context = data_context
        self._vm = data_context

        self._setup_()

    def _on_child_data_context_received_(self, tool_frame, child_data_context):
        """
        Callback for handling new data contexts that have been added in the tool frame section
        :param tool_frame: tool frame that caused the event
        :param child_data_context: data context that was added
        :return:
        """
        self._vm.other_data_contexts.append(child_data_context)

        if is_non_strict_subclass(type(child_data_context), BaseContextAwareViewModel):
            self._vm.connect_context_aware_view_model(child_data_context)

    def _on_child_data_context_removed_(self, child_data_context):
        """
        Callback for handling new data contexts that have been removed in the tool frame section
        :param child_data_context: data context that was removed
        :return:
        """
        if child_data_context in self._vm.other_data_contexts:
            self._vm.other_data_contexts.remove(child_data_context)

        if is_non_strict_type(type(child_data_context), BaseContextAwareViewModel):
            self._vm.disconnect_context_aware_view_model(child_data_context)

    def _setup_(self):
        """
        Generate UI
        """

        # content
        self._main_widget = ModularFrameHost(self)
        self._main_widget.data_context_received.connect(self._on_child_data_context_received_)
        self._main_widget.data_context_removed.connect(self._on_child_data_context_removed_)
        self._main_widget.load(self._vm.frame_save_path)

        # menu bar
        self._bar = self.menuBar()
        self._tools = self._bar.addMenu('Tools')
        self._tools.addAction(self._reload_tools)
        self._tools.triggered[QAction].connect(self._handle_tools_)

        self.setCentralWidget(self._main_widget)
        self.setWindowTitle('Lab Master')
        self.resize(self._vm.width, self._vm.height)

    def closeEvent(self, *args, **kwargs):
        self._main_widget.save(self._vm.frame_save_path)
        self._vm.save_configuration()   # save configuration on close

    def _handle_tools_(self, q):
        action = q.text()
        if action == self._reload_tools:
            self._main_widget.reload_possible_frames()

