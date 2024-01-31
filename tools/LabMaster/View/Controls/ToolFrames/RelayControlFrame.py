from PyQt5.QtWidgets import QGridLayout, QLabel, QLineEdit, QPushButton, QCheckBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ViewModel.ToolViewModels.RelayControlViewModel import RelayControlViewModel
from QtModularUiPack.Widgets import EmptyFrame


class RelayControlFrame(EmptyFrame):
    """
    Frame to control the Arduino relay
    """

    name = 'Relay Control'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_context = RelayControlViewModel()
        self._layout = QGridLayout()
        self._layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self._layout)
        self._setup_()

    def _setup_(self):
        # set title
        title = QLabel(self.name)
        title.setFont(QFont('Arial', 10))
        self._layout.addWidget(title, 0, 0, 1, 2, Qt.AlignLeft)

        # controls
        self._layout.addWidget(self.add_widget(QCheckBox('enabled'), 'enabled', 'setChecked'), 1, 0, 1, 3, Qt.AlignLeft)
        self._layout.addWidget(QLabel('Port:'), 2, 0)
        self._layout.addWidget(self.add_widget(QLineEdit(), 'port', 'setText', width=80), 2, 1, 1, 1, Qt.AlignLeft)
        connect_button = QPushButton()
        self.bindings.set_binding('connect_button_text', connect_button, 'setText')
        connect_button.clicked.connect(self.data_context.connect_disconnect)
        self._layout.addWidget(connect_button, 2, 2, 1, 1, Qt.AlignLeft)


if __name__ == '__main__':
    RelayControlFrame.standalone_application()
