from QtModularUiPack.Widgets import EmptyFrame
from ViewModel.ToolViewModels.ArduinoTriggerViewModel import ArduinoTriggerViewModel
from PyQt5.QtWidgets import QGridLayout, QLineEdit, QPushButton, QLabel, QCheckBox, QSpinBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class ArduinoTriggerFrame(EmptyFrame):

    name = 'Arduino Trigger'

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.data_context = ArduinoTriggerViewModel()
        self._layout = QGridLayout()
        self._layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.setLayout(self._layout)
        self._setup_()

    def _setup_(self):
        # set title
        title = QLabel('Trigger Signal')
        title.setFont(QFont('Arial', 10))
        self._layout.addWidget(title, 0, 0, 1, 2, Qt.AlignLeft)

        # trigger button
        trigger_button = QPushButton('trigger')
        trigger_button.clicked.connect(self.data_context.trigger)
        self._layout.addWidget(trigger_button, 1, 0, 1, 2, Qt.AlignLeft)

        # config
        self._layout.addWidget(self.add_widget(QLabel('port:'), width=40), 2, 0)
        self._layout.addWidget(self.add_widget(QLineEdit(), 'port', 'setText', width=50), 2, 1)
        trigger_experiment_check = self.add_widget(QCheckBox('trigger experiments'), 'trigger_experiment', 'setChecked')
        self.bindings.set_binding('experiment_available', trigger_experiment_check, 'setEnabled')
        self._layout.addWidget(trigger_experiment_check, 3, 1, 1, 2, Qt.AlignLeft)
        self._layout.addWidget(QLabel('experiment index:'), 4, 0)
        experiment_index_box = QSpinBox()
        self.bindings.set_binding('experiment_index', experiment_index_box, 'setValue')
        self._layout.addWidget(experiment_index_box, 4, 1)


if __name__ == '__main__':
    ArduinoTriggerFrame.standalone_application()
