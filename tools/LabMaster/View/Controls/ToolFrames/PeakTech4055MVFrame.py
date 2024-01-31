from PyQt5.QtWidgets import QGridLayout, QLabel, QDoubleSpinBox, QComboBox, QPushButton, QCheckBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from ViewModel.ToolViewModels.PeakTech4055MVViewModel import PeakTech4055MVViewModel, WF
from QtModularUiPack.Widgets import EmptyFrame


class PeakTech4055MVFrame(EmptyFrame):
    """
    Frame to control the PeakTech signal generator
    """

    name = 'PeakTech 4055MV Signal Generator'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_context = PeakTech4055MVViewModel()
        self._layout = QGridLayout()
        self._layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self._layout)
        self._setup_()

    def _add_spinbox_(self, variable, minimum, maximum, increment=1, units=None):
        spinbox = QDoubleSpinBox()
        if units is not None:
            spinbox.setSuffix(' '+units)
        spinbox.setMinimum(minimum)
        spinbox.setMaximum(maximum)
        spinbox.setSingleStep(increment)
        self.bindings.set_binding(variable, spinbox, 'setValue')
        return spinbox

    def _add_combobox_(self):
        combobox = QComboBox()
        combobox.addItems(WF)
        self.bindings.set_binding('waveform_index', combobox, 'setCurrentIndex')
        return combobox

    @staticmethod
    def _add_button_(text, method):
        button = QPushButton(text)
        button.clicked.connect(method)
        return button

    def _setup_(self):
        # set title
        title = QLabel(self.name)
        title.setFont(QFont('Arial', 10))
        self._layout.addWidget(title, 0, 0, 1, 2, Qt.AlignLeft)

        # controls
        self._layout.addWidget(QLabel('amplitude:'), 1, 0)
        self._layout.addWidget(self._add_spinbox_('amplitude', 0, 9, 0.01, 'Vpp'), 1, 1)
        self._layout.addWidget(QLabel('offset'), 2, 0)
        self._layout.addWidget(self._add_spinbox_('offset', -9.350, 9.350, 0.01, 'Vdc'), 2, 1)
        self._layout.addWidget(QLabel('frequency:'), 3, 0)
        self._layout.addWidget(self._add_spinbox_('frequency', 0.01e-3, 3e6, 1, 'Hz'), 3, 1)
        self._layout.addWidget(QLabel('waveform:'), 4, 0)
        self._layout.addWidget(self._add_combobox_(), 4, 1)
        self._layout.addWidget(self._add_button_('apply', self.data_context.apply), 1, 2)
        self._layout.addWidget(self.add_widget(QCheckBox('enable output'), 'output_enabled', 'setChecked'), 2, 2)


if __name__ == '__main__':
    PeakTech4055MVFrame.standalone_application()
