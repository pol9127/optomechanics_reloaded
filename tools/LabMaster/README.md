# Introduction
The LabMaster is a PyQt5 application that allows the user to quickly and efficiently design user interfaces based on the needs in a lab environment. The data binding capabilities makes this framework ideal to work with concepts like MVVM (Model-View-Viewmodel) or MVP (Model-View-Presentation). New functionalities do not require the application to be re-started but can be tested after a click on the "reload tools" button.

Key-Features:
- Fully Customize-able UI
- Pre-Defined Tool Frames (can be used as templates)
- Data-Binding
- Flexible Module Management (reload modules at runtime)

# Quick Start
Start the application by running main.py from the optomechanics/tools/LabMaster folder.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial1.png)

Click on the button on the bottom left of the window as indicated by the red arrow.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial2.png)

Now you can split the window into several sub frames and resize them to your liking. If you close the application it will save the frame settings as you have configured them and load them again on the next start.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial3.png)

Go to an empty frame and again click on the bottom left button. Now in the sub menu view select one of the available tools.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial4.png)

Once you have designated each frame the real power of the Lab Master will become apparent.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial5.png)

The device control sections are pretty standard and should be self explanatory. The really special tool in this window is the sub frame titled "Tool Command Line". This command line has the ability to control all other frames in the application. This can be very useful when testing a new experiment or when a new frame is being developed.

To test this type the following into the console:

    ```python
    tools.help()
    ```

This function returns all the available tools. If you add or remove frames, the tools variable in the console will get updated.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial6.png)

Lastly, the console frame is a fully-fledged python console. You can import modules and even create plots from within the console.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/tutorial7.png)

Now you are ready to write your own tool frame.

# Developing your own frame
The development of custom frames is the one of the core functionalities of the Lab Master. This short tutorial will guide you through the steps necessary to create a very simple frame which can easily be extended.

Start by creating a new python file in the folder /optomechanics/tools/LabMaster/View/Controls/ToolFrames:

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/advanced_tutorial1.PNG)

Now open the file and begin with the proper imports:

```python
from View.Controls.ToolFrames.EmptyFrame import EmptyFrame
from PyQt5.QtWidgets import QLineEdit, QPushButton, QGridLayout
```

The first line imports the EmptyFrame class which we will inherit. The second line makes the PyQt controls available to us. write the frame:

from View.Controls.ToolFrames.EmptyFrame import EmptyFrame
from PyQt5.QtWidgets import QLineEdit, QPushButton, QGridLayout

```python
class HelloWorldFrame(EmptyFrame):

    name = 'Hello World Frame'

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._layout = QGridLayout()
        self.setLayout(self._layout)
        self._setup_()

    def _setup_(self):
        upper_box = QLineEdit('Hello World')
        lower_box = QLineEdit('Goodbye')
        switch_button = QPushButton('switch')

        self._layout.addWidget(upper_box, 0, 0)
        self._layout.addWidget(lower_box, 1, 0)
        self._layout.addWidget(switch_button, 1, 1)
```

If you now go into the Lab Master and click on the button "Tools -> Reload Tools" you should now be able to load in your frame into the main frame.

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/advanced_tutorial2.PNG)

This is all nice but not really useful. And while this tutorial will ultimately not produce something super useful the next few steps will show you how elegant the static behavior of your frame can be made dynamic.
Start by adding a new file to /optomechanics/tools/LabMaster/ViewModel/ToolViewModels

![tutorial image](https://git.ee.ethz.ch/dwindey/optomechanics/raw/LabMasterBranch/tools/LabMaster/img/advanced_tutorial3.PNG)

Open the file and type the following:

```python
from ViewModel.BaseViewModel import BaseViewModel

class HelloWorldViewModel(BaseViewModel):

    @property
    def upper_text(self):
        return self._upper_text

    @upper_text.setter
    def upper_text(self, value):
        self._upper_text = value
        self.notify_change('upper_text')

    @property
    def lower_text(self):
        return self._lower_text

    @lower_text.setter
    def lower_text(self, value):
        self._lower_text = value
        self.notify_change('lower_text')

    def __init__(self):
        super().__init__()
        self._upper_text = 'Hello World!'
        self._lower_text = 'Goodbye'

    def switch(self):
        temp = self.upper_text
        self.upper_text = self.lower_text
        self.lower_text = temp
```

The view model is the data context of the frame. This means that the data will be stored here rather than on the user interface which makes it much easier to focus on the logic rather than the UI when working on experiments. Notice how the properties "upper_text" and "lower_text" are exposed: Both have a private member variable indicated by the "_" in front of them. In the setter method of both of them we have a call to "notify_change" which makes sure that the UI will update properly.
To connect this data context to the UI we go back to the first file and add the following at the top:

```python
from ViewModel.ToolViewModels.HelloWorldViewModel import HelloWorldViewModel
```

and this line to the constructor:

```python
self.data_context = HelloWorldViewModel()
```

The only thing left to do is to assign the properties from the data context to the widgets on the UI. To do this use the binding manager of the frame:

```python
self.bindings.set_binding('upper_text', upper_box, 'setText')
self.bindings.set_binding('lower_text', lower_box, 'setText')
switch_button.clicked.connect(self.data_context.switch)
```

The entire file should now look like this:

```python
from View.Controls.ToolFrames.EmptyFrame import EmptyFrame
from PyQt5.QtWidgets import QLineEdit, QPushButton, QGridLayout
from ViewModel.ToolViewModels.HelloWorldViewModel import HelloWorldViewModel


class HelloWorldFrame(EmptyFrame):

    name = 'Hello World Frame'

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.data_context = HelloWorldViewModel()
        self._layout = QGridLayout()
        self.setLayout(self._layout)
        self._setup_()

    def _setup_(self):
        upper_box = QLineEdit('Hello World')
        lower_box = QLineEdit('Goodbye')
        switch_button = QPushButton('switch')

        self.bindings.set_binding('upper_text', upper_box, 'setText')
        self.bindings.set_binding('lower_text', lower_box, 'setText')
        switch_button.clicked.connect(self.data_context.switch)

        self._layout.addWidget(upper_box, 0, 0)
        self._layout.addWidget(lower_box, 1, 0)
        self._layout.addWidget(switch_button, 1, 1)
```

If you now reload the tools, a click on the button should make the texts switch places. Congratulations, you've built your first tool frame!


go to:

https://git.ee.ethz.ch/dwindey/optomechanics/wikis/Lab-Master