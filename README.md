# Optomechanics Python Module

This Python module contains functions and drivers used in and around the 
particle trapping experiments in the photonics group at ETH. The module has the 
following structure:

 * Theory: Contains theory functions for calculations related to the trapping
   experiments.
 * Experiment: Contains device drivers and experiment control software that is
   to be used in the lab.
 * Post-Processing: Contains functions for post-processing recorded data (e.g.
   derivation of spectra, fitting, ...).
   
## How to add this to the python environmet

The location of the folder should be added to the pythonpath, which you can 
show using "import sys; print(sys.path)".
One way is to add the executing script at the beginning of the code using 
"sys.appened('/file/to/optomechanics/')".
Alternatively, it can be permenantly added by modifying the environment variable 
$PYTHONPATH. This is OS dependent and is sometimes annoyingly complicated.
Another way, is shown here: https://www.semicolonworld.com/question/43048/permanently-add-a-directory-to-pythonpath
Essentially:

1. You find the path where your python program is stored.

2. You go to Lib/site-packages (that's where the installed packages live).

3. You add a file which ends in .pth (name does not matter, eg. ' optomecahnics.pth')

4. In this file, you write the location of optomechanics (eg. C:\).
   
5. Now, if you execute "import sys; print(sys.path)", then it shows this location, too.


