PeakTech Signal Generator 4055MV
********************************
The purpose of this folder is to provide the tools necessary to remotely control the PeakTech Signal Generators via USB. Sadly, the API is only available on Windows systems.

############################
Installation
############################
1. install the appropriate drivers from the drivers folder
2. Test the installation with "PeakTech4055MV.exe"
3. To use python include the file "PeakTech4055MW.py" in your script
	a) make sure that the class ProcessHandler is available (This class can be found in optomechanics/devices/ProcessHandler.py)
	b) make other locations available with the following code:
		import sys
		sys.path.append(parent_path_of_process_handler)