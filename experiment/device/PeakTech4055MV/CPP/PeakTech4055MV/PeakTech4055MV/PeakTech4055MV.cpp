// PeakTech4055MV.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <vector>
#include <Windows.h>
#include "CH375DLL.H"

using namespace std;

// constants
const string cmd_exit = "exit";
const string cmd_connect = "connect";
const string cmd_write = "write";

// global variables
ULONG command_size = 1024;
ULONG connected_device_index = 0;
bool device_connected = false;


/*
Connect to a signal generator
*/
int connect(ULONG index) {
	device_connected = false;
	HANDLE hnd;
	hnd = CH375OpenDevice(index);
	if (hnd == INVALID_HANDLE_VALUE)
	{
		cout << "\r\nOpen CH372 failure\r\n";
		return false;
	}
	else
	{
		//setup CH372 timeout
		if (CH375SetTimeout(0, 0xFFFFFFFF, 4000) == FALSE)
		{
			cout << "\r\nConnection CH375 overtime\r\n";
			return false;
		}
		cout << "\r\nConnected device: " << CH375GetDeviceName(index) << "\r\n";
	}
	device_connected = true;
	connected_device_index = index;
	return true;
}

/*
Write command to signal generator
*/
void write(string command) {
	char *ptr;
	ptr = (char*)malloc(command_size);
	strcpy_s(ptr, command_size, command.c_str());
	strcat_s(ptr, command_size, "\n");
	CH375WriteData(0, ptr, &command_size);
	free(ptr);
}

void write2(string command) {
	char *ptr;
	ptr = (char*)malloc(1024);
	char const *str;
	str = command.c_str();
	strcpy(ptr, str);
	strcpy(ptr, str);
	strcat(ptr, "\n");
	ULONG len;
	len = strlen(ptr);
	CH375WriteData(0, ptr, &len);
	free(ptr);
}

/*
Split string into vector
*/
vector<string> split(const string& str, const string& delim)
{
	vector<string> tokens;
	size_t prev = 0, pos = 0;
	do
	{
		pos = str.find(delim, prev);
		if (pos == string::npos) pos = str.length();
		string token = str.substr(prev, pos - prev);
		if (!token.empty()) tokens.push_back(token);
		prev = pos + delim.length();
	} while (pos < str.length() && prev < str.length());
	return tokens;
}

int main()
{
	cout << "PeakTech Signal Generator 4055MV - Control Tool\r\n\r\n";
	cout << "------- Commands ----------\r\n";
	cout << "connect:[device index] -> connects to a device\r\n";
	cout << "write:[command] -> runs command on device(refer to programmer's manual)\r\n";
	cout << "exit -> disconnects device\r\n\r\n";
	bool run = true;

	// run until exit command is entered
	while (run) {
		cout << "> ";
		
		string command_line;
		getline(cin, command_line);		// read command line
		if (command_line == "" || command_line.find_first_not_of(" ") == -1) continue;	// check if command is valid

		vector<string> command_parts = split(command_line, ";");	// split command from its arguments
		string command = command_parts[0];

		if (command == cmd_connect) {
			if (command_parts.size() == 2) connect(stoi(command_parts[1]));
			else cout << "Wrong number of arguments. This method uses one argument (device index).\r\n";
		}

		if (device_connected) {
			if (command == cmd_write) {
				if (command_parts.size() == 2) write2(command_parts[1]);
				else cout << "Wrong number of arguments. This method uses one argument (command).\r\n";
			}
		}
		else {
			cout << "No device is currently connected.\r\n";
		}

		if (command == cmd_exit) {
			run = false;
			if (device_connected) CH375CloseDevice(connected_device_index);
		}
	}
	return 0;
}
