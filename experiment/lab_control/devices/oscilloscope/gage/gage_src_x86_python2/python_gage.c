/*
By Christophe V. Culan
*/

#include <Python.h>
#include <numpy/arrayobject.h>

#include <windows.h>
#include <stdio.h>


#define USE_LOCAL_TCHAR

#include "CsPrototypes.h"
#include "CsAppSupport.h"
#include "CsTchar.h"
#include "CsSdkMisc.h"


static	int32						gage_i32Status = CS_SUCCESS;
static	void*						gage_pBuffer = NULL;
static	float*						gage_pVBuffer = NULL;
static	int64*						gage_pTimeStamp = NULL;
static  double*						gage_pDoubleData = NULL;
static	uInt32						gage_u32Mode;
static	CSHANDLE					gage_hSystem = 0;
static	IN_PARAMS_TRANSFERDATA		gage_InData = {0};
static	OUT_PARAMS_TRANSFERDATA		gage_OutData = {0};
static	CSSYSTEMINFO				gage_CsSysInfo = {0};
static	CSAPPLICATIONDATA			gage_CsAppData = {0};
static	CSACQUISITIONCONFIG			gage_CsAcqCfg = {0};
static	CSCHANNELCONFIG				gage_CsChanCfg = {0};
static	CSTRIGGERCONFIG				gage_CsTrigCfg = {0};
static	uInt32						gage_u32ChannelIndexIncrement;

static  npy_intp					gage_dims[1] = {0};
static  npy_intp					gage_timestampdims[1] = {0};

int32 TransferTimeStamp(CSHANDLE gage_hSystem, uInt32 u32SegmentStart, uInt32 u32SegmentCount, void* pTimeStamp)
{
	IN_PARAMS_TRANSFERDATA		InTSData = {0};
	OUT_PARAMS_TRANSFERDATA		OutTSData = {0};
	int32						gage_i32Status = CS_SUCCESS;

	InTSData.u16Channel = 1;
	InTSData.u32Mode = TxMODE_TIMESTAMP;
	InTSData.i64StartAddress = 1;
	InTSData.i64Length = (int64)u32SegmentCount;
	InTSData.u32Segment = u32SegmentStart;

	ZeroMemory(pTimeStamp,(size_t)(u32SegmentCount * sizeof(int64)));
	InTSData.pDataBuffer = pTimeStamp;

	gage_i32Status = CsTransfer(gage_hSystem, &InTSData, &OutTSData);
	if (CS_FAILED(gage_i32Status))
	{
/*
		if the error is INVALID_TRANSFER_MODE it just means that this systems
		doesn't support time stamp. We can continue on with the program.
*/
		if (CS_INVALID_TRANSFER_MODE == gage_i32Status)
			_ftprintf (stderr, _T("Time stamp is not supported in this system.\n"));
		else
			DisplayErrorString(gage_i32Status);

		VirtualFree(pTimeStamp, 0, MEM_RELEASE);
		pTimeStamp = NULL;
		return (gage_i32Status);
	}
/*
	Tick count is stored in OutTSData.i32LowPart.
*/
	return OutTSData.i32LowPart;
}

static PyObject * gage_init(PyObject * self, PyObject * args)
{
    uInt32 gage_u32BoardType = 0;
    uInt32 gage_u32Channels = 0;
    uInt32 gage_u32SampleBits = 0;
    int16 gage_i16Index = 0;

	if (!PyArg_ParseTuple(args, "|kkkI", &gage_u32BoardType,
	                      &gage_u32Channels, &gage_u32SampleBits,
	                      &gage_i16Index))
			return NULL;
			
	/*
	Initializes the CompuScope boards found in the system. If the
	system is not found a message with the error code will appear.
	Otherwise gage_i32Status will contain the number of systems found.
	*/
	gage_i32Status = CsInitialize();

	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		return (-1);
	}
	
	/*
	Get System. This sample program only supports one system. If
	2 systems or more are found, the first system that is found
	will be the system that will be used. gage_hSystem will hold a unique
	system identifier that is used when referencing the system.
	*/
	gage_i32Status = CsGetSystem(&gage_hSystem, gage_u32BoardType,
	                             gage_u32Channels, gage_u32SampleBits,
	                             gage_i16Index);
	_ftprintf(stdout, _T("Handle: %d\n"), gage_hSystem);
	
	/*
	Get System information. The u32Size field must be filled in
	 prior to calling CsGetSystemInfo
	*/
	gage_CsSysInfo.u32Size = sizeof(CSSYSTEMINFO);
	gage_i32Status = CsGetSystemInfo(gage_hSystem, &gage_CsSysInfo);

	/*
	Display the system name from the driver
	*/
	_ftprintf(stdout, _T("Board Name: %s\n"), gage_CsSysInfo.strBoardName);
	
	return PyInt_FromLong(gage_hSystem);
}

static PyObject * gage_system_info(PyObject * self, PyObject * args)
{
    if (!PyArg_ParseTuple(args, ""))
			return NULL;

    gage_CsSysInfo.u32Size = sizeof(CSSYSTEMINFO);
	gage_i32Status = CsGetSystemInfo(gage_hSystem, &gage_CsSysInfo);

	return Py_BuildValue("{skslskslsLsksssksksksksk}",
	                     "SampleBits", gage_CsSysInfo.u32SampleBits,
	                     "SampleResolution",
	                     gage_CsSysInfo.i32SampleResolution,
	                     "SampleSize", gage_CsSysInfo.u32SampleSize,
	                     "SampleOffset", gage_CsSysInfo.i32SampleOffset,
	                     "MaxMemory", gage_CsSysInfo.i64MaxMemory,
	                     "BoardType", gage_CsSysInfo.u32BoardType,
	                     "BoardName", gage_CsSysInfo.strBoardName,
	                     "AddonOptions", gage_CsSysInfo.u32AddonOptions,
	                     "BaseBoardOptions",
	                     gage_CsSysInfo.u32BaseBoardOptions,
	                     "BoardCount", gage_CsSysInfo.u32BoardCount,
	                     "ChannelCount", gage_CsSysInfo.u32ChannelCount,
	                     "TriggerMachineCount",
	                     gage_CsSysInfo.u32TriggerMachineCount);
}

static PyObject * gage_get_configuration(PyObject * self,
                                         PyObject * args)
{
    int32 gage_nIndex;
    int32 gage_nConfig = CS_CURRENT_CONFIGURATION;
    uInt32 gage_number = 1;

    if (!PyArg_ParseTuple(args, "l|lk", &gage_nIndex,
	                      &gage_nConfig, &gage_number))
			return NULL;

    if (gage_nIndex == CS_ACQUISITION)
    {
        gage_CsAcqCfg.u32Size = sizeof(CSACQUISITIONCONFIG);
        gage_i32Status = CsGet(gage_hSystem, CS_ACQUISITION, gage_nConfig, &gage_CsAcqCfg);
        if (CS_FAILED(gage_i32Status))
        {
            DisplayErrorString(gage_i32Status);
            CsFreeSystem(gage_hSystem);
            return (-1);
        }

        return Py_BuildValue("{sLskskskskslsksksLsLsksLsLslsk}",
                             "SampleRate", gage_CsAcqCfg.i64SampleRate,
                             "ExtClk", gage_CsAcqCfg.u32ExtClk,
                             "ExtClkSampleSkip",
                             gage_CsAcqCfg.u32ExtClkSampleSkip,
                             "Mode", gage_CsAcqCfg.u32Mode,
                             "SampleBits", gage_CsAcqCfg.u32SampleBits,
                             "SampleRes",
                             gage_CsAcqCfg.i32SampleRes,
                             "SampleSize", gage_CsAcqCfg.u32SampleSize,
                             "SegmentCount",
                             gage_CsAcqCfg.u32SegmentCount,
                             "Depth", gage_CsAcqCfg.i64Depth,
                             "TriggerTimeout",
                             gage_CsAcqCfg.i64TriggerTimeout,
                             "TrigEnginesEn",
                             gage_CsAcqCfg.u32TrigEnginesEn,
                             "TriggerDelay",
                             gage_CsAcqCfg.i64TriggerDelay,
                             "TriggerHoldoff",
                             gage_CsAcqCfg.i64TriggerHoldoff,
                             "SampleOffset",
                             gage_CsAcqCfg.i32SampleOffset,
                             "TimeStampConfig",
                             gage_CsAcqCfg.u32TimeStampConfig);
    } else if (gage_nIndex == CS_CHANNEL) {
        gage_CsChanCfg.u32Size = sizeof(CSCHANNELCONFIG);
        gage_CsChanCfg.u32ChannelIndex = gage_number;
        CsGet(gage_hSystem, CS_CHANNEL, gage_nConfig, &gage_CsChanCfg);
        if (CS_FAILED(gage_i32Status))
        {
            DisplayErrorString(gage_i32Status);
            CsFreeSystem(gage_hSystem);
            return (-1);
        }

        return Py_BuildValue("{sksksksksl}",
                             "Term", gage_CsChanCfg.u32Term,
                             "InputRange", gage_CsChanCfg.u32InputRange,
                             "Impedance", gage_CsChanCfg.u32Impedance,
                             "Filter", gage_CsChanCfg.u32Filter,
                             "DcOffset", gage_CsChanCfg.i32DcOffset);
    } else if (gage_nIndex == CS_TRIGGER) {
        gage_CsTrigCfg.u32Size = sizeof(CSTRIGGERCONFIG);
        gage_CsTrigCfg.u32TriggerIndex = gage_number;
        CsGet(gage_hSystem, CS_TRIGGER, gage_nConfig, &gage_CsTrigCfg);
        if (CS_FAILED(gage_i32Status))
        {
            DisplayErrorString(gage_i32Status);
            CsFreeSystem(gage_hSystem);
            return (-1);
        }

        return Py_BuildValue("{skslslsksksksk}",
                             "Condition", gage_CsTrigCfg.u32Condition,
                             "Level", gage_CsTrigCfg.i32Level,
                             "Source", gage_CsTrigCfg.i32Source,
                             "ExtCoupling",
                             gage_CsTrigCfg.u32ExtCoupling,
                             "ExtTriggerRange",
                             gage_CsTrigCfg.u32ExtTriggerRange,
                             "ExtImpedance",
                             gage_CsTrigCfg.u32ExtImpedance,
                             "Relation", gage_CsTrigCfg.u32Relation);
    } else {
        return PyInt_FromLong(-1);
    }
}

static PyObject * gage_config(PyObject * self, PyObject * args)
{	
	TCHAR * szIniFile;
	
	if (!PyArg_ParseTuple(args, "z",&szIniFile))
		return NULL;
	
	gage_i32Status = CsAs_ConfigureSystem(gage_hSystem, (int)gage_CsSysInfo.u32ChannelCount, 1, (LPCTSTR) szIniFile, &gage_u32Mode);
	
	if (CS_FAILED(gage_i32Status))
	{
		if (CS_INVALID_FILENAME == gage_i32Status)
		{
		/*
			Display message but continue on using defaults.
		*/
			_ftprintf(stdout, _T("Cannot find %s - using default parameters.\n"), szIniFile);
		}
		else
		{	
			/*
			Otherwise the call failed.  If the call did fail we should free the CompuScope
			system so it's available for another application
			*/
			DisplayErrorString(gage_i32Status);
			CsFreeSystem(gage_hSystem);
			return(-1);
		}
	}
	/*
	If the return value is greater than  1, then either the application, 
	acquisition, some of the Channel and / or some of the Trigger sections
	were missing from the ini file and the default parameters were used. 
	*/
	if (CS_USING_DEFAULT_ACQ_DATA & gage_i32Status)
		_ftprintf(stdout, _T("No ini entry for acquisition. Using defaults.\n"));

	if (CS_USING_DEFAULT_CHANNEL_DATA & gage_i32Status)
		_ftprintf(stdout, _T("No ini entry for one or more Channels. Using defaults for missing items.\n"));

	if (CS_USING_DEFAULT_TRIGGER_DATA & gage_i32Status)
		_ftprintf(stdout, _T("No ini entry for one or more Triggers. Using defaults for missing items.\n"));

	gage_i32Status = CsAs_LoadConfiguration(gage_hSystem, szIniFile, APPLICATION_DATA, &gage_CsAppData);

	if (CS_FAILED(gage_i32Status))
	{
		if (CS_INVALID_FILENAME == gage_i32Status)
			_ftprintf(stdout, _T("Using default application parameters.\n"));
		else
		{
			DisplayErrorString(gage_i32Status);
			CsFreeSystem(gage_hSystem);
			return (-1);
		}
	}
	else if (CS_USING_DEFAULT_APP_DATA & gage_i32Status)
	{
		/*
		If the return value is CS_USING_DEFAULT_APP_DATA (defined in ConfigSystem.h) 
		then there was no entry in the ini file for Application and we will use
		the application default values, which were set in CsAs_LoadConfiguration.
		*/
		_ftprintf(stdout, _T("No ini entry for application data. Using defaults\n"));
	}
	/*
	Send the acqcuisition, channel and trigger parameters that we've read
	from the ini file to the hardware.
	*/
	/*
	Commit the values to the driver.  This is where the values get sent to the
	hardware.  Any invalid parameters will be caught here and an error returned.
	*/
	gage_i32Status = CsDo(gage_hSystem, ACTION_COMMIT);
	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		CsFreeSystem(gage_hSystem);
		return (-1);
	}
	/*
	Get the current sample size, resolution and offset parameters from the driver
	by calling CsGet for the ACQUISTIONCONFIG structure. These values are used
	when saving the file.
	*/
	gage_CsAcqCfg.u32Size = sizeof(gage_CsAcqCfg);
	gage_i32Status = CsGet(gage_hSystem, CS_ACQUISITION, CS_ACQUISITION_CONFIGURATION, &gage_CsAcqCfg);
	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		CsFreeSystem(gage_hSystem);
		return (-1);
	}	
	
	return PyInt_FromLong(0);
}

static PyObject * gage_acquire(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
	/*
	Start the data Acquisition
	*/
	gage_i32Status = CsDo(gage_hSystem, ACTION_START);
	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		CsFreeSystem(gage_hSystem);
		return (-1);
	}
	
	/*
	Free any buffers that have been allocated
	*/

	if( NULL != gage_pTimeStamp)
	{
		VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
		gage_pTimeStamp = NULL;
	}
	
	if( NULL != gage_pDoubleData )
	{
		VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
		gage_pDoubleData = NULL;
	}

	if ( NULL != gage_pVBuffer )
	{
		VirtualFree(gage_pVBuffer, 0, MEM_RELEASE);
		gage_pVBuffer = NULL;
	}

	if ( NULL != gage_pBuffer)
	{
		VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		gage_pBuffer = NULL;
	}

	_ftprintf(stdout,_T("Freed memory.\n"));
	
	/*
	We need to allocate a buffer
	for transferring the data
	*/
	gage_pBuffer  = VirtualAlloc(NULL, (size_t)(gage_CsAppData.i64TransferLength * gage_CsAcqCfg.u32SampleSize), MEM_COMMIT, PAGE_READWRITE);

	if (NULL == gage_pBuffer)
	{
		_ftprintf (stderr, _T("Unable to allocate memory\n"));
		CsFreeSystem(gage_hSystem);
		return PyErr_NoMemory();
	}
	/*
	We also need to allocate a buffer for transferring the timestamp
	*/
	gage_pTimeStamp = (int64 *)VirtualAlloc(NULL, (size_t)(gage_CsAppData.u32TransferSegmentCount * sizeof(int64)), MEM_COMMIT, PAGE_READWRITE);
	if (NULL == gage_pTimeStamp)
	{
		_ftprintf (stderr, _T("Unable to allocate memory\n"));
		if (NULL != gage_pBuffer)
			VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		return PyErr_NoMemory();
	}
	if (TYPE_FLOAT == gage_CsAppData.i32SaveFormat)
	{
		/*
		Allocate another buffer to pass the data that is going to be converted
		into voltages
		*/
		gage_pVBuffer  = (float *)VirtualAlloc(NULL, (size_t)(gage_CsAppData.i64TransferLength * sizeof(float)), MEM_COMMIT, PAGE_READWRITE);
		if (NULL == gage_pVBuffer)
		{
			_ftprintf (stderr, _T("Unable to allocate memory\n"));
			CsFreeSystem(gage_hSystem);
			if (NULL != gage_pBuffer)
				VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
			if (NULL != gage_pTimeStamp)
				VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
			return PyErr_NoMemory();
		}
	}
	
	ZeroMemory(gage_pBuffer,(size_t)(gage_CsAppData.i64TransferLength * gage_CsAcqCfg.u32SampleSize));
	
	_ftprintf(stdout, _T("Allocated memory for data transfer.\n"));
	
	/*
	Now prepare to transfer the actual acquired data for each desired multiple group.
	Fill in the gage_InData structure for transferring the data
	*/
	gage_InData.u32Mode = TxMODE_DEFAULT;
	gage_InData.i64StartAddress = gage_CsAppData.i64TransferStartPosition;
	gage_InData.i64Length =  gage_CsAppData.i64TransferLength;
	gage_InData.pDataBuffer = gage_pBuffer;
	gage_u32ChannelIndexIncrement = CsAs_CalculateChannelIndexIncrement(&gage_CsAcqCfg, &gage_CsSysInfo );
	
	return PyInt_FromLong(0);
}

static PyObject * gage_wait(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args,""))
		return NULL;
		
	Py_BEGIN_ALLOW_THREADS
	/*
	DataCaptureComplete queries the system to see when
	the capture is complete
	*/
	if (!DataCaptureComplete(gage_hSystem))
	{
		CsFreeSystem(gage_hSystem);
		return (-1);
	}
	/*
	Acquisition is now complete.
	*/
	Py_END_ALLOW_THREADS

	_ftprintf(stdout, _T("Acquisition completed.\n"));
	
	return PyInt_FromLong(0); 
}

static PyObject * gage_freememory(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args,""))
		return NULL;
		
	if( NULL != gage_pTimeStamp)
	{
		VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
		gage_pTimeStamp = NULL;
	}
	
	if( NULL != gage_pDoubleData )
	{
		VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
		gage_pDoubleData = NULL;
	}

	if ( NULL != gage_pVBuffer )
	{
		VirtualFree(gage_pVBuffer, 0, MEM_RELEASE);
		gage_pVBuffer = NULL;
	}

	if ( NULL != gage_pBuffer)
	{
		VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		gage_pBuffer = NULL;
	}
	
	_ftprintf(stdout, _T("Freed memory.\n"));
	
	return PyInt_FromLong(0);
}

static PyObject * gage_timestamp(PyObject * self, PyObject * args)
{
	uInt32 u32Count;
	int32 i32TickFrequency;
	PyObject * ret; 
	
	if (!PyArg_ParseTuple(args,""))
		return NULL;
	
	i32TickFrequency = 0;
	
	/*
	Free gage_pDoubleData buffer if required
	*/
	if( NULL != gage_pDoubleData )
	{
		VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
		gage_pDoubleData = NULL;
	}

	gage_pDoubleData = (double *)VirtualAlloc(NULL, (size_t)(gage_CsAppData.u32TransferSegmentCount * sizeof(double)), MEM_COMMIT, PAGE_READWRITE);
	
	if (NULL == gage_pDoubleData)
		return PyErr_NoMemory();
		
	/*
	Call the function TransferTimeStamp. This function is used to transfer the timestamp
	data. The i32TickFrequency, which is returned from this fuction, is the clock rate
	of the counter used to acquire the timestamp data.
	*/
	i32TickFrequency = TransferTimeStamp(gage_hSystem, gage_CsAppData.u32TransferSegmentStart, gage_CsAppData.u32TransferSegmentCount, gage_pTimeStamp);

	/*
	If TransferTimeStamp fails, i32TickFrequency will be negative,
	which represents an error code. If there is an error we'll set
	the time stamp info in gage_pDoubleData to 0.
	*/
	ZeroMemory(gage_pDoubleData, gage_CsAppData.u32TransferSegmentCount * sizeof(double));
	if (CS_SUCCEEDED(i32TickFrequency))
	{
	/*
		Allocate a buffer of doubles to store the the timestamp data after we have
		converted it to microseconds.
	*/
		for (u32Count = 0; u32Count < gage_CsAppData.u32TransferSegmentCount; u32Count++)
		{
		/*
			The number of ticks that have ocurred / tick count(the number of ticks / second)
			= the number of seconds elapsed. Multiple by 1000000 to get the number of
			mircoseconds
		*/
			gage_pDoubleData[u32Count] = (double)(* (gage_pTimeStamp + u32Count)) * 1.e6 / (double)(i32TickFrequency);
		}
	}
	
	gage_timestampdims[0] = gage_CsAppData.u32TransferSegmentCount;
	ret = PyArray_SimpleNewFromData(1, gage_timestampdims, NPY_DOUBLE, gage_pDoubleData);
	
	return ret;
}

static PyObject * gage_get(PyObject * self, PyObject * args)
{
	int nSegment = 1;
	uInt32 u32ChannelNumber;
	int64 i64Depth;
	
	PyObject * ret;
	
	if (!PyArg_ParseTuple(args, "i|i", &u32ChannelNumber, &nSegment))
		return NULL;
	
	gage_InData.u16Channel = (uInt16)u32ChannelNumber;
	
	/*
	Transfer the captured data
	*/
	gage_InData.u32Segment = (uInt32)nSegment;
	gage_i32Status = CsTransfer(gage_hSystem, &gage_InData, &gage_OutData);
	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		if (NULL != gage_pBuffer)
			VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		if (NULL != gage_pVBuffer)
			VirtualFree(gage_pVBuffer, 0, MEM_RELEASE);
		if (NULL != gage_pTimeStamp)
			VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
		CsFreeSystem(gage_hSystem);
		return (-1);
	}
	
	gage_CsChanCfg.u32Size = sizeof(CSCHANNELCONFIG);
	gage_CsChanCfg.u32ChannelIndex = u32ChannelNumber;
	CsGet(gage_hSystem, CS_CHANNEL, CS_ACQUISITION_CONFIGURATION, &gage_CsChanCfg);
	
	/*
	Call the ConvertToVolts function. This function will convert the raw
	data to voltages. We pass the actual length, which will be converted
	from 0 to actual length.  Any invalid samples at the beginning are
	handled in the SaveFile routine.
	*/
	i64Depth = gage_OutData.i64ActualLength;

	gage_i32Status = CsAs_ConvertToVolts(i64Depth, gage_CsChanCfg.u32InputRange, gage_CsAcqCfg.u32SampleSize,
									gage_CsAcqCfg.i32SampleOffset, gage_CsAcqCfg.i32SampleRes, 
									gage_CsChanCfg.i32DcOffset, gage_pBuffer, gage_pVBuffer);
	if (CS_FAILED(gage_i32Status))
	{
		DisplayErrorString(gage_i32Status);
		return (-1);
	}
	
	gage_dims[0] = i64Depth;
	ret = PyArray_SimpleNewFromData(1, gage_dims, NPY_FLOAT, gage_pVBuffer);
	
	return ret;
}

static PyObject * gage_save(PyObject * self, PyObject * args)
{
	TCHAR szFileName[MAX_PATH];
	TCHAR szFormatString[MAX_PATH];
	int nMaxSegmentNumber;
	int nMaxChannelNumber;
	uInt32 u32Count;
	uInt32 u32ChannelNumber;
	int64 i64Depth;
	int32 i32TickFrequency;
	FileHeaderStruct stHeader = {0};
	
	if (!PyArg_ParseTuple(args, "s", szFileName))
		return NULL; 
	
	i32TickFrequency = 0;
	
	/*	
	format a string with the number of segments  and channels so all filenames will have
	the same number of characters.
	*/
	_stprintf(szFormatString, _T("%d"), gage_CsAppData.u32TransferSegmentStart + gage_CsAppData.u32TransferSegmentCount - 1);
	nMaxSegmentNumber = (int)_tcslen(szFormatString);
	
	_stprintf(szFormatString, _T("%d"), gage_CsSysInfo.u32ChannelCount);
	nMaxChannelNumber = (int)_tcslen(szFormatString);
	
	_stprintf(szFormatString, _T("%%s_CH%%0%dd-%%0%dd.dat"), nMaxChannelNumber, nMaxSegmentNumber);
	
	/*
	Call the function TransferTimeStamp. This function is used to transfer the timestamp
	data. The i32TickFrequency, which is returned from this fuction, is the clock rate
	of the counter used to acquire the timestamp data.
	*/
	i32TickFrequency = TransferTimeStamp(gage_hSystem, gage_CsAppData.u32TransferSegmentStart, gage_CsAppData.u32TransferSegmentCount, gage_pTimeStamp);

	/*
	If TransferTimeStamp fails, i32TickFrequency will be negative,
	which represents an error code. If there is an error we'll set
	the time stamp info in gage_pDoubleData to 0.
	*/
	
	/*
	Frees gage_pDoubleData if needed
	*/
	if( NULL != gage_pDoubleData )
	{
		VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
		gage_pDoubleData = NULL;
	}
	
	gage_pDoubleData = (double *)VirtualAlloc(NULL, (size_t)(gage_CsAppData.u32TransferSegmentCount * sizeof(double)), MEM_COMMIT, PAGE_READWRITE);
	
	ZeroMemory(gage_pDoubleData, gage_CsAppData.u32TransferSegmentCount * sizeof(double));
	
	if (NULL == gage_pDoubleData)
		return PyErr_NoMemory();
		
	if (CS_SUCCEEDED(i32TickFrequency))
	{
	/*
		Allocate a buffer of doubles to store the the timestamp data after we have
		converted it to microseconds.
	*/
		for (u32Count = 0; u32Count < gage_CsAppData.u32TransferSegmentCount; u32Count++)
		{
		/*
			The number of ticks that have ocurred / tick count(the number of ticks / second)
			= the number of seconds elapsed. Multiple by 1000000 to get the number of
			mircoseconds
		*/
			gage_pDoubleData[u32Count] = (double)(* (gage_pTimeStamp + u32Count)) * 1.e6 / (double)(i32TickFrequency);
		}
	}
	
	for	(u32ChannelNumber = 1; u32ChannelNumber <= gage_CsSysInfo.u32ChannelCount; u32ChannelNumber += gage_u32ChannelIndexIncrement)
	{
		int nMulRecGroup;
		int nTimeStampIndex;
		/*
		Variable that will contain either raw data or data in Volts depending on requested format
		*/
		void* pSrcBuffer = NULL;

		ZeroMemory(gage_pBuffer,(size_t)(gage_CsAppData.i64TransferLength * gage_CsAcqCfg.u32SampleSize));
		gage_InData.u16Channel = (uInt16)u32ChannelNumber;

		/*
		This for loop transfers each multiple record segment to a seperate file. It also
		writes the time stamp information for the segment to the header of the file. Note
		that the timestamp array (gage_pDoubleData) starts at index 0, even if the starting transfer
		segment is not 0. Note that the user is responsible for ensuring that the ini file
		has valid values and the segments that are being tranferred have been captured.
		*/
		for (nMulRecGroup = gage_CsAppData.u32TransferSegmentStart, nTimeStampIndex = 0; nMulRecGroup < (int)(gage_CsAppData.u32TransferSegmentStart + gage_CsAppData.u32TransferSegmentCount);
									nMulRecGroup++, nTimeStampIndex++)
		{
			/*
			Transfer the captured data
			*/
			gage_InData.u32Segment = (uInt32)nMulRecGroup;
			gage_i32Status = CsTransfer(gage_hSystem, &gage_InData, &gage_OutData);
			if (CS_FAILED(gage_i32Status))
			{
				DisplayErrorString(gage_i32Status);
				if (NULL != gage_pBuffer)
					VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
				if (NULL != gage_pVBuffer)
					VirtualFree(gage_pVBuffer, 0, MEM_RELEASE);
				if (NULL != gage_pTimeStamp)
					VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
				if (NULL != gage_pDoubleData)
					VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
				CsFreeSystem(gage_hSystem);
				return (-1);
			}
		/*
		Note: to optimize the transfer loop, everything from
		this point on in the loop could be moved out and done
		after all the channels are transferred.
		*/

			/*
			Assign a file name for each channel that we want to save
			*/
			_stprintf(szFileName, szFormatString, gage_CsAppData.lpszSaveFileName, u32ChannelNumber, nMulRecGroup);

			/*
			Gather up the information needed for the volt conversion and/or file header
			*/
			gage_CsChanCfg.u32Size = sizeof(CSCHANNELCONFIG);
			gage_CsChanCfg.u32ChannelIndex = u32ChannelNumber;
			CsGet(gage_hSystem, CS_CHANNEL, CS_ACQUISITION_CONFIGURATION, &gage_CsChanCfg);

			if (TYPE_FLOAT == gage_CsAppData.i32SaveFormat)
			{
				/*
				Call the ConvertToVolts function. This function will convert the raw
				data to voltages. We pass the actual length, which will be converted
				from 0 to actual length.  Any invalid samples at the beginning are
				handled in the SaveFile routine.
				*/
				i64Depth = gage_OutData.i64ActualLength;

				gage_i32Status = CsAs_ConvertToVolts(i64Depth, gage_CsChanCfg.u32InputRange, gage_CsAcqCfg.u32SampleSize,
												gage_CsAcqCfg.i32SampleOffset, gage_CsAcqCfg.i32SampleRes, 
												gage_CsChanCfg.i32DcOffset, gage_pBuffer, gage_pVBuffer);
				if (CS_FAILED(gage_i32Status))
				{
					DisplayErrorString(gage_i32Status);
					continue;
				}

				pSrcBuffer = gage_pVBuffer;
			}
			else
			{
				pSrcBuffer = gage_pBuffer;
			}
			/*
			The driver may have had to change the start address and length
			due to alignment issues, so we'll get the actual start and length
			from the driver.
			*/

			stHeader.i64Start = gage_OutData.i64ActualStart;
			stHeader.i64Length = gage_OutData.i64ActualLength;
			stHeader.u32SampleSize = gage_CsAcqCfg.u32SampleSize;
			stHeader.i32SampleRes = gage_CsAcqCfg.i32SampleRes;
			stHeader.i32SampleOffset = gage_CsAcqCfg.i32SampleOffset;
			stHeader.u32InputRange = gage_CsChanCfg.u32InputRange;
			stHeader.i32DcOffset = gage_CsChanCfg.i32DcOffset;
			stHeader.u32SegmentCount = gage_CsAcqCfg.u32SegmentCount;
			stHeader.u32SegmentNumber = gage_InData.u32Segment;
			stHeader.dTimeStamp = gage_pDoubleData[nTimeStampIndex];

			gage_i32Status = (int32)CsAs_SaveFile(szFileName, pSrcBuffer, gage_CsAppData.i32SaveFormat, &stHeader);
			if ( 0 > gage_i32Status )
			{
				if (CS_MISC_ERROR == gage_i32Status)
				{
					_ftprintf(stderr, CsAs_GetLastFileError());
					_ftprintf(stderr, _T("\n"));
				}
				else
				{
					DisplayErrorString(gage_i32Status);
				}
				continue;
			}
		}
	}
	
	_ftprintf(stdout, _T("All channels are saved as ASCII data files in the current working directory.\n"));
	
	return PyInt_FromLong(0);
}

static PyObject * gage_close(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;
		
	/*
	Free the CompuScope system and any resources it's been using
	*/
	gage_i32Status = CsFreeSystem(gage_hSystem);
	
	_ftprintf(stdout,_T("Closed communication with the card\n"));
	
	/*
	Free any buffers that have been allocated
	*/

	if( NULL != gage_pTimeStamp)
	{
		VirtualFree(gage_pTimeStamp, 0, MEM_RELEASE);
		gage_pTimeStamp = NULL;
	}
	
	if( NULL != gage_pDoubleData )
	{
		VirtualFree(gage_pDoubleData, 0, MEM_RELEASE);
		gage_pDoubleData = NULL;
	}

	if ( NULL != gage_pVBuffer )
	{
		VirtualFree(gage_pVBuffer, 0, MEM_RELEASE);
		gage_pVBuffer = NULL;
	}

	if ( NULL != gage_pBuffer)
	{
		VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		gage_pBuffer = NULL;
	}

	_ftprintf(stdout,_T("Freed memory.\n"));
	
	return PyInt_FromLong(0);
}

static char gage_init_docstring[] =
	"Initialize communication with the gage driver.";
	
static char gage_config_docstring[] =
	"Pass the configuration to the gage driver.";
	
static char gage_acquire_docstring[] =
	"Launches the acquisition.";
	
static char gage_wait_docstring[] =
	"Wait for the the acquisition to finish.";

static char gage_freememory_docstring[] =
	"Frees allocated memory ressources.";
	
static char gage_timestamp_docstring[] =
	"Gets the timestamp data.";
	
static char gage_get_docstring[] =
	"Gets the data corresponding to given segment and channel.";
	
static char gage_save_docstring[] =
	"Saves the data corresponding to each channel and segment in separated ASCII data files at the specified location";

static char gage_close_docstring[] =
	"Close communication with the card and free allocated memory ressources.";

static char gage_system_info_docstring[] =
	"Get system info from the gage driver.";

static char gage_get_configuration_docstring[] =
	"Get configuration from the gage driver.";

static PyMethodDef module_methods[] = 
{
    {"init", gage_init, METH_VARARGS, gage_init_docstring},
	{"config", gage_config, METH_VARARGS, gage_config_docstring},
	{"acquire", gage_acquire, METH_VARARGS, gage_acquire_docstring},
	{"wait", gage_wait, METH_VARARGS, gage_wait_docstring},
	{"freememory", gage_freememory, METH_VARARGS, gage_freememory_docstring},
	{"timestamp", gage_timestamp, METH_VARARGS, gage_timestamp},
	{"get", gage_get, METH_VARARGS, gage_get_docstring},
	{"save", gage_save, METH_VARARGS, gage_save_docstring},
	{"close", gage_close, METH_VARARGS, gage_close_docstring},
	{"system_info", gage_system_info, METH_VARARGS, gage_system_info_docstring},
	{"get_configuration", gage_get_configuration, METH_VARARGS, gage_get_configuration_docstring},
    {NULL, NULL, 0, NULL}
};

static char module_docstring[] =
    "Python API for Gage Compuscope. Based on the C SDK of Compuscope software, used in combination with numpy.";

PyMODINIT_FUNC init_gage(void)
{
    PyObject *m = Py_InitModule3("_gage", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
};