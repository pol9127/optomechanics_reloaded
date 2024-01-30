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
static	uInt32						gage_u32Mode;
static	CSHANDLE					gage_hSystem = 0;
static	IN_PARAMS_TRANSFERDATA		gage_InData = {0};
static	OUT_PARAMS_TRANSFERDATA		gage_OutData = {0};
static	CSSYSTEMINFO				gage_CsSysInfo = {0};
static	CSAPPLICATIONDATA			gage_CsAppData = {0};
static	CSACQUISITIONCONFIG			gage_CsAcqCfg = {0};
static	CSCHANNELCONFIG				gage_CsChanCfg = {0};
static	CSTRIGGERCONFIG				gage_CsTrigCfg = {0};
static	CSACQUISITIONCONFIG			gage_CsAcqCfg_set = {0};
static	CSTRIGGERCONFIG				gage_CsTrigCfg_set = {0};
static	CSCHANNELCONFIG				gage_CsChanCfg_set = {0};
static	uInt32						gage_u32ChannelIndexIncrement;

static  npy_intp					gage_dims[1] = {0};
static  npy_intp					gage_timestampdims[1] = {0};

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
	
	return PyLong_FromLong(gage_hSystem);
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

        return Py_BuildValue("{sLskskskskslsksksLsLsksLsLslsksL}",
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
                             "DepthPostTrigger", gage_CsAcqCfg.i64Depth,
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
                             gage_CsAcqCfg.u32TimeStampConfig,
                             "DepthPreTrigger",
                             gage_CsAcqCfg.i64SegmentSize - gage_CsAcqCfg.i64Depth);
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
        return PyLong_FromLong(-1);
    }
}

static PyObject * gage_set_configuration(PyObject * self,
                                         PyObject * args)
{
    PyObject * gage_configuration = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &gage_configuration))
			return NULL;

	PyObject * gage_configuration_acqu = PyDict_GetItem(gage_configuration, PyUnicode_FromString("Acquisition"));
    PyObject * gage_configuration_trigger = PyDict_GetItem(gage_configuration, PyUnicode_FromString("Trigger"));
	PyObject * gage_configuration_channel = PyDict_GetItem(gage_configuration, PyUnicode_FromString("Channel"));
	PyObject * gage_configuration_app = PyDict_GetItem(gage_configuration, PyUnicode_FromString("Application"));

    gage_CsAcqCfg_set.u32Size = sizeof(CSACQUISITIONCONFIG);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SampleRate")), "L", &gage_CsAcqCfg_set.i64SampleRate);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("ExtClk")), "k", &gage_CsAcqCfg_set.u32ExtClk);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("ExtClkSampleSkip")), "k", &gage_CsAcqCfg_set.u32ExtClkSampleSkip);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("Mode")), "k", &gage_CsAcqCfg_set.u32Mode);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SampleBits")), "k", &gage_CsAcqCfg_set.u32SampleBits);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SampleRes")), "l", &gage_CsAcqCfg_set.i32SampleRes);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SampleSize")), "k", &gage_CsAcqCfg_set.u32SampleSize);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SegmentCount")), "k", &gage_CsAcqCfg_set.u32SegmentCount);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("DepthPostTrigger")), "L", &gage_CsAcqCfg_set.i64Depth);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("TriggerTimeout")), "L", &gage_CsAcqCfg_set.i64TriggerTimeout);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("TrigEnginesEn")), "k", &gage_CsAcqCfg_set.u32TrigEnginesEn);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("TriggerDelay")), "L", &gage_CsAcqCfg_set.i64TriggerDelay);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("TriggerHoldoff")), "L", &gage_CsAcqCfg_set.i64TriggerHoldoff);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("SampleOffset")), "l", &gage_CsAcqCfg_set.i32SampleOffset);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("TimeStampConfig")), "k", &gage_CsAcqCfg_set.u32TimeStampConfig);
    PyArg_Parse(PyDict_GetItem(gage_configuration_acqu, PyUnicode_FromString("DepthPreTrigger")), "L", &gage_CsAcqCfg_set.i64SegmentSize);
    gage_CsAppData.i64TransferStartPosition = -1 * gage_CsAcqCfg_set.i64SegmentSize;
    gage_CsAcqCfg_set.i64SegmentSize = gage_CsAcqCfg_set.i64SegmentSize + gage_CsAcqCfg_set.i64Depth;
    gage_CsAppData.i64TransferLength = gage_CsAcqCfg_set.i64SegmentSize;

//	_ftprintf(stdout, _T("TransferLength: %d\n"), gage_CsAppData.i64TransferLength);
//	 _ftprintf(stdout, _T("StartPosition: %d\n"), gage_CsAppData.i64TransferStartPosition);

//    _ftprintf(stdout, _T("SampleRate: %d\n"), gage_CsAcqCfg_set.i64SampleRate);
//    _ftprintf(stdout, _T("ExtClk: %d\n"), gage_CsAcqCfg_set.u32ExtClk);
//    _ftprintf(stdout, _T("ExtClkSampleSkip: %d\n"), gage_CsAcqCfg_set.u32ExtClkSampleSkip);
//    _ftprintf(stdout, _T("Mode: %d\n"), gage_CsAcqCfg_set.u32Mode);
//    _ftprintf(stdout, _T("SampleBits: %d\n"), gage_CsAcqCfg_set.u32SampleBits);
//    _ftprintf(stdout, _T("SampleRes: %d\n"), gage_CsAcqCfg_set.i32SampleRes);
//    _ftprintf(stdout, _T("SegmentCount: %d\n"), gage_CsAcqCfg_set.u32SegmentCount);
//    _ftprintf(stdout, _T("DepthPostTrigger: %d\n"), gage_CsAcqCfg_set.i64Depth);
//    _ftprintf(stdout, _T("SegmentSize: %d\n"), gage_CsAcqCfg_set.i64SegmentSize);

//    _ftprintf(stdout, _T("TriggerTimeout: %d\n"), gage_CsAcqCfg_set.i64TriggerTimeout);
//    _ftprintf(stdout, _T("TrigEnginesEn: %d\n"), gage_CsAcqCfg_set.u32TrigEnginesEn);
//    _ftprintf(stdout, _T("TriggerDelay: %d\n"), gage_CsAcqCfg_set.i64TriggerDelay);
//    _ftprintf(stdout, _T("TriggerHoldoff: %d\n"), gage_CsAcqCfg_set.i64TriggerHoldoff);
//    _ftprintf(stdout, _T("SampleOffset: %d\n"), gage_CsAcqCfg_set.i32SampleOffset);
//    _ftprintf(stdout, _T("TimeStampConfig: %d\n"), gage_CsAcqCfg_set.u32TimeStampConfig);

    gage_i32Status = CsSet(gage_hSystem, CS_ACQUISITION, &gage_CsAcqCfg_set);
    if (CS_FAILED(gage_i32Status)) DisplayErrorString(gage_i32Status);

    int n_trigger = (int)PyList_Size(gage_configuration_trigger);
    PyObject* gage_configuration_trigger_tmp;
    for (int trigger_count = 0; trigger_count < n_trigger; trigger_count++)
    {
        gage_configuration_trigger_tmp = PyList_GET_ITEM(gage_configuration_trigger, (Py_ssize_t) trigger_count);

        gage_CsTrigCfg_set.u32TriggerIndex = trigger_count + 1;
        gage_CsTrigCfg_set.u32Size = sizeof(CSTRIGGERCONFIG);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("Condition")), "k", &gage_CsTrigCfg_set.u32Condition);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("Level")), "l", &gage_CsTrigCfg_set.i32Level);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("Source")), "l", &gage_CsTrigCfg_set.i32Source);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("ExtCoupling")), "k", &gage_CsTrigCfg_set.u32ExtCoupling);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("ExtTriggerRange")), "k", &gage_CsTrigCfg_set.u32ExtTriggerRange);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("ExtImpedance")), "k", &gage_CsTrigCfg_set.u32ExtImpedance);
        PyArg_Parse(PyDict_GetItem(gage_configuration_trigger_tmp, PyUnicode_FromString("Relation")), "k", &gage_CsTrigCfg_set.u32Relation);

//        _ftprintf(stdout, _T("Condition: %d\n"), gage_CsTrigCfg_set.u32Condition);
//        _ftprintf(stdout, _T("Level: %d\n"), gage_CsTrigCfg_set.i32Level);
//        _ftprintf(stdout, _T("Source: %d\n"), gage_CsTrigCfg_set.i32Source);
//        _ftprintf(stdout, _T("ExtCoupling: %d\n"), gage_CsTrigCfg_set.u32ExtCoupling);
//        _ftprintf(stdout, _T("ExtTriggerRange: %d\n"), gage_CsTrigCfg_set.u32ExtTriggerRange);
//        _ftprintf(stdout, _T("ExtImpedance: %d\n"), gage_CsTrigCfg_set.u32ExtImpedance);
//        _ftprintf(stdout, _T("Relation: %d\n"), gage_CsTrigCfg_set.u32Relation);

        gage_i32Status = CsSet(gage_hSystem, CS_TRIGGER, &gage_CsTrigCfg_set);
        if (CS_FAILED(gage_i32Status)) DisplayErrorString(gage_i32Status);

    }

    int n_channel = (int)PyList_Size(gage_configuration_channel);
    PyObject* gage_configuration_channel_tmp;
    for (int channel_count = 0; channel_count < n_channel; channel_count++)
    {
        gage_configuration_channel_tmp = PyList_GET_ITEM(gage_configuration_channel, (Py_ssize_t) channel_count);

        gage_CsChanCfg_set.u32ChannelIndex = channel_count + 1;
        gage_CsChanCfg_set.u32Size = sizeof(CSCHANNELCONFIG);
        PyArg_Parse(PyDict_GetItem(gage_configuration_channel_tmp, PyUnicode_FromString("Term")), "k", &gage_CsChanCfg_set.u32Term);
        PyArg_Parse(PyDict_GetItem(gage_configuration_channel_tmp, PyUnicode_FromString("InputRange")), "k", &gage_CsChanCfg_set.u32InputRange);
        PyArg_Parse(PyDict_GetItem(gage_configuration_channel_tmp, PyUnicode_FromString("Impedance")), "k", &gage_CsChanCfg_set.u32Impedance);
        PyArg_Parse(PyDict_GetItem(gage_configuration_channel_tmp, PyUnicode_FromString("Filter")), "k", &gage_CsChanCfg_set.u32Filter);
        PyArg_Parse(PyDict_GetItem(gage_configuration_channel_tmp, PyUnicode_FromString("DcOffset")), "l", &gage_CsChanCfg_set.i32DcOffset);

//        _ftprintf(stdout, _T("Term: %d\n"), gage_CsChanCfg_set.u32Term);
//        _ftprintf(stdout, _T("InputRange: %d\n"), gage_CsChanCfg_set.u32InputRange);
//        _ftprintf(stdout, _T("Impedance: %d\n"), gage_CsChanCfg_set.u32Impedance);
//        _ftprintf(stdout, _T("Filter: %d\n"), gage_CsChanCfg_set.u32Filter);
//        _ftprintf(stdout, _T("DcOffset: %d\n"), gage_CsChanCfg_set.i32DcOffset);

        gage_i32Status = CsSet(gage_hSystem, CS_CHANNEL, &gage_CsChanCfg_set);
        if (CS_FAILED(gage_i32Status)) DisplayErrorString(gage_i32Status);

    }

    gage_i32Status = CsDo(gage_hSystem, ACTION_COMMIT_COERCE);
    if (CS_FAILED(gage_i32Status)) {
        DisplayErrorString(gage_i32Status);
        }
    return PyLong_FromLong(0);
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
		_ftprintf (stderr, _T("Unable to allocate memory 1\n"));
		CsFreeSystem(gage_hSystem);
		return PyErr_NoMemory();
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
	
	return PyLong_FromLong(0);
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
	
	return PyLong_FromLong(0);
}

static PyObject * gage_freememory(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args,""))
		return NULL;
		
	if ( NULL != gage_pBuffer)
	{
		VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		gage_pBuffer = NULL;
	}
	
	_ftprintf(stdout, _T("Freed memory.\n"));
	
	return PyLong_FromLong(0);
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
		CsFreeSystem(gage_hSystem);
		return (-1);
	}

	gage_dims[0] = gage_OutData.i64ActualLength;
	ret = PyArray_SimpleNewFromData(1, gage_dims, NPY_INT16, gage_pBuffer);

	return ret;
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

	if ( NULL != gage_pBuffer)
	{
		VirtualFree(gage_pBuffer, 0, MEM_RELEASE);
		gage_pBuffer = NULL;
	}

	_ftprintf(stdout,_T("Freed memory.\n"));
	
	return PyLong_FromLong(0);
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

static char gage_get_docstring[] =
	"Gets the data corresponding to given segment and channel.";

static char gage_close_docstring[] =
	"Close communication with the card and free allocated memory ressources.";

static char gage_system_info_docstring[] =
	"Get system info from the gage driver.";

static char gage_get_configuration_docstring[] =
	"Get configuration from the gage driver.";

static char gage_set_configuration_docstring[] =
	"Set configuration of the gage driver.";

static PyMethodDef module_methods[] =
{
    {"init", gage_init, METH_VARARGS, gage_init_docstring},
	{"acquire", gage_acquire, METH_VARARGS, gage_acquire_docstring},
	{"wait", gage_wait, METH_VARARGS, gage_wait_docstring},
	{"freememory", gage_freememory, METH_VARARGS, gage_freememory_docstring},
	{"get", gage_get, METH_VARARGS, gage_get_docstring},
	{"close", gage_close, METH_VARARGS, gage_close_docstring},
	{"system_info", gage_system_info, METH_VARARGS, gage_system_info_docstring},
	{"get_configuration", gage_get_configuration, METH_VARARGS, gage_get_configuration_docstring},
    {"set_configuration", gage_set_configuration, METH_VARARGS, gage_set_configuration_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_gage",     /* m_name */
        "Python API for Gage Compuscope. Based on the C SDK of Compuscope software, used in combination with numpy.",  /* m_doc */
        -1,                  /* m_size */
        module_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit__gage(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    /* Load `numpy` functionality. */
    import_array();
    return m;
};