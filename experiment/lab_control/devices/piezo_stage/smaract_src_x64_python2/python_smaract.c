#include <Python.h>
#include <numpy/arrayobject.h>

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <MCSControl.h>

static	SA_INDEX					mcsHandle = 0;
static  SA_STATUS                   error = SA_OK;

/* All MCS commands return a status/error code which helps analyzing
   problems */
void ExitIfError(SA_STATUS st) {
    if(st != SA_OK) {
        printf("MCS error %u\n",st);
        exit(1);
    }
}

//int main(int argc, char* argv[])
static PyObject * smaract_init(PyObject * self, PyObject * args)
{
    const char *loc;
    int loc_length = 0;
    if (!PyArg_ParseTuple(args, "|s#", &loc, &loc_length))
		return NULL;

	if (loc_length != 0)
	{
	    error = SA_OpenSystem(&mcsHandle, loc, "sync");
	}
	else
	{
	    error = SA_OpenSystem(&mcsHandle, "usb:ix:0", "sync");
	}

    // ----------------------------------------------------------------------------------
    // open the first MCS with USB interface in synchronous communication mode

    printf("Open system: Error: %u\n", error);
    if(error)
        return 1;

    return PyInt_FromLong(0);
}

static PyObject * smaract_close(PyObject * self, PyObject * args)
{
	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	ExitIfError( SA_CloseSystem(mcsHandle) );

	return PyInt_FromLong(0);
}

static PyObject * smaract_get_configuration(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    unsigned int sensorTypeTmp;
    unsigned int* sensorType = NULL;
    int* positionLimitMin = NULL;
    int* positionLimitMax = NULL;
    int positionLimitMinTmp;
    int positionLimitMaxTmp;

    unsigned int scaleInvertedTmp;
    unsigned int* scaleInverted = NULL;
    int* scale = NULL;
    int scaleTmp;


    unsigned int i;
    npy_intp dims[1] = {1};

	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));
	sensorType = (unsigned int*) calloc ( numOfChannels, sizeof(unsigned int));
	positionLimitMin = (int*) calloc ( numOfChannels, sizeof(int));
	positionLimitMax = (int*) calloc ( numOfChannels, sizeof(int));
	scaleInverted = (unsigned int*) calloc ( numOfChannels, sizeof(unsigned int));
	scale = (int*) calloc ( numOfChannels, sizeof(int));

	for(i=0; i<numOfChannels; i++)
    {
        ExitIfError(SA_GetSensorType_S(mcsHandle, i, &sensorTypeTmp));
        *(sensorType + i) = sensorTypeTmp;
        ExitIfError(SA_GetPositionLimit_S(mcsHandle, i, &positionLimitMinTmp, &positionLimitMaxTmp));
        *(positionLimitMin + i) = positionLimitMinTmp;
        *(positionLimitMax + i) = positionLimitMaxTmp;
        ExitIfError(SA_GetScale_S(mcsHandle, i, &scaleTmp, &scaleInvertedTmp));
        *(scaleInverted + i) = scaleInvertedTmp;
        *(scale + i) = scaleTmp;
    }

    dims[0] = numOfChannels;
//	return Py_BuildValue("{sIsOsOsOsOsO}",
//                             "NumOfChannels", numOfChannels,
//                             "SensorType", PyArray_SimpleNewFromData(1, dims, NPY_UINT32, sensorType),
//                             "PositionLimitMin", PyArray_SimpleNewFromData(1, dims, NPY_INT32, positionLimitMin),
//                             "PositionLimitMax", PyArray_SimpleNewFromData(1, dims, NPY_INT32, positionLimitMax),
//                             "ScaleInverted", PyArray_SimpleNewFromData(1, dims, NPY_UINT32, scaleInverted),
//                             "Scale", PyArray_SimpleNewFromData(1, dims, NPY_INT32, scale));
	return Py_BuildValue("{sI}",
                             "NumOfChannels", numOfChannels);

}

static PyObject * smaract_get_channel_configuration(PyObject * self, PyObject * args)
{
    unsigned int* channel = 0;
    int* channel_property = 0;

    npy_intp dims[1] = {1};

	if (!PyArg_ParseTuple(args, "I", &channel))
		return NULL;

	ExitIfError(SA_GetChannelProperty_S(mcsHandle, channel, SA_EPK(SA_GENERAL, SA_LOW_VIBRATION, SA_OPERATION_MODE), &channel_property));

	return Py_BuildValue("{si}",
                             "property", channel_property);

}

static PyObject * smaract_set_channel_stickslip_mode(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    int* channel_property = 0;

    npy_intp dims[1] = {1};
    unsigned int i;

	if (!PyArg_ParseTuple(args, "i", &channel_property))
		return NULL;

    ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));

	for(i=0; i<numOfChannels; i++)
    {
        if (channel_property == 0)
        {
            ExitIfError(SA_SetChannelProperty_S(mcsHandle, i, SA_EPK(SA_GENERAL, SA_LOW_VIBRATION, SA_OPERATION_MODE), SA_DISABLED));
        }
        else if (channel_property == 1)
        {
            ExitIfError(SA_SetChannelProperty_S(mcsHandle, i, SA_EPK(SA_GENERAL, SA_LOW_VIBRATION, SA_OPERATION_MODE), SA_ENABLED));
        };
    }
	return PyInt_FromLong(0);
}

static PyObject * smaract_set_closed_loop_move_acceleration(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    unsigned int* acceleration = 0;

    npy_intp dims[1] = {1};
    unsigned int i;

	if (!PyArg_ParseTuple(args, "I", &acceleration))
		return NULL;

    ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));

	for(i=0; i<numOfChannels; i++)
    {
        ExitIfError(SA_SetClosedLoopMoveAcceleration_S(mcsHandle, i, acceleration));
    }
	return PyInt_FromLong(0);
}

static PyObject * smaract_set_closed_loop_move_speed(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    unsigned int* speed = 0;

    npy_intp dims[1] = {1};
    unsigned int i;

	if (!PyArg_ParseTuple(args, "I", &speed))
		return NULL;

    ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));

	for(i=0; i<numOfChannels; i++)
    {
        ExitIfError(SA_SetClosedLoopMoveAcceleration_S(mcsHandle, i, speed));
    }
	return PyInt_FromLong(0);
}

static PyObject * smaract_get_position(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    int* position = NULL;
    int positionTmp;

    unsigned int i;
    npy_intp dims[1] = {1};

	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));
	position = (int*) calloc ( numOfChannels, sizeof(int));

	for(i=0; i<numOfChannels; i++)
    {
        ExitIfError(SA_GetPosition_S(mcsHandle, i, &positionTmp));
        *(position + i) = positionTmp;
    }

    dims[0] = numOfChannels;
	return PyArray_SimpleNewFromData(1, dims, NPY_INT32, position);
}

static PyObject * smaract_move_relative(PyObject * self, PyObject * args)
{
    unsigned int channel;
    int position;

	if (!PyArg_ParseTuple(args, "Ii", &channel, &position))
		return NULL;

	ExitIfError(SA_GotoPositionRelative_S(mcsHandle, channel, position, 0));

	return PyInt_FromLong(0);
}

static PyObject * smaract_move_absolute(PyObject * self, PyObject * args)
{
    unsigned int channel;
    int position;

	if (!PyArg_ParseTuple(args, "Ii", &channel, &position))
		return NULL;

	ExitIfError(SA_GotoPositionAbsolute_S(mcsHandle, channel, position, 0));

	return PyInt_FromLong(0);
}

static PyObject * smaract_get_status(PyObject * self, PyObject * args)
{
    unsigned int numOfChannels = 0;
    unsigned int* status = NULL;
    unsigned int statusTmp;
    unsigned int i;
    npy_intp dims[1] = {1};

	if (!PyArg_ParseTuple(args, ""))
		return NULL;

	ExitIfError(SA_GetNumberOfChannels(mcsHandle,&numOfChannels));
	status = (unsigned int*) calloc ( numOfChannels, sizeof(unsigned int));

	for(i=0; i<numOfChannels; i++)
    {
        ExitIfError(SA_GetStatus_S(mcsHandle, i, &statusTmp));
        *(status + i) = statusTmp;
    }

    dims[0] = numOfChannels;
    return PyArray_SimpleNewFromData(1, dims, NPY_UINT32, status);
}




static char smaract_init_docstring[] =
	"Initialize communication with the smaract driver.";

static char smaract_close_docstring[] =
	"Close communication with the smaract.";

static char smaract_get_configuration_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_get_channel_configuration_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_set_channel_stickslip_mode_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_set_closed_loop_move_acceleration_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_set_closed_loop_move_speed_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_get_position_docstring[] =
	"Get Configuration of smaract hardware.";

static char smaract_move_relative_docstring[] =
	"Move Piezo relative to current position.";

static char smaract_move_absolute_docstring[] =
	"Move Piezo to abolute position.";

static char smaract_get_status_docstring[] =
	"Get Status of smaract hardware.";


static PyMethodDef module_methods[] = 
{
    {"init", smaract_init, METH_VARARGS, smaract_init_docstring},
	{"close", smaract_close, METH_VARARGS, smaract_close_docstring},
	{"get_configuration", smaract_get_configuration, METH_VARARGS, smaract_get_configuration_docstring},
	{"get_channel_configuration", smaract_get_channel_configuration, METH_VARARGS, smaract_get_channel_configuration_docstring},
	{"set_channel_stickslip_mode", smaract_set_channel_stickslip_mode, METH_VARARGS, smaract_set_channel_stickslip_mode_docstring},
	{"set_closed_loop_move_acceleration", smaract_set_closed_loop_move_acceleration, METH_VARARGS, smaract_set_closed_loop_move_acceleration_docstring},
	{"set_closed_loop_move_speed", smaract_set_closed_loop_move_speed, METH_VARARGS, smaract_set_closed_loop_move_speed_docstring},
	{"get_position", smaract_get_position, METH_VARARGS, smaract_get_position_docstring},
    {"move_relative", smaract_move_relative, METH_VARARGS, smaract_move_relative_docstring},
    {"move_absolute", smaract_move_absolute, METH_VARARGS, smaract_move_absolute_docstring},
    {"get_status", smaract_get_status, METH_VARARGS, smaract_get_status_docstring},
    {NULL, NULL, 0, NULL}
};

static char module_docstring[] =
    "Python API for Smaract Hardware, used in combination with numpy.";

PyMODINIT_FUNC init_smaract(void)
{
    PyObject *m = Py_InitModule3("_smaract", module_methods, module_docstring);
    if (m == NULL)
        return;

    /* Load `numpy` functionality. */
    import_array();
};