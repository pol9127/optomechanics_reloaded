/*
 * FPGA Interface C API example for GCC for computers running Linux.
 *
 * NOTE: In order to run this example, you must compile a LabVIEW FPGA bitfile
 *       and generate a C API for it. For more information about using this
 *       example, refer to the Examples topic of the FPGA Interface C API Help,
 *       located under
 *       Start>>All Programs>>National Instruments>>FPGA Interface C API.
 */

#include "NiFpga_toplevel.h"
#include <stdio.h>
#include <stdlib.h>

void start_fpga(NiFpga_Session* session, NiFpga_Status* status)
{
	/* must be called before any other calls */
	*status = NiFpga_Initialize();

	if (NiFpga_IsNotError(*status))
	{
		/* opens a session, downloads the bitstream, and runs the FPGA */
		NiFpga_MergeStatus(status, NiFpga_Open(NiFpga_toplevel_Bitfile,
												NiFpga_toplevel_Signature,
												"RIO0",
												NiFpga_OpenAttribute_NoRun,
												session));
		if (NiFpga_IsNotError(*status))
		{
			/* run the FPGA application */
			NiFpga_MergeStatus(status, NiFpga_Run(*session, 0));
		}
	}
}

void stop_fpga(NiFpga_Session* session, NiFpga_Status* status)
{
	/* close the session now that we're done */
	NiFpga_MergeStatus(status, NiFpga_Close(*session, 0));

	/* must be called after all other calls */
	NiFpga_MergeStatus(status, NiFpga_Finalize());
}

int16_t read_DeviceTemperature(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_DeviceTemperature,&value));
	return value;
}

uint16_t read_LoopTicks(NiFpga_Session* session, NiFpga_Status* status)
{
	uint16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadU16(*session,NiFpga_toplevel_IndicatorU16_Ticks,&value));
	return value;
}

void set_LoopTicks(uint16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_Countticks,value));
}

void set_FifoTimeout(int32_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI32(*session,NiFpga_toplevel_ControlI32_FIFOWriteTimeout,value));
}

size_t configure_FIFO_AI(size_t requestedDepth, NiFpga_Session* session, NiFpga_Status* status)
{
	size_t actualDepth;
	NiFpga_MergeStatus(status, NiFpga_ConfigureFifo2(*session, NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI, requestedDepth, &actualDepth));
	return actualDepth;
}

void start_FIFO_AI(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_StartFifo(*session, NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI));
}

void stop_FIFO_AI(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_StopFifo(*session, NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI));
}

void toggle_AI_acquisition(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_HighSpeedAIrunning,state));
}

_Bool AI_acquisition_active(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_Bool state;

	NiFpga_MergeStatus(status, NiFpga_ReadBool(*session,NiFpga_toplevel_IndicatorBool_HighSpeedAIactive,&state));
	return state;
}

void start_relaxation_measurement(uint16_t ms_msrmt, uint16_t ms_after_msrmt, uint16_t ms_before_fb, uint16_t ms_fb, _Bool co2_switch, uint16_t ms_before_co2, uint16_t ms_co2, int16_t co2_on, int16_t co2_off, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msMeasurement,ms_msrmt));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msafterMsrmt,ms_after_msrmt));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msbeforeFBoff,ms_before_fb));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msFBoff,ms_fb));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msbeforeCO2on,ms_before_co2));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msCO2on,ms_co2));
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_CO2on,co2_on));
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_CO2off,co2_off));
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_SwitchCO2,co2_switch));
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_StartRelaxationMsrmt,1));
}

void start_timed_measurement(uint16_t ms_msrmt, uint16_t ms_after_msrmt, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msMeasurement,ms_msrmt));
	NiFpga_MergeStatus(status, NiFpga_WriteU16(*session,NiFpga_toplevel_ControlU16_msafterMsrmt,ms_after_msrmt));
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,   NiFpga_toplevel_ControlBool_StarttimedMeasurement,1));
}

void read_FIFO_AI(uint64_t* input, size_t size, NiFpga_Session* session, NiFpga_Status* status,size_t* elementsRemaining)
{
	/* copy FIFO data from the FPGA */
	NiFpga_MergeStatus(status,
					   NiFpga_ReadFifoU64(*session,
							   	   	   	  NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI,
										  input,
										  size,
										  NiFpga_InfiniteTimeout,
										  elementsRemaining));
}

void read_FIFO_AI_timeout(uint64_t* input, size_t size, uint32_t timeout, NiFpga_Session* session, NiFpga_Status* status,size_t* elementsRemaining)
{
	/* copy FIFO data from the FPGA */
	NiFpga_MergeStatus(status,
					   NiFpga_ReadFifoU64(*session,
							   	   	   	  NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI,
										  input,
										  size,
										  timeout,
										  elementsRemaining));
}

void unpack_data(uint64_t* input, int16_t* AI0, int16_t* AI1, int16_t* AI2, uint16_t* ticks, size_t size)
{
	int iter;
	for (iter = 0; iter < size; ++iter) {
		uint64_t incoming = input[iter];
		AI0[iter] = (int16_t) (incoming >> 48);
		AI1[iter] = (int16_t) ((incoming >> 32) & 0xffff);
		AI2[iter] = (int16_t) ((incoming >> 16) & 0xffff);
		ticks[iter] = incoming & 0xffff;
	}
}

void read_FIFO_AI_unpack(int16_t* AI0, int16_t* AI1, int16_t* AI2, uint16_t* ticks, size_t size, NiFpga_Session* session, NiFpga_Status* status,size_t* elementsRemaining)
{
	uint64_t input[size];

	read_FIFO_AI(input, size, session, status, elementsRemaining);

	unpack_data(input,AI0,AI1,AI2,ticks,size);
}

void read_FIFO_AI_unpack_timeout(int16_t* AI0, int16_t* AI1, int16_t* AI2, uint16_t* ticks, size_t size, uint32_t timeout, NiFpga_Session* session, NiFpga_Status* status,size_t* elementsRemaining)
{
	uint64_t input[size];

	read_FIFO_AI_timeout(input, size, timeout, session, status, elementsRemaining);

	unpack_data(input,AI0,AI1,AI2,ticks,size);
}

int16_t read_AI3(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_Connector0AI3,&value));
	return value;
}

int16_t read_AI4(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_Connector0AI4,&value));
	return value;
}

int16_t read_AI5(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_Connector0AI5,&value));
	return value;
}

int16_t read_AI6(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_Connector0AI6,&value));
	return value;
}

int16_t read_AI7(NiFpga_Session* session, NiFpga_Status* status)
{
	int16_t value;

	NiFpga_MergeStatus(status, NiFpga_ReadI16(*session,NiFpga_toplevel_IndicatorI16_Connector0AI7,&value));
	return value;
}

void set_AO0(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO0,value));
}

void set_AO1(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO1,value));
}

void set_AO2(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO2,value));
}

void set_AO3(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO3,value));
}

void set_AO4(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO4,value));
}

void set_AO5(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO5,value));
}

void set_AO6(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO6,value));
}

void set_AO7(int16_t value, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteI16(*session,NiFpga_toplevel_ControlI16_Connector0AO7,value));
}

void set_DIO0(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO0,state));
}

void set_DIO1(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO1,state));
}

void set_DIO2(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO2,state));
}

void set_DIO3(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO3,state));
}

void set_DIO4(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO4,state));
}

void set_DIO5(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO5,state));
}

void set_DIO6(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO6,state));
}

void set_DIO7(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO7,state));
}

void set_DIO8(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO8,state));
}

void set_DIO9(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO9,state));
}

void set_DIO10(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO10,state));
}

void set_DIO11(_Bool state, NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_MergeStatus(status, NiFpga_WriteBool(*session,NiFpga_toplevel_ControlBool_Connector0DIO11,state));
}

_Bool read_DIO12(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_Bool state;

	NiFpga_MergeStatus(status, NiFpga_ReadBool(*session,NiFpga_toplevel_IndicatorBool_Connector0DIO12,&state));
	return state;
}

_Bool read_DIO13(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_Bool state;

	NiFpga_MergeStatus(status, NiFpga_ReadBool(*session,NiFpga_toplevel_IndicatorBool_Connector0DIO13,&state));
	return state;
}

_Bool read_DIO14(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_Bool state;

	NiFpga_MergeStatus(status, NiFpga_ReadBool(*session,NiFpga_toplevel_IndicatorBool_Connector0DIO14,&state));
	return state;
}

_Bool read_DIO15(NiFpga_Session* session, NiFpga_Status* status)
{
	NiFpga_Bool state;

	NiFpga_MergeStatus(status, NiFpga_ReadBool(*session,NiFpga_toplevel_IndicatorBool_Connector0DIO15,&state));
	return state;
}

/*
int main()
{
	NiFpga_Session session=0;
	NiFpga_Status status=0;

	size_t size=10000;
	int16_t input[size];

	start_fpga(&session, &status);
	printf("%i", status);
	configure_FIFO_AI0(size,&session,&status);
	start_FIFO_AI0(&session,&status);

	int var = 0;

	read_FIFO_AI0(input, size, &session, &status);

	for (var = 1020; var < 1030; ++var) {
		printf("input[%i]=%i\n",var,input[var]);
	}

//	read_FIFO_AI0(input, size, &session, &status);
//
//	for (var = 0; var < size; ++var) {
//		printf("input[%i]=%i\n",var,input[var]);
//	}

	stop_FIFO_AI0(&session,&status);
	stop_fpga(&session, &status);

	return 0;
}
*/
