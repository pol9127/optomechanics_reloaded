/*
 * Generated with the FPGA Interface C API Generator 14.0.0
 * for NI-RIO 14.0.0 or later.
 */

#ifndef __NiFpga_toplevel_h__
#define __NiFpga_toplevel_h__

#ifndef NiFpga_Version
   #define NiFpga_Version 1400
#endif

#include "NiFpga.h"

/**
 * The filename of the FPGA bitfile.
 *
 * This is a #define to allow for string literal concatenation. For example:
 *
 *    static const char* const Bitfile = "C:\\" NiFpga_toplevel_Bitfile;
 */
#define NiFpga_toplevel_Bitfile "NiFpga_toplevel.lvbitx"

/**
 * The signature of the FPGA bitfile.
 */
static const char* const NiFpga_toplevel_Signature = "9CB7CA81D9F5C69C8A3A01F00A945D20";

typedef enum
{
   NiFpga_toplevel_IndicatorBool_Connector0DIO12 = 0x815A,
   NiFpga_toplevel_IndicatorBool_Connector0DIO13 = 0x8156,
   NiFpga_toplevel_IndicatorBool_Connector0DIO14 = 0x8152,
   NiFpga_toplevel_IndicatorBool_Connector0DIO15 = 0x814E,
   NiFpga_toplevel_IndicatorBool_HighSpeedAIactive = 0x8116,
   NiFpga_toplevel_IndicatorBool_TimedOut = 0x8146,
} NiFpga_toplevel_IndicatorBool;

typedef enum
{
   NiFpga_toplevel_IndicatorI16_Connector0AI3 = 0x81AE,
   NiFpga_toplevel_IndicatorI16_Connector0AI4 = 0x81BA,
   NiFpga_toplevel_IndicatorI16_Connector0AI5 = 0x81B6,
   NiFpga_toplevel_IndicatorI16_Connector0AI6 = 0x81B2,
   NiFpga_toplevel_IndicatorI16_Connector0AI7 = 0x810E,
   NiFpga_toplevel_IndicatorI16_DeviceTemperature = 0x814A,
} NiFpga_toplevel_IndicatorI16;

typedef enum
{
   NiFpga_toplevel_IndicatorU16_Ticks = 0x81C6,
} NiFpga_toplevel_IndicatorU16;

typedef enum
{
   NiFpga_toplevel_ControlBool_Connector0DIO0 = 0x816E,
   NiFpga_toplevel_ControlBool_Connector0DIO1 = 0x816A,
   NiFpga_toplevel_ControlBool_Connector0DIO10 = 0x8176,
   NiFpga_toplevel_ControlBool_Connector0DIO11 = 0x8172,
   NiFpga_toplevel_ControlBool_Connector0DIO2 = 0x8166,
   NiFpga_toplevel_ControlBool_Connector0DIO3 = 0x8162,
   NiFpga_toplevel_ControlBool_Connector0DIO4 = 0x815E,
   NiFpga_toplevel_ControlBool_Connector0DIO5 = 0x818A,
   NiFpga_toplevel_ControlBool_Connector0DIO6 = 0x8186,
   NiFpga_toplevel_ControlBool_Connector0DIO7 = 0x8182,
   NiFpga_toplevel_ControlBool_Connector0DIO8 = 0x817E,
   NiFpga_toplevel_ControlBool_Connector0DIO9 = 0x817A,
   NiFpga_toplevel_ControlBool_HighSpeedAIrunning = 0x8126,
   NiFpga_toplevel_ControlBool_StartRelaxationMsrmt = 0x8122,
   NiFpga_toplevel_ControlBool_StarttimedMeasurement = 0x8112,
   NiFpga_toplevel_ControlBool_SwitchCO2 = 0x811A,
} NiFpga_toplevel_ControlBool;

typedef enum
{
   NiFpga_toplevel_ControlI16_CO2off = 0x812E,
   NiFpga_toplevel_ControlI16_CO2on = 0x812A,
   NiFpga_toplevel_ControlI16_Connector0AO0 = 0x8196,
   NiFpga_toplevel_ControlI16_Connector0AO1 = 0x8192,
   NiFpga_toplevel_ControlI16_Connector0AO2 = 0x818E,
   NiFpga_toplevel_ControlI16_Connector0AO3 = 0x81AA,
   NiFpga_toplevel_ControlI16_Connector0AO4 = 0x81A6,
   NiFpga_toplevel_ControlI16_Connector0AO5 = 0x81A2,
   NiFpga_toplevel_ControlI16_Connector0AO6 = 0x819A,
   NiFpga_toplevel_ControlI16_Connector0AO7 = 0x819E,
} NiFpga_toplevel_ControlI16;

typedef enum
{
   NiFpga_toplevel_ControlU16_Countticks = 0x81BE,
   NiFpga_toplevel_ControlU16_msCO2on = 0x8132,
   NiFpga_toplevel_ControlU16_msFBoff = 0x813A,
   NiFpga_toplevel_ControlU16_msMeasurement = 0x8142,
   NiFpga_toplevel_ControlU16_msafterMsrmt = 0x811E,
   NiFpga_toplevel_ControlU16_msbeforeCO2on = 0x8136,
   NiFpga_toplevel_ControlU16_msbeforeFBoff = 0x813E,
} NiFpga_toplevel_ControlU16;

typedef enum
{
   NiFpga_toplevel_ControlI32_FIFOWriteTimeout = 0x81C0,
} NiFpga_toplevel_ControlI32;

typedef enum
{
   NiFpga_toplevel_TargetToHostFifoU64_FIFO_AI = 0,
} NiFpga_toplevel_TargetToHostFifoU64;

#endif
