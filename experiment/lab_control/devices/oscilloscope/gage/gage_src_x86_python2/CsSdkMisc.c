#include "windows.h"
#include "stdio.h"
#include "CsPrototypes.h"
#include "CsTchar.h"
#include "CsSdkMisc.h"


static BOOL g_bSuccess = TRUE;


void DisplayErrorString(const int32 i32Status)
{
	TCHAR	szErrorString[255];
	if( CS_FAILED (i32Status) )
	{
		g_bSuccess = FALSE;
	}

	CsGetErrorString(i32Status, szErrorString, 255);
	_ftprintf(stderr, _T("\n%s\n"), szErrorString);
}

void DisplayFinishString(void)
{
	if ( g_bSuccess )
	{
		_ftprintf (stdout, _T("\nAcquisition completed. \nAll channels are saved as ASCII data files in the current working directory.\n"));
	}
	else
	{
		_ftprintf (stderr, _T("\nAn error has occurred.\n"));
	}
}

BOOL DataCaptureComplete(const CSHANDLE hSystem)
{
	int32 i32Status;
/*
	Wait until the acquisition is complete.
*/
	i32Status = CsGetStatus(hSystem);
	while (!(ACQ_STATUS_READY == i32Status))
	{
		i32Status = CsGetStatus(hSystem);
	}

	return TRUE;
}