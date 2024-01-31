#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <math.h>
#include <complex.h>

#include "gsl_const_mks.c"
#include "custom_module.c"
#include "numeric_module.c"

static int ret_dims[1] = {0};
static long ret_dims_2d[2] = {0L, 0L};
static long ret_dims_3d[3] = {0L, 0L, 0L};
static long ret_dims_4d[4] = {0L, 0L, 0L, 0L};

static PyObject * r_s(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, k1, k2;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOdddd", &kx, &ky, &k1, &k2, &mu1, &mu2))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _r_s(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * r_p(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double eps1, eps2, k1, k2;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOdddd", &kx, &ky, &k1, &k2, &eps1, &eps2))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _r_p(_kx[kx_itr], _ky[ky_itr], k1, k2, eps1, eps2);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * t_s(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, k1, k2;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOdddd", &kx, &ky, &k1, &k2, &mu1, &mu2))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _t_s(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * t_p(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, eps1, eps2, k1, k2;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOdddddd", &kx, &ky, &k1, &k2, &mu1, &mu2, &eps1, &eps2))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _t_p(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2, eps1, eps2);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * r_s_membrane(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, k1, k2, d;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOddddd", &kx, &ky, &k1, &k2, &mu1, &mu2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _r_s_membrane(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2, d);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * r_p_membrane(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double eps1, eps2, k1, k2, d;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOddddd", &kx, &ky, &k1, &k2, &eps1, &eps2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _r_p_membrane(_kx[kx_itr], _ky[ky_itr], k1, k2, eps1, eps2, d);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * t_s_membrane(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, k1, k2, d;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOddddd", &kx, &ky, &k1, &k2, &mu1, &mu2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _t_s_membrane(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2, d);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * t_p_membrane(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double mu1, mu2, eps1, eps2, k1, k2, d;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val;
	if (!PyArg_ParseTuple(args, "OOddddddd", &kx, &ky, &k1, &k2, &mu1, &mu2, &eps1, &eps2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            ret_val = _t_p_membrane(_kx[kx_itr], _ky[ky_itr], k1, k2, mu1, mu2, eps1, eps2, d);
            (outdata + kx_itr * ky_len)[ky_itr].real = (npy_double) creal(ret_val);
            (outdata + kx_itr * ky_len)[ky_itr].imag = (npy_double) cimag(ret_val);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_inc(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double f;
    double w0;
    double k1;
    double E0;

    long i;
    PyObject *arr;
    npy_double *outdata;

	if (!PyArg_ParseTuple(args, "OOdddd", &kx, &ky, &f, &w0, &k1, &E0))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));


    ret_dims_2d[0] = kx_len;
    ret_dims_2d[1] = ky_len;

    arr = PyArray_SimpleNew(2, ret_dims_2d, PyArray_DOUBLE);
    outdata = (npy_double *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            (outdata + kx_itr * ky_len)[ky_itr] = (npy_double) _E_inc(_kx[kx_itr], _ky[ky_itr], f, w0, k1, E0);
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_inf(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double f;
    double w0;
    double k1;
    double E0;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "OOdddd", &kx, &ky, &f, &w0, &k1, &E0))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    ret_dims_3d[0] = kx_len;
    ret_dims_3d[1] = ky_len;
    ret_dims_3d[2] = 3;


    arr = PyArray_SimpleNew(3, ret_dims_3d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_inf(_kx[kx_itr], _ky[ky_itr], f, w0, k1, E0, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
            };
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_r_inf(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double f;
    double w0;
    double k1, k2;
    double E0;
    double mu1;
    double mu2;
    double eps1;
    double eps2;
    double d = 0;
    double z0;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "OOdddddddddd|d", &kx, &ky, &f, &w0, &k1, &k2, &E0, &z0, &mu1, &mu2, &eps1, &eps2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    ret_dims_3d[0] = kx_len;
    ret_dims_3d[1] = ky_len;
    ret_dims_3d[2] = 3;


    arr = PyArray_SimpleNew(3, ret_dims_3d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_r_inf(_kx[kx_itr], _ky[ky_itr], f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
            };
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_r_integrand(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;
    double x, y, z;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double f;
    double w0;
    double k1, k2;
    double E0;
    double mu1;
    double mu2;
    double eps1;
    double eps2;
    double d = 0;
    double z0;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "dddOOdddddddddd|d", &x, &y, &z, &kx, &ky, &f, &w0, &k1, &k2, &E0, &z0, &mu1, &mu2, &eps1, &eps2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    ret_dims_3d[0] = kx_len;
    ret_dims_3d[1] = ky_len;
    ret_dims_3d[2] = 3;


    arr = PyArray_SimpleNew(3, ret_dims_3d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_r_integrand(x, y, z, _kx[kx_itr], _ky[ky_itr], f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
            };
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_f_integrand(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;
    double x, y, z;

    double * _kx;
    double * _ky;

    long kx_len;
    long ky_len;

    long kx_itr;
    long ky_itr;

    double f;
    double w0;
    double k1, k2;
    double E0;

    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "dddOOdddd", &x, &y, &z, &kx, &ky, &f, &w0, &k1, &E0))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    ret_dims_3d[0] = kx_len;
    ret_dims_3d[1] = ky_len;
    ret_dims_3d[2] = 3;


    arr = PyArray_SimpleNew(3, ret_dims_3d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_f_integrand(x, y, z, _kx[kx_itr], _ky[ky_itr], f, w0, k1, E0, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                (outdata + kx_itr * ky_len * 3 + ky_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
            };
        };
    };

    free(_kx);
    free(_ky);

    return arr;
}

static PyObject * E_r(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;
    double * _kx;
    double * _ky;
    long kx_len;
    long ky_len;
    long kx_itr;
    long ky_itr;
    double delta_kx;
    double delta_ky;

    PyObject * x;
    PyObject * y;
    PyObject * z;
    double * _x;
    double * _y;
    double * _z;
    long x_len;
    long y_len;
    long z_len;
    long x_itr;
    long y_itr;
    long z_itr;

    double f;
    double w0;
    double k1, k2;
    double E0;
    double mu1;
    double mu2;
    double eps1;
    double eps2;
    double d = 0;
    double z0;
    double complex prefactor;
    double kz1;
    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "OOOOOdddddddddddd|d", &x, &y, &z, &kx, &ky, &delta_kx, &delta_ky, &f, &w0, &k1, &k2, &E0, &z0, &mu1, &mu2, &eps1, &eps2, &d))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    double complex *** array = (double complex ***)malloc(kx_len * sizeof(double complex**));
    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        array[kx_itr] = (double complex **) malloc(ky_len * sizeof(double complex *));
        for (ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            array[kx_itr][ky_itr] = (double complex *)malloc(3 * sizeof(double complex));
        };
    };

    double complex ** factor_xyz = (double complex **)malloc(kx_len * sizeof(double complex*));
    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        factor_xyz[kx_itr] = (double complex *) malloc(ky_len * sizeof(double complex));
    };


    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_r_inf(_kx[kx_itr], _ky[ky_itr], f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                array[kx_itr][ky_itr][i] = ret_val_3[i];
            };
        };
    };

    x_len = (long)PyList_Size(x);
    _x = malloc(x_len * sizeof(double));
    if(_x == NULL)
    {
        printf("allocating memory for x failed\n");
    }
    for(i = 0; i < x_len; i++)
        PyArg_Parse(PyList_GetItem(x, i), "d", (_x + i));

    y_len = (long)PyList_Size(y);
    _y = malloc(y_len * sizeof(double));
    if(_y == NULL)
    {
        printf("allocating memory for y failed\n");
    }
    for(i = 0; i < y_len; i++)
        PyArg_Parse(PyList_GetItem(y, i), "d", (_y + i));

    z_len = (long)PyList_Size(z);
    _z = malloc(z_len * sizeof(double));
    if(_z == NULL)
    {
        printf("allocating memory for z failed\n");
    }
    for(i = 0; i < z_len; i++)
        PyArg_Parse(PyList_GetItem(z, i), "d", (_z + i));

    ret_dims_4d[0] = x_len;
    ret_dims_4d[1] = y_len;
    ret_dims_4d[2] = z_len;
    ret_dims_4d[3] = 3;

    arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(x_itr = 0; x_itr < x_len; x_itr++)
    {
        for(y_itr = 0; y_itr < y_len; y_itr++)
        {
            for(z_itr = 0; z_itr < z_len; z_itr++)
            {
                prefactor = -I * f / (2 * M_PI) * (cos(-k1 * f) + I * sin(-k1 * f));
                for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
                {
                    for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
                    {
                        if(_kx[kx_itr] * _kx[kx_itr] + _ky[ky_itr] * _ky[ky_itr] < k1 * k1)
                        {
                            kz1 = sqrt(k1 * k1 - _kx[kx_itr] * _kx[kx_itr] - _ky[ky_itr] * _ky[ky_itr]);
                            factor_xyz[kx_itr][ky_itr] = prefactor * (cos(_kx[kx_itr] * _x[x_itr] + _ky[ky_itr] * _y[y_itr] - kz1 * _z[z_itr]) + I * sin(_kx[kx_itr] * _x[x_itr] + _ky[ky_itr] * _y[y_itr] - kz1 * _z[z_itr])) / kz1;
                        }
                        else
                        {
                            factor_xyz[kx_itr][ky_itr] = 0;
                        };
                    };
                };
                _dblsimps(factor_xyz, array, delta_kx, delta_ky, kx_len, ky_len, ret_val_3);
                for(i = 0; i < 3; i++)
                {
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
                };
            };
        };
    };

    free(_x);
    free(_y);
    free(_z);
    free(_kx);
    free(_ky);
    free(array);
    return arr;
}

static PyObject * E_f(PyObject * self, PyObject * args)
{
    PyObject * kx;
    PyObject * ky;
    double * _kx;
    double * _ky;
    long kx_len;
    long ky_len;
    long kx_itr;
    long ky_itr;
    double delta_kx;
    double delta_ky;

    PyObject * x;
    PyObject * y;
    PyObject * z;
    double * _x;
    double * _y;
    double * _z;
    long x_len;
    long y_len;
    long z_len;
    long x_itr;
    long y_itr;
    long z_itr;

    double f;
    double w0;
    double k1, k2;
    double E0;
    double mu1;
    double mu2;
    double eps1;
    double eps2;
    double d = 0;
    double z0;
    double complex prefactor;
    double kz1;
    long i;
    PyObject *arr;
    npy_complex128 *outdata;
    double complex ret_val_3[3];

	if (!PyArg_ParseTuple(args, "OOOOOdddddd", &x, &y, &z, &kx, &ky, &delta_kx, &delta_ky, &f, &w0, &k1, &E0))
			return NULL;


    kx_len = (long)PyList_Size(kx);
    _kx = malloc(kx_len * sizeof(double));
    if(_kx == NULL)
    {
        printf("allocating memory for kx failed\n");
    }
    for(i = 0; i < kx_len; i++)
        PyArg_Parse(PyList_GetItem(kx, i), "d", (_kx + i));

    ky_len = (long)PyList_Size(ky);
    _ky = malloc(ky_len * sizeof(double));
    if(_ky == NULL)
    {
        printf("allocating memory for ky failed\n");
    }
    for(i = 0; i < ky_len; i++)
        PyArg_Parse(PyList_GetItem(ky, i), "d", (_ky + i));

    double complex *** array = (double complex ***)malloc(kx_len * sizeof(double complex**));
    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        array[kx_itr] = (double complex **) malloc(ky_len * sizeof(double complex *));
        for (ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            array[kx_itr][ky_itr] = (double complex *)malloc(3 * sizeof(double complex));
        };
    };

    double complex ** factor_xyz = (double complex **)malloc(kx_len * sizeof(double complex*));
    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        factor_xyz[kx_itr] = (double complex *) malloc(ky_len * sizeof(double complex));
    };


    for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
        {
            _E_inf(_kx[kx_itr], _ky[ky_itr], f, w0, k1, E0, ret_val_3);
            for(i = 0; i < 3; i++)
            {
                array[kx_itr][ky_itr][i] = ret_val_3[i];
            };
        };
    };

    x_len = (long)PyList_Size(x);
    _x = malloc(x_len * sizeof(double));
    if(_x == NULL)
    {
        printf("allocating memory for x failed\n");
    }
    for(i = 0; i < x_len; i++)
        PyArg_Parse(PyList_GetItem(x, i), "d", (_x + i));

    y_len = (long)PyList_Size(y);
    _y = malloc(y_len * sizeof(double));
    if(_y == NULL)
    {
        printf("allocating memory for y failed\n");
    }
    for(i = 0; i < y_len; i++)
        PyArg_Parse(PyList_GetItem(y, i), "d", (_y + i));

    z_len = (long)PyList_Size(z);
    _z = malloc(z_len * sizeof(double));
    if(_z == NULL)
    {
        printf("allocating memory for z failed\n");
    }
    for(i = 0; i < z_len; i++)
        PyArg_Parse(PyList_GetItem(z, i), "d", (_z + i));

    ret_dims_4d[0] = x_len;
    ret_dims_4d[1] = y_len;
    ret_dims_4d[2] = z_len;
    ret_dims_4d[3] = 3;

    arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
    outdata = (npy_complex128 *) PyArray_DATA(arr);

    for(x_itr = 0; x_itr < x_len; x_itr++)
    {
        for(y_itr = 0; y_itr < y_len; y_itr++)
        {
            for(z_itr = 0; z_itr < z_len; z_itr++)
            {
                prefactor = -I * f / (2 * M_PI) * (cos(-k1 * f) + I * sin(-k1 * f));
                for(kx_itr = 0; kx_itr < kx_len; kx_itr++)
                {
                    for(ky_itr = 0; ky_itr < ky_len; ky_itr++)
                    {
                        if(_kx[kx_itr] * _kx[kx_itr] + _ky[ky_itr] * _ky[ky_itr] < k1 * k1)
                        {
                            kz1 = sqrt(k1 * k1 - _kx[kx_itr] * _kx[kx_itr] - _ky[ky_itr] * _ky[ky_itr]);
                            factor_xyz[kx_itr][ky_itr] = prefactor * (cos(_kx[kx_itr] * _x[x_itr] + _ky[ky_itr] * _y[y_itr] + kz1 * _z[z_itr]) + I * sin(_kx[kx_itr] * _x[x_itr] + _ky[ky_itr] * _y[y_itr] + kz1 * _z[z_itr])) / kz1;
                        }
                        else
                        {
                            factor_xyz[kx_itr][ky_itr] = 0;
                        };
                    };
                };
                _dblsimps(factor_xyz, array, delta_kx, delta_ky, kx_len, ky_len, ret_val_3);
                for(i = 0; i < 3; i++)
                {
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i].real = (npy_double)creal(ret_val_3[i]);
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i].imag = (npy_double)cimag(ret_val_3[i]);
                };
            };
        };
    };

    free(_x);
    free(_y);
    free(_z);
    free(_kx);
    free(_ky);
    free(array);
    return arr;
}


static char r_s_docstring[] =
	"Integrate simpson 2d.";

static char r_p_docstring[] =
	"Integrate simpson 2d.";

static char t_s_docstring[] =
	"Integrate simpson 2d.";

static char t_p_docstring[] =
	"Integrate simpson 2d.";

static char r_s_membrane_docstring[] =
	"Integrate simpson 2d.";

static char r_p_membrane_docstring[] =
	"Integrate simpson 2d.";

static char t_s_membrane_docstring[] =
	"Integrate simpson 2d.";

static char t_p_membrane_docstring[] =
	"Integrate simpson 2d.";

static char E_inc_docstring[] =
	"Integrate simpson 2d.";

static char E_inf_docstring[] =
	"Integrate simpson 2d.";

static char E_r_inf_docstring[] =
	"Integrate simpson 2d.";

static char E_r_integrand_docstring[] =
	"Integrate simpson 2d.";

static char E_f_integrand_docstring[] =
	"Integrate simpson 2d.";

static char E_r_docstring[] =
	"Integrate simpson 2d.";

static char E_f_docstring[] =
	"Integrate simpson 2d.";

static PyMethodDef module_methods[] =
{
    {"r_s", r_s, METH_VARARGS, r_s_docstring},
    {"r_p", r_p, METH_VARARGS, r_p_docstring},
    {"t_s", t_s, METH_VARARGS, t_s_docstring},
    {"t_p", t_p, METH_VARARGS, t_p_docstring},
    {"r_s_membrane", r_s_membrane, METH_VARARGS, r_s_membrane_docstring},
    {"r_p_membrane", r_p_membrane, METH_VARARGS, r_p_membrane_docstring},
    {"t_s_membrane", t_s_membrane, METH_VARARGS, t_s_membrane_docstring},
    {"t_p_membrane", t_p_membrane, METH_VARARGS, t_p_membrane_docstring},
    {"E_inc", E_inc, METH_VARARGS, E_inc_docstring},
    {"E_inf", E_inf, METH_VARARGS, E_inf_docstring},
    {"E_r_inf", E_r_inf, METH_VARARGS, E_r_inf_docstring},
    {"E_r_integrand", E_r_integrand, METH_VARARGS, E_r_integrand_docstring},
    {"E_f_integrand", E_f_integrand, METH_VARARGS, E_f_integrand_docstring},
    {"E_r", E_r, METH_VARARGS, E_r_docstring},
    {"E_f", E_f, METH_VARARGS, E_f_docstring},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "_custom_module",     /* m_name */
        "My Function in C.",  /* m_doc */
        -1,                  /* m_size */
        module_methods,    /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };

PyMODINIT_FUNC PyInit__custom_module(void)
{
    PyObject *m = PyModule_Create(&moduledef);
    if (m == NULL)
        return NULL;

    /* Load `numpy` functionality. */
    import_array();
    return m;
};