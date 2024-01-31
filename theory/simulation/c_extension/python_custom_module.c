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
static long ret_dims_4d[4] = {0L, 0L, 0L, 0L};

static PyObject * i00(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I00(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i01(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I01(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i02(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I02(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i10(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I10(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i11(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I11(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i12(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I12(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i13(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I13(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}

static PyObject * i14(PyObject * self, PyObject * args)
{
    double rho;
    double z;
    double theta_max_;
    double k;
    double filling_factor;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "ddddd", &rho, &z, &theta_max_, &k, &filling_factor))
			return NULL;

    ret_val = I14(rho, z, theta_max_, k, filling_factor);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
	return PyComplex_FromCComplex(c);
}


static PyObject * rayleigh_length(PyObject * self, PyObject * args)
{
    double width_x;
    double width_y = 0;
    double wavelength = 1550E-9;

	if (!PyArg_ParseTuple(args, "d|dd", &width_x, &width_y, &wavelength))
			return NULL;

	return PyFloat_FromDouble(_rayleigh_length(width_x, width_y, wavelength));
}

static PyObject * width(PyObject * self, PyObject * args)
{
    double z;
    double width_x;
    double width_y = 0;
    char axis = 'x';
    double wavelength = 1550E-9;
    double rayleigh_len = 0;

	if (!PyArg_ParseTuple(args, "dd|dcdd", &z, &width_x, &width_y, &axis, &wavelength, &rayleigh_len))
			return NULL;

	return PyFloat_FromDouble(_width(z, width_x, width_y, axis, wavelength, rayleigh_len));
}

static PyObject * wavefront_radius(PyObject * self, PyObject * args)
{
    double z;
    double width_x;
    double width_y = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;

	if (!PyArg_ParseTuple(args, "dd|ddd", &z, &rayleigh_len, &width_x, &width_y, &wavelength))
			return NULL;

	return PyFloat_FromDouble(_wavefront_radius(z, rayleigh_len, width_x, width_y, wavelength));
}

static PyObject * intensity_gauss(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double width_x;
    double width_y = 0;
    double e_field = -1;
    double power = -1;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;

	if (!PyArg_ParseTuple(args, "dddd|dDddd", &x, &y, &z, &width_x, &width_y, &e_field, &power, &wavelength, &rayleigh_len))
			return NULL;

	return PyFloat_FromDouble(_intensity_gauss(x, y, z, width_x, width_y, e_field, power, wavelength, rayleigh_len));
}

static PyObject * polarizability(PyObject * self, PyObject * args)
{
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0. * I;
    double complex permittivity_medium = 1;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "|ddDd", &volume, &radius, &permittivity_particle, &permittivity_medium))
			return NULL;
    ret_val = _polarizability(volume, radius, permittivity_particle, permittivity_medium);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
    return PyComplex_FromCComplex(c);
}

static PyObject * effective_polarizability(PyObject * self, PyObject * args)
{
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    Py_complex c;
    double complex ret_val;

	if (!PyArg_ParseTuple(args, "|dddDD", &volume, &radius, &wavelength, &permittivity_particle, &permittivity_medium))
			return NULL;

    ret_val = _effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium);
    c.real = creal(ret_val);
    c.imag = cimag(ret_val);
    return PyComplex_FromCComplex(c);

}

static PyObject * gradient_force_gaussian(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double power = -1;
    double width_x;
    double width_y = 0;
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double *ret_val = malloc(3 * sizeof(double));

	if (!PyArg_ParseTuple(args, "ddddd|dddddDD", &x, &y, &z, &power, &width_x, &width_y, &volume, &radius, &wavelength, &rayleigh_len, &permittivity_particle, &permittivity_medium))
			return NULL;

	ret_dims[0] = 3;
	_gradient_force_gaussian(x, y, z, power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, ret_val);
	return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * scattering_force_gaussian(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double power = -1;
    double width_x;
    double width_y = 0;
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double *ret_val = malloc(3 * sizeof(double));

	if (!PyArg_ParseTuple(args, "ddddd|dddddDD", &x, &y, &z, &power, &width_x, &width_y, &volume, &radius, &wavelength, &rayleigh_len, &permittivity_particle, &permittivity_medium))
			return NULL;

	ret_dims[0] = 3;
	_scattering_force_gaussian(x, y, z, power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, ret_val);
	return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * total_force_gaussian(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double power = -1;
    double width_x;
    double width_y = 0;
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double *ret_val = malloc(3 * sizeof(double));

	if (!PyArg_ParseTuple(args, "ddddd|dddddDD", &x, &y, &z, &power, &width_x, &width_y, &volume, &radius, &wavelength, &rayleigh_len, &permittivity_particle, &permittivity_medium))
			return NULL;

	ret_dims[0] = 3;
	_total_force_gaussian(x, y, z, power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, ret_val);
	return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}


static PyObject * fluctuating_force(PyObject * self, PyObject * args)
{
    double damping_rate;
    double mass;
    double temperature=300;
    double dt=1;
    int size=3;
    double *ret_val;

	if (!PyArg_ParseTuple(args, "dd|ddi", &damping_rate, &mass, &temperature, &dt, &size))
			return NULL;

    ret_val = malloc(size * sizeof(double));
	ret_dims[0] = size;
	_fluctuating_force(damping_rate, mass, temperature, dt, size, ret_val);
	return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * ode_runge_kutta(PyObject * self, PyObject * args)
{
    char *problem;
    PyObject * y0 = NULL;
    PyObject * t = NULL;
    long y0_len;
    long t_len;
    long i;
    double *_y0;
    double *_t;
    double **ret_val;
    double *ret_val_cont;
    double power = -1;
    double width_x;
    double width_y = 0;
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double damping_rate;
    double mass;
    double temperature=300;
    double focal_distance=0;
    double NA=0;
    double e_field=1;
    PyObject * jones_vector_py;
    double complex jones_vector[2] = {1, 0};
    double n_1=1;
    double n_2=1;
    double filling_factor=-1;
    double aperture_radius=1;
    double width_inc=1;
    double delta=1E-10;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

	if (!PyArg_ParseTuple(args, "sOOdddd|ddddddDDdddOdddddididd", &problem, &y0, &t, &power, &width_x, &damping_rate, &mass, &width_y, &volume, &radius, &wavelength, &temperature, &rayleigh_len, &permittivity_particle, &permittivity_medium, &focal_distance, &NA, &e_field, &jones_vector_py, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;
    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));
    y0_len = (long)PyList_Size(y0);
    _y0 = malloc(y0_len * sizeof(double));
    if(_y0 == NULL)
    {
        printf("allocating memory for y0 failed\n");
    }
    for(i = 0; i < y0_len; i++)
        PyArg_Parse(PyList_GetItem(y0, i), "d", (_y0 + i));

    t_len = (long)PyList_Size(t);
    _t = malloc(t_len * sizeof(double));
    if(_t == NULL)
    {
        printf("allocating memory for t0 failed\n");
    }

    for(i = 0; i < t_len; i++)
        PyArg_Parse(PyList_GetItem(t, i), "d", (_t + i));

    ret_dims_2d[0] = y0_len;
    ret_dims_2d[1] = t_len;

    ret_val = malloc( y0_len  * sizeof *ret_val );
    if(ret_val == NULL)
    {
        printf("allocating memory for ret_val failed\n");
    }

    ret_val_cont = malloc(y0_len * t_len * sizeof(double));
    if(ret_val_cont == NULL)
    {
        printf("allocating memory for ret_val_cont failed\n");
    }

    for (i = 0; i < y0_len; i++)
        ret_val[i] = ret_val_cont + i * t_len;

    if(strcmp(problem, "initial_value_optical_gaussian") == 0)
    {
        _ode_runge_kutta(0, _y0, _t, power, width_x, width_y, volume, radius, wavelength, rayleigh_len,
        permittivity_particle, permittivity_medium,focal_distance, NA, e_field, jones_vector, n_1, n_2,
        filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, ret_dims_2d, 0, surface, d_surf, r_surf, ret_val);
    }
    else if(strcmp(problem, "initial_value_optical") == 0)
    {
        _ode_runge_kutta(1, _y0, _t, power, width_x, width_y, volume, radius, wavelength, rayleigh_len,
        permittivity_particle, permittivity_medium,focal_distance, NA, e_field, jones_vector, n_1, n_2,
        filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, ret_dims_2d, field_kind, surface, d_surf, r_surf, ret_val);
    };

    free(_y0);
    free(_t);
	return PyArray_SimpleNewFromData(2, ret_dims_2d, NPY_DOUBLE, ret_val_cont);
}

static PyObject * ode_euler(PyObject * self, PyObject * args)
{
    char *problem;
    PyObject * y0 = NULL;
    PyObject * t = NULL;
    long y0_len;
    long t_len;
    long i;
    double *_y0;
    double *_t;
    double **ret_val;
    double *ret_val_cont;
    double power = -1;
    double width_x;
    double width_y = 0;
    double volume = 0;
    double radius = 0;
    double wavelength = 1550E-9;
    double rayleigh_len = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double damping_rate;
    double mass;
    double temperature=300;
    double focal_distance=0;
    double NA=0;
    double e_field=1;
    PyObject * jones_vector_py;
    double complex jones_vector[2] = {1, 0};
    double n_1=1;
    double n_2=1;
    double filling_factor=-1;
    double aperture_radius=1;
    double width_inc=1;
    double delta=1E-10;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;


	if (!PyArg_ParseTuple(args, "sOOdddd|ddddddDDdddOdddddididd", &problem, &y0, &t, &power, &width_x, &damping_rate, &mass, &width_y, &volume, &radius, &wavelength, &temperature, &rayleigh_len, &permittivity_particle, &permittivity_medium, &focal_distance, &NA, &e_field, &jones_vector_py, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

    y0_len = (long)PyList_Size(y0);
    _y0 = malloc(y0_len * sizeof(double));
    if(_y0 == NULL)
    {
        printf("allocating memory for y0 failed\n");
    }
    for(i = 0; i < y0_len; i++)
        PyArg_Parse(PyList_GetItem(y0, i), "d", (_y0 + i));

    t_len = (long)PyList_Size(t);
    _t = malloc(t_len * sizeof(double));
    if(_t == NULL)
    {
        printf("allocating memory for t0 failed\n");
    }

    for(i = 0; i < t_len; i++)
        PyArg_Parse(PyList_GetItem(t, i), "d", (_t + i));

    ret_dims_2d[0] = y0_len;
    ret_dims_2d[1] = t_len;

    ret_val = malloc( y0_len  * sizeof *ret_val );
    if(ret_val == NULL)
    {
        printf("allocating memory for ret_val failed\n");
    }

    ret_val_cont = malloc(y0_len * t_len * sizeof(double));
    if(ret_val_cont == NULL)
    {
        printf("allocating memory for ret_val_cont failed\n");
    }

    for (i = 0; i < y0_len; i++)
        ret_val[i] = ret_val_cont + i * t_len;

    if(strcmp(problem, "initial_value_optical_gaussian") == 0)
    {
        _ode_euler(0, _y0, _t, power, width_x, width_y, volume, radius, wavelength, rayleigh_len,
        permittivity_particle, permittivity_medium,focal_distance, NA, e_field, jones_vector, n_1, n_2,
        filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, ret_dims_2d, 0, surface, d_surf, r_surf, ret_val);
    }
    else if(strcmp(problem, "initial_value_optical") == 0)
    {
        _ode_euler(1, _y0, _t, power, width_x, width_y, volume, radius, wavelength, rayleigh_len,
        permittivity_particle, permittivity_medium,focal_distance, NA, e_field, jones_vector, n_1, n_2,
        filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, ret_dims_2d, field_kind, surface, d_surf, r_surf, ret_val);
    };

    free(_y0);
    free(_t);
	return PyArray_SimpleNewFromData(2, ret_dims_2d, NPY_DOUBLE, ret_val_cont);
}

static PyObject * fields_00(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double focal_distance;
    double NA;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    int field = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

    double complex ret_val_3[3];
    double complex ret_val_6[6];
    int i;
    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;

	if (!PyArg_ParseTuple(args, "ddddd|ddOddddddiidd", &x, &y, &z, &focal_distance, &NA, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field, &surface, &d_surf, &r_surf))
			return NULL;


    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));


    if(field != 2)
    {
    	ret_dims[0] = 3;
	    _fields_00(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_3);
	    arr = PyArray_SimpleNew(1, ret_dims, PyArray_COMPLEX128);
	    outdata = (npy_complex128 *) PyArray_DATA(arr);

	    for(i = 0; i < 3; i++)
	    {
            outdata[i].real = (npy_double)creal(ret_val_3[i]);
            outdata[i].imag = (npy_double)cimag(ret_val_3[i]);

	    };

	    return arr;
    }
    else
    {
        ret_dims[0] = 6;
	    _fields_00(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_6);
        arr = PyArray_SimpleNew(1, ret_dims, PyArray_COMPLEX128);
	    outdata = (npy_complex128 *) PyArray_DATA(arr);
	    for(i = 0; i < 6; i++)
	    {
            outdata[i].real = (npy_double)creal(ret_val_6[i]);
            outdata[i].imag = (npy_double)cimag(ret_val_6[i]);
	    };
	    return arr;
    }
}

static PyObject * fields_00_vect(PyObject * self, PyObject * args)
{
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

    double focal_distance;
    double NA;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    int field = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

    double complex ret_val_3[3];
    double complex ret_val_6[6];
    int i;
    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;

	if (!PyArg_ParseTuple(args, "OOOdd|ddOddddddiidd", &x, &y, &z, &focal_distance, &NA, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

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

    if(field != 2)
    {
        ret_dims_4d[3] = 3;

        arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
        outdata = (npy_complex128 *) PyArray_DATA(arr);

        for(x_itr = 0; x_itr < x_len; x_itr++)
        {
            for(y_itr = 0; y_itr < y_len; y_itr++)
            {
                for(z_itr = 0; z_itr < z_len; z_itr++)
                {
                    _fields_00(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_3);
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

    return arr;
    }
    else
    {
        ret_dims_4d[3] = 6;

        arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
        outdata = (npy_complex128 *) PyArray_DATA(arr);

        for(x_itr = 0; x_itr < x_len; x_itr++)
        {
            for(y_itr = 0; y_itr < y_len; y_itr++)
            {
                for(z_itr = 0; z_itr < z_len; z_itr++)
                {
                    _fields_00(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_6);
                    for(i = 0; i < 6; i++)
                    {
                        (outdata + x_itr * x_len + y_itr * y_len + z_itr * z_len)[i].real = (npy_double)creal(ret_val_6[i]);
                        (outdata + x_itr * x_len + y_itr * y_len + z_itr * z_len)[i].imag = (npy_double)cimag(ret_val_6[i]);
                    };
                };
            };
        };

    free(_x);
    free(_y);
    free(_z);

    return arr;
    }
}

static PyObject * fields_doughnut_rad(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double focal_distance;
    double NA;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    int field = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;
    double complex ret_val_3[3];
    double complex ret_val_6[6];
    int i;
    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;

	if (!PyArg_ParseTuple(args, "ddddd|ddOddddddiidd", &x, &y, &z, &focal_distance, &NA, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));


    if(field != 2)
    {
    	ret_dims[0] = 3;
	    _fields_doughnut_rad(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_3);
	    arr = PyArray_SimpleNew(1, ret_dims, PyArray_COMPLEX128);
	    outdata = (npy_complex128 *) PyArray_DATA(arr);
	    for(i = 0; i < 3; i++)
	    {
            outdata[i].real = (npy_double)creal(ret_val_3[i]);
            outdata[i].imag = (npy_double)cimag(ret_val_3[i]);
	    };
	    return arr;
    }
    else
    {
        ret_dims[0] = 6;
	    _fields_doughnut_rad(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_6);
        arr = PyArray_SimpleNew(1, ret_dims, PyArray_COMPLEX128);
	    outdata = (npy_complex128 *) PyArray_DATA(arr);
	    for(i = 0; i < 6; i++)
	    {
            outdata[i].real = (npy_double)creal(ret_val_6[i]);
            outdata[i].imag = (npy_double)cimag(ret_val_6[i]);
	    };
	    return arr;
    }
}

static PyObject * fields_doughnut_rad_vect(PyObject * self, PyObject * args)
{
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

    double focal_distance;
    double NA;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    int field = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;
    double complex ret_val_3[3];
    double complex ret_val_6[6];
    int i;
    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;

	if (!PyArg_ParseTuple(args, "OOOdd|ddOddddddiidd", &x, &y, &z, &focal_distance, &NA, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

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

    if(field != 2)
    {
        ret_dims_4d[3] = 3;

        arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
        outdata = (npy_complex128 *) PyArray_DATA(arr);

        for(x_itr = 0; x_itr < x_len; x_itr++)
        {
            for(y_itr = 0; y_itr < y_len; y_itr++)
            {
                for(z_itr = 0; z_itr < z_len; z_itr++)
                {
                    _fields_doughnut_rad(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_3);
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

    return arr;
    }
    else
    {
        ret_dims_4d[3] = 6;

        arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_COMPLEX128);
        outdata = (npy_complex128 *) PyArray_DATA(arr);

        for(x_itr = 0; x_itr < x_len; x_itr++)
        {
            for(y_itr = 0; y_itr < y_len; y_itr++)
            {
                for(z_itr = 0; z_itr < z_len; z_itr++)
                {
                    _fields_doughnut_rad(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, surface, d_surf, r_surf, ret_val_6);
                    for(i = 0; i < 6; i++)
                    {
                        (outdata + x_itr * x_len + y_itr * y_len + z_itr * z_len)[i].real = (npy_double)creal(ret_val_6[i]);
                        (outdata + x_itr * x_len + y_itr * y_len + z_itr * z_len)[i].imag = (npy_double)cimag(ret_val_6[i]);
                    };
                };
            };
        };

    free(_x);
    free(_y);
    free(_z);

    return arr;
    }
}


static PyObject * gradient_force(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double *ret_val = malloc(3 * sizeof(double));
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

	if (!PyArg_ParseTuple(args, "ddddd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));


    _gradient_force(x, y, z, focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);

    ret_dims[0] = 3;
    return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * gradient_force_vect(PyObject * self, PyObject * args)
{
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

    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double ret_val[3];
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;
    PyObject *arr;
    npy_double *outdata;
    npy_double c;

	if (!PyArg_ParseTuple(args, "OOOdd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

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

    arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_DOUBLE);
    outdata = (npy_double *) PyArray_DATA(arr);

    for(x_itr = 0; x_itr < x_len; x_itr++)
    {
        for(y_itr = 0; y_itr < y_len; y_itr++)
        {
            for(z_itr = 0; z_itr < z_len; z_itr++)
            {
                _gradient_force(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);
                for(i = 0; i < 3; i++)
                {
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i] = (npy_double) ret_val[i];
                };
            };
        };
    };

    free(_x);
    free(_y);
    free(_z);

    return arr;
}

static PyObject * scattering_force(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double *ret_val = malloc(3 * sizeof(double));
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;


	if (!PyArg_ParseTuple(args, "ddddd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));


    _scattering_force(x, y, z, focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);

    ret_dims[0] = 3;
    return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * scattering_force_vect(PyObject * self, PyObject * args)
{
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

    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double ret_val[3];
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

    PyObject *arr;
    npy_double *outdata;
    npy_double c;

	if (!PyArg_ParseTuple(args, "OOOdd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

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

    arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_DOUBLE);
    outdata = (npy_double *) PyArray_DATA(arr);

    for(x_itr = 0; x_itr < x_len; x_itr++)
    {
        for(y_itr = 0; y_itr < y_len; y_itr++)
        {
            for(z_itr = 0; z_itr < z_len; z_itr++)
            {
                _scattering_force(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);
                for(i = 0; i < 3; i++)
                {
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i] = (npy_double) ret_val[i];
                };
            };
        };
    };

    free(_x);
    free(_y);
    free(_z);

    return arr;
}

static PyObject * total_force(PyObject * self, PyObject * args)
{
    double x;
    double y;
    double z;
    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double *ret_val = malloc(3 * sizeof(double));
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;
    PyObject *arr;
    npy_complex128 *outdata;
    npy_complex128 c;


	if (!PyArg_ParseTuple(args, "ddddd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));


    _total_force(x, y, z, focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);

    ret_dims[0] = 3;
    return PyArray_SimpleNewFromData(1, ret_dims, NPY_DOUBLE, ret_val);
}

static PyObject * total_force_vect(PyObject * self, PyObject * args)
{
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

    double focal_distance;
    double NA;
    double volume = 0;
    double radius = 0;
    double complex permittivity_particle = 2.101 + 0 * I;
    double complex permittivity_medium = 1;
    double e_field = -1;
    double power = -1;
    double complex jones_vector[2] = {1, 0};
    PyObject * jones_vector_py;
    double wavelength = 1550E-9;
    double n_1 = 1.;
    double n_2 = 1.;
    double filling_factor = -1;
    double aperture_radius = 1;
    double width_inc = 1;
    double delta = 1E-10;
    double ret_val[3];
    int i;
    int field_kind = 0;
    int surface = 0;
    double d_surf = 0;
    double r_surf = 0;

    PyObject *arr;
    npy_double *outdata;
    npy_double c;

	if (!PyArg_ParseTuple(args, "OOOdd|ddDDddOddddddididd", &x, &y, &z, &focal_distance, &NA, &volume, &radius, &permittivity_particle, &permittivity_medium, &e_field, &power, &jones_vector_py, &wavelength, &n_1, &n_2, &filling_factor, &aperture_radius, &width_inc, &field_kind, &delta, &surface, &d_surf, &r_surf))
			return NULL;

    for(i = 0; i < 2; i++)
        PyArg_Parse(PyList_GetItem(jones_vector_py, i), "D", (jones_vector + i));

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

    arr = PyArray_SimpleNew(4, ret_dims_4d, PyArray_DOUBLE);
    outdata = (npy_double *) PyArray_DATA(arr);

    for(x_itr = 0; x_itr < x_len; x_itr++)
    {
        for(y_itr = 0; y_itr < y_len; y_itr++)
        {
            for(z_itr = 0; z_itr < z_len; z_itr++)
            {
                _total_force(_x[x_itr], _y[y_itr], _z[z_itr], focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, ret_val);
                for(i = 0; i < 3; i++)
                {
                    (outdata + x_itr * y_len * z_len * 3 + y_itr * z_len * 3 + z_itr * 3)[i] = (npy_double) ret_val[i];
                };
            };
        };
    };

    free(_x);
    free(_y);
    free(_z);

    return arr;
}


static char rayleigh_length_docstring[] =
	"Calculate the Rayleigh Length of a Gaussian Beam.";

static char width_docstring[] =
	"Calculate the width of a Gaussian Beam.";

static char wavefront_radius_docstring[] =
	"Calculate the width of a Gaussian Beam.";

static char polarizability_docstring[] =
	"Calculate the polarizability of a spherical particle.";

static char effective_polarizability_docstring[] =
	"Calculate the effective polarizability of a spherical particle.";

static char intensity_gauss_docstring[] =
	"Calculate the effective polarizability of a spherical particle.";

static char gradient_force_gaussian_docstring[] =
	"Calculate the gradient force within gaussian approximation.";

static char scattering_force_gaussian_docstring[] =
	"Calculate the scattering force within gaussian approximation.";

static char total_force_gaussian_docstring[] =
	"Calculate the total force within gaussian approximation.";

static char fluctuating_force_docstring[] =
	"Calculate the total force within gaussian approximation.";

static char ode_runge_kutta_docstring[] =
	"Calculate the total force within gaussian approximation.";

static char ode_euler_docstring[] =
	"Calculate the total force within gaussian approximation.";

static char i00_docstring[] =
	"Integral 00.";
static char i01_docstring[] =
	"Integral 01.";
static char i02_docstring[] =
	"Integral 02.";
static char i10_docstring[] =
	"Integral 10.";
static char i11_docstring[] =
	"Integral 11.";
static char i12_docstring[] =
	"Integral 12.";
static char i13_docstring[] =
	"Integral 13.";
static char i14_docstring[] =
	"Integral 14.";

static char fields_00_docstring[] =
	"Calculate the 00 order electric and magnetic field of a focussed beam.";

static char fields_00_vect_docstring[] =
	"Calculate the 00 order electric and magnetic field of a focussed beam.";

static char fields_doughnut_rad_docstring[] =
	"Calculate the 00 order electric and magnetic field of a focussed beam.";

static char fields_doughnut_rad_vect_docstring[] =
	"Calculate the 00 order electric and magnetic field of a focussed beam.";

static char gradient_force_docstring[] =
	"Calculate the gradient force of a focussed gaussian beam beyond paraxial approximation.";

static char scattering_force_docstring[] =
	"Calculate the scattering force of a focussed gaussian beam beyond paraxial approximation.";

static char total_force_docstring[] =
	"Calculate the total optical force of a focussed gaussian beam beyond paraxial approximation.";

static char gradient_force_vect_docstring[] =
	"Calculate the gradient force of a focussed gaussian beam beyond paraxial approximation.";

static char scattering_force_vect_docstring[] =
	"Calculate the scattering force of a focussed gaussian beam beyond paraxial approximation.";

static char total_force_vect_docstring[] =
	"Calculate the total optical force of a focussed gaussian beam beyond paraxial approximation.";

static PyMethodDef module_methods[] =
{
    {"rayleigh_length", rayleigh_length, METH_VARARGS, rayleigh_length_docstring},
    {"width", width, METH_VARARGS, width_docstring},
    {"wavefront_radius", wavefront_radius, METH_VARARGS, wavefront_radius_docstring},
    {"polarizability", polarizability, METH_VARARGS, polarizability_docstring},
    {"effective_polarizability", effective_polarizability, METH_VARARGS, effective_polarizability_docstring},
    {"intensity_gauss", intensity_gauss, METH_VARARGS, intensity_gauss_docstring},
    {"gradient_force_gaussian", gradient_force_gaussian, METH_VARARGS, gradient_force_gaussian_docstring},
    {"scattering_force_gaussian", scattering_force_gaussian, METH_VARARGS, scattering_force_gaussian_docstring},
    {"total_force_gaussian", total_force_gaussian, METH_VARARGS, total_force_gaussian_docstring},
    {"fluctuating_force", fluctuating_force, METH_VARARGS, fluctuating_force_docstring},
    {"ode_runge_kutta", ode_runge_kutta, METH_VARARGS, ode_runge_kutta_docstring},
    {"ode_euler", ode_euler, METH_VARARGS, ode_euler_docstring},
    {"i00", i00, METH_VARARGS, i00_docstring},
    {"i01", i01, METH_VARARGS, i01_docstring},
    {"i02", i02, METH_VARARGS, i02_docstring},
    {"i10", i10, METH_VARARGS, i10_docstring},
    {"i11", i11, METH_VARARGS, i11_docstring},
    {"i12", i12, METH_VARARGS, i12_docstring},
    {"i13", i13, METH_VARARGS, i13_docstring},
    {"i14", i14, METH_VARARGS, i14_docstring},
    {"fields_00", fields_00, METH_VARARGS, fields_00_docstring},
    {"fields_00_vect", fields_00_vect, METH_VARARGS, fields_00_vect_docstring},
    {"fields_doughnut_rad", fields_doughnut_rad, METH_VARARGS, fields_doughnut_rad_docstring},
    {"fields_doughnut_rad_vect", fields_doughnut_rad_vect, METH_VARARGS, fields_doughnut_rad_vect_docstring},
    {"gradient_force", gradient_force, METH_VARARGS, gradient_force_docstring},
    {"scattering_force", scattering_force, METH_VARARGS, scattering_force_docstring},
    {"total_force", total_force, METH_VARARGS, total_force_docstring},
    {"gradient_force_vect", gradient_force_vect, METH_VARARGS, gradient_force_vect_docstring},
    {"scattering_force_vect", scattering_force_vect, METH_VARARGS, scattering_force_vect_docstring},
    {"total_force_vect", total_force_vect, METH_VARARGS, total_force_vect_docstring},
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