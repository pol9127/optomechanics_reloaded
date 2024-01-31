#ifndef __CUSTOMMODULE_CONST__
#define __CUSTOMMODULE_CONST__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>

#include "gsl_const_mks.c"
#include "numeric_module.c"

static int phase = 0;
static double V1, V2, S;
static time_t t;
static int seed = 1;

static double _gaussrand()
{
	double X;
	if(seed == 1)
	{
	    srand((unsigned) time(&t));
	    seed = 0;
	};

	if(phase == 0) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
			} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	return X;
}

static double _rayleigh_length(double width_x, double width_y, double wavelength)
{
	if (width_y == 0)
	    width_y = width_x;

    return M_PI * width_x * width_y / wavelength;
}

static double _width(double z, double width_x, double width_y, char axis, double wavelength, double rayleigh_len)
{
    double width_0;

	if (width_y == 0)
	    width_y = width_x;

	if (rayleigh_len == 0)
	    rayleigh_len = _rayleigh_length(width_x, width_y, wavelength);

    if (axis == 'x')
        width_0 = width_x;
    else if (axis == 'y')
        width_0 = width_y;
    else
        width_0 = 0.;

    return width_0 * sqrt(1. + (z / rayleigh_len) * (z / rayleigh_len));
}

static double _wavefront_radius(double z, double rayleigh_len, double width_x, double width_y, double wavelength)
{
    double radius;

	if (width_y == 0)
	    width_y = width_x;

    if (rayleigh_len == 0)
	    rayleigh_len = _rayleigh_length(width_x, width_y, wavelength);


    if (z != 0)
    {
        radius = z * (1 + (rayleigh_len /  z) * (rayleigh_len /  z));
    }
    else
    {
        radius = INFINITY;
    };

    return radius;
}


static double _intensity_gauss(double x, double y, double z, double width_x, double width_y, double e_field, double power, double wavelength, double rayleigh_len)
{
    double width2;
    double width_x_;
    double width_y_;
    double exponent;

    if (width_y == 0)
	    width_y = width_x;

	if (rayleigh_len == 0)
	    rayleigh_len = _rayleigh_length(width_x, width_y, wavelength);

	if (power == -1)
	    power = GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * M_PI * width_x * width_y * e_field * conj(e_field) / 4.;

    width2 = _width(z, width_x, width_y, 'x', wavelength, rayleigh_len) * _width(z, width_x, width_y, 'y', wavelength, rayleigh_len);
    width_x_ = _width(z, width_x, width_y, 'x', wavelength, rayleigh_len);
    width_y_ = _width(z, width_x, width_y, 'y', wavelength, rayleigh_len);

    exponent = -2. * ((x * x) / (width_x_ * width_x_) + (y * y) / (width_y_ * width_y_));

    return power / (M_PI * width2 / 2.) * exp(exponent);
}

static double complex _polarizability(double volume, double radius, double complex permittivity_particle, double complex permittivity_medium)
{
    if (volume == 0)
        volume = 4. / 3. * M_PI * radius * radius * radius;

    return 3. * volume * GSL_CONST_MKS_VACUUM_PERMITTIVITY * (
        permittivity_particle - permittivity_medium) / (
        permittivity_particle + 2 * permittivity_medium);
}

static double complex _effective_polarizability(double volume, double radius, double wavelength, double complex permittivity_particle, double complex permittivity_medium)
{
	double k = 2. * M_PI / wavelength;
	double complex alpha = _polarizability(volume, radius, permittivity_particle, permittivity_medium);

    return alpha / (1. - 1. * I * k * k * k * alpha / (6. * M_PI * GSL_CONST_MKS_VACUUM_PERMITTIVITY));
}

static void _gradient_force_gaussian(double x, double y, double z, double power, double width_x, double width_y, double volume, double radius, double wavelength,
	double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium, double* ret_val)
{
    double prefactor;

	if (width_y == 0)
	    width_y = width_x;

	if (rayleigh_len == 0)
	    rayleigh_len = _rayleigh_length(width_x, width_y, wavelength);

    if (volume == 0)
        volume = 4. / 3. * M_PI * radius * radius * radius;


    prefactor = -1. * creal(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium));
    prefactor *= _intensity_gauss(x, y, z, width_x, width_y, 0., power, wavelength, rayleigh_len) * 2.;

    prefactor /= GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY;
    ret_val[0] = prefactor * x * (rayleigh_len * rayleigh_len) / ((width_x * width_x) * ((z * z) + (rayleigh_len * rayleigh_len)));
    ret_val[1] = prefactor * y * (rayleigh_len * rayleigh_len) / ((width_y * width_y) * ((z * z) + (rayleigh_len * rayleigh_len)));

    ret_val[2] = prefactor * z * (((z / rayleigh_len) * (z / rayleigh_len)) + (1. - 2. * (x * x) / (width_x * width_x) - 2. * (y * y) / (width_y * width_y)))
    * (rayleigh_len * rayleigh_len) / (2. * (((z * z) + (rayleigh_len * rayleigh_len)) * ((z * z) + (rayleigh_len * rayleigh_len))));
}

static void _scattering_force_gaussian(double x, double y, double z, double power, double width_x, double width_y, double volume, double radius, double wavelength,
	double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium, double* ret_val)
{
    double prefactor;
    double k;
    double wavefront_radius;

	if (width_y == 0)
	    width_y = width_x;

	if (rayleigh_len == 0)
	    rayleigh_len = _rayleigh_length(width_x, width_y, wavelength);

    if (volume == 0)
        volume = 4. / 3. * M_PI * radius * radius * radius;

    k = 2. * M_PI / wavelength;

    prefactor = cimag(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium)) / 2.;
    prefactor *= _intensity_gauss(x, y, z, width_x, width_y, 0., power, wavelength, rayleigh_len) * 2.;
    prefactor /= GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY;
    prefactor *= k;

    wavefront_radius = _wavefront_radius(z, rayleigh_len, width_x, width_y, wavelength);

    if (z != 0)
    {
        ret_val[0] = x / wavefront_radius * prefactor;
        ret_val[1] = y / wavefront_radius * prefactor;
        ret_val[2] = 1 + ((x * x) + (y * y)) * (rayleigh_len * rayleigh_len) / ((z * z) * (wavefront_radius * wavefront_radius));
        ret_val[2] -= ((x * x) + (y * y) + 2 * z * rayleigh_len) / (2 * z * wavefront_radius);
        ret_val[2] *= prefactor;
    }
    else
    {
        ret_val[0] = 0.;
        ret_val[1] = 0.;
        ret_val[2] = prefactor;
    };
}

static void _total_force_gaussian(double x, double y, double z, double power, double width_x, double width_y, double volume, double radius, double wavelength,
	double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium, double* ret_val)
{
    double *gradient_val = malloc(3 * sizeof(double));
    double *scattering_val = malloc(3 * sizeof(double));
    int i;

    _gradient_force_gaussian(x, y, z, power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, gradient_val);
    _scattering_force_gaussian(x, y, z, power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, scattering_val);

    for (i = 0; i < 3; i++)
    {
        ret_val[i] = gradient_val[i] + scattering_val[i];
    };
    free(gradient_val);
    free(scattering_val);
}

static void _fluctuating_force(double damping_rate, double mass, double temperature, double dt, int size, double* ret_val)
{
    int i;

    for (i = 0; i < size; i++)
    {
        ret_val[i] = sqrt(2. * GSL_CONST_MKS_BOLTZMANN * temperature * damping_rate * mass / dt) * _gaussrand();
    };
}

static void _gradient_force(double x, double y, double z, double focal_distance, double NA, double volume, double radius, double complex permittivity_particle, double complex permittivity_medium, double e_field, double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, double delta, int field_kind, int surface, double d_surf, double r_surf, double *ret_val)
{
    double complex grad_e_x[3], grad_e_y[3], grad_e_z[3], field[3], field_conj[3], ret_tmp[3] = {0., 0., 0.};
    int i;
    double polarizability_real = creal(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium));

    if(field_kind == 0)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_00(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    }
    else if (field_kind == 1)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_doughnut_rad(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    };

    for(i = 0; i < 3; i++)
    {
        field_conj[i] = conj(field[i]);
        ret_tmp[i] = 0.;
    };
    for(i = 0; i < 3; i++)
    {
        ret_tmp[0] += field_conj[i] * grad_e_x[i];
        ret_tmp[1] += field_conj[i] * grad_e_y[i];
        ret_tmp[2] += field_conj[i] * grad_e_z[i];
    };
    for(i = 0; i < 3; i++)
    {
        ret_val[i] = polarizability_real / 2. * creal(ret_tmp[i]);
    };
}

static void _scattering_force(double x, double y, double z, double focal_distance, double NA, double volume, double radius, double complex permittivity_particle, double complex permittivity_medium, double e_field, double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, double delta, int field_kind, int surface, double d_surf, double r_surf, double *ret_val)
{
    double complex grad_e_x[3], grad_e_y[3], grad_e_z[3], field[3], field_conj[3], ret_tmp[3] = {0., 0., 0.};
    int i;
    double polarizability_imag = cimag(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium));


    if(field_kind == 0)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_00(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    }
    else if (field_kind == 1)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_doughnut_rad(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    };


    for(i = 0; i < 3; i++)
    {
        field_conj[i] = conj(field[i]);
        ret_tmp[i] = 0.;
    };
    for(i = 0; i < 3; i++)
    {
        ret_tmp[0] += field_conj[i] * grad_e_x[i];
        ret_tmp[1] += field_conj[i] * grad_e_y[i];
        ret_tmp[2] += field_conj[i] * grad_e_z[i];
    };

    for(i = 0; i < 3; i++)
    {
        ret_val[i] = polarizability_imag / 2. * cimag(ret_tmp[i]);
    };
}

static void _total_force(double x, double y, double z, double focal_distance, double NA, double volume, double radius, double complex permittivity_particle, double complex permittivity_medium, double e_field, double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, double delta, int field_kind, int surface, double d_surf, double r_surf, double *ret_val)
{
    double complex grad_e_x[3], grad_e_y[3], grad_e_z[3], field[3], field_conj[3], ret_tmp[3] = {0., 0., 0.};
    int i;
    double polarizability_real = creal(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium));
    double polarizability_imag = cimag(_effective_polarizability(volume, radius, wavelength, permittivity_particle, permittivity_medium));

    if(field_kind == 0)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_00(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    }
    else if (field_kind == 1)
    {
    first_derivative_efields(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, grad_e_x, grad_e_y, grad_e_z, field_kind, delta, surface, d_surf, r_surf);
    _fields_doughnut_rad(x, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, field);
    };

    for(i = 0; i < 3; i++)
    {
        field_conj[i] = conj(field[i]);
        ret_tmp[i] = 0.;
    };
    for(i = 0; i < 3; i++)
    {
        ret_tmp[0] += field_conj[i] * grad_e_x[i];
        ret_tmp[1] += field_conj[i] * grad_e_y[i];
        ret_tmp[2] += field_conj[i] * grad_e_z[i];
    };
    for(i = 0; i < 3; i++)
    {
        ret_val[i] = 0.5 * (polarizability_real * creal(ret_tmp[i]) + polarizability_imag * cimag(ret_tmp[i]));
    };
}

static void _runge_kutta_func_0(double* position_velocity_vector, double t, double dt, double power, double width_x, double width_y, double volume, double radius, double wavelength,
	double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium, double damping_rate, double mass, double temperature, double* ret_val)
 {
    double * tot_force_gaussian = malloc(3 * sizeof(double));
    double * fluct_force = malloc(3 * sizeof(double));
    int i;

    _total_force_gaussian(position_velocity_vector[0], position_velocity_vector[1], position_velocity_vector[2], power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, tot_force_gaussian);
    _fluctuating_force(damping_rate, mass, temperature, dt, 3, fluct_force);

    for (i=0; i<3; i++)
    {
        ret_val[i] = position_velocity_vector[i + 3];
//        ret_val[i] = tot_force_gaussian[i];
    };
    for (i=3; i<6; i++)
    {
        ret_val[i] = -1. * damping_rate * position_velocity_vector[i] + (tot_force_gaussian[i - 3] + fluct_force[i - 3]) / mass;
//        ret_val[i] = fluct_force[i - 3];

    };

    free(tot_force_gaussian);
    free(fluct_force);
}

static void _runge_kutta_func_1(double* position_velocity_vector, double t, double dt, double focal_distance, double NA,
double volume, double radius, double complex permittivity_particle, double complex permittivity_medium, double e_field,
double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius,
double width_inc, double delta, double damping_rate, double mass, double temperature, int field_kind, int surface, double d_surf, double r_surf, double* ret_val)
 {
    double * tot_force = malloc(3 * sizeof(double));
    double * fluct_force = malloc(3 * sizeof(double));
    int i;

    _total_force(position_velocity_vector[0], position_velocity_vector[1], position_velocity_vector[2], focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, field_kind, surface, d_surf, r_surf, tot_force);
    _fluctuating_force(damping_rate, mass, temperature, dt, 3, fluct_force);

    for (i=0; i<3; i++)
    {
        ret_val[i] = position_velocity_vector[i + 3];
    };
    for (i=3; i<6; i++)
    {
        ret_val[i] = -1. * damping_rate * position_velocity_vector[i] + (tot_force[i - 3] + fluct_force[i - 3]) / mass;

    };

    free(tot_force);
    free(fluct_force);
}


static void _ode_runge_kutta(int func, double* y0, double* t, double power, double width_x, double width_y, double volume,
double radius, double wavelength,double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium,
double focal_distance, double NA, double e_field, double complex *jones_vector, double n_1, double n_2, double filling_factor,
double aperture_radius, double width_inc, double delta, double damping_rate, double mass, double temperature, long* ret_dims_2d, int field_kind, int surface, double d_surf, double r_surf, double** ret_val)
{
    long i, j;
    double dt;
    double k1[6], k2[6], k3[6], k4[6], y[6], y_k[6];


    for (i = 0; i < ret_dims_2d[0]; i++)
    {
        ret_val[i][0] = y0[i];
    };


    for (j = 1; j < ret_dims_2d[1]; j++)
        {
            dt = t[j] - t[j - 1];
            for (i = 0; i < ret_dims_2d[0]; i++)
            {
                y[i] = ret_val[i][j - 1];
            };

            if(func==0)
            {

                _runge_kutta_func_0(y, t[j - 1], dt / 4., power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, damping_rate, mass, temperature, k1);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k1[i] / 2.;
                };
                _runge_kutta_func_0(y_k, t[j - 1] + dt / 2., dt / 4., power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, damping_rate, mass, temperature, k2);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k2[i] / 2.;
                };
                _runge_kutta_func_0(y_k, t[j - 1] + dt / 2., dt / 4., power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, damping_rate, mass, temperature, k3);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k3[i];
                };
                _runge_kutta_func_0(y_k, t[j - 1] + dt, dt / 4., power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, damping_rate, mass, temperature, k4);
            }
            else if(func==1)
            {

                _runge_kutta_func_1(y, t[j - 1], dt / 4., focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, field_kind, surface, d_surf, r_surf, k1);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k1[i] / 2.;
                };
                _runge_kutta_func_1(y_k, t[j - 1] + dt / 2., dt / 4., focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, field_kind, surface, d_surf, r_surf, k2);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k2[i] / 2.;
                };
                _runge_kutta_func_1(y_k, t[j - 1] + dt / 2., dt / 4., focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, field_kind, surface, d_surf, r_surf, k3);
                for (i = 0; i < ret_dims_2d[0]; i++)
                {
                    y_k[i] = y[i] + dt * k3[i];
                };
                _runge_kutta_func_1(y_k, t[j - 1] + dt, dt / 4., focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, field_kind, surface, d_surf, r_surf, k4);
            };


            for (i = 0; i < ret_dims_2d[0]; i++)
            {
                ret_val[i][j] = y[i] + dt / 6. * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
//                ret_val[i][j] = y[j - 1];
            };
        };

}

static void _ode_euler(int func, double* y0, double* t, double power, double width_x, double width_y, double volume,
double radius, double wavelength,double rayleigh_len, double complex permittivity_particle, double complex permittivity_medium,
double focal_distance, double NA, double e_field, double complex *jones_vector, double n_1, double n_2, double filling_factor,
double aperture_radius, double width_inc, double delta, double damping_rate, double mass, double temperature, long* ret_dims_2d, int field_kind, int surface, double d_surf, double r_surf, double** ret_val)
{
    long i, j;
    double dt;
    double k1[6], y[6];


    for (i = 0; i < ret_dims_2d[0]; i++)
    {
        ret_val[i][0] = y0[i];
    };


    for (j = 1; j < ret_dims_2d[1]; j++)
        {
            dt = t[j] - t[j - 1];
            for (i = 0; i < ret_dims_2d[0]; i++)
            {
                y[i] = ret_val[i][j - 1];
            };

            if(func==0)
            {

                _runge_kutta_func_0(y, t[j - 1], dt / 4., power, width_x, width_y, volume, radius, wavelength, rayleigh_len, permittivity_particle, permittivity_medium, damping_rate, mass, temperature, k1);
            }
            else if(func==1)
            {

                _runge_kutta_func_1(y, t[j - 1], dt / 4., focal_distance, NA, volume, radius, permittivity_particle, permittivity_medium, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, delta, damping_rate, mass, temperature, field_kind, surface, d_surf, r_surf, k1);
            };


            for (i = ret_dims_2d[0] / 2; i < ret_dims_2d[0]; i++)
            {
                ret_val[i][j] = y[i] + dt * k1[i];
            };
            for (i = 0; i < ret_dims_2d[0] / 2; i++)
            {
                ret_val[i][j] = y[i] + dt * ret_val[i + ret_dims_2d[0] / 2][j];
            };

        };

}

#endif /* __CUSTOMMODULE_CONST__ */
