#ifndef __NUMERICMODULE_CONST__
#define __NUMERICMODULE_CONST__

#if defined(NO_APPEND_FORTRAN)
  #if defined(UPPERCASE_FORTRAN)
  /* nothing to do here */
  #else
    #define DQAGSE dqagse
    #define DQAGIE dqagie
    #define DQAGPE dqagpe
    #define DQAWOE dqawoe
    #define DQAWFE dqawfe
    #define DQAWSE dqawse
    #define DQAWCE dqawce
  #endif
#else
  #if defined(UPPERCASE_FORTRAN)
    #define DQAGSE DQAGSE_
    #define DQAGIE DQAGIE_
    #define DQAGPE DQAGPE_
    #define DQAWOE DQAWOE_
    #define DQAWFE DQAWFE_
    #define DQAWSE DQAWSE_
    #define DQAWCE DQAWCE_
#else
    #define DQAGSE dqagse_
    #define DQAGIE dqagie_
    #define DQAGPE dqagpe_
    #define DQAWOE dqawoe_
    #define DQAWFE dqawfe_
    #define DQAWSE dqawse_
    #define DQAWCE dqawce_
  #endif
#endif

//#include "cephes/jn.c"
#include <math.h>
#include <complex.h>
#include "gsl_const_mks.c"
#include "custom_module.c"
void DQAGSE();

static double rho__;
static double x__;
static double y__;
static double z__;
static double theta_max__;
static double k__;
static double filling_factor__;

//static double focal_distance__;
//static double NA__;
//static double e_field__;
//static double power__;
//static double *jones_vector__;
//static double wavelength__;
//static double n_1__;
//static double n_2__;
//static double aperture_radius__;
//static double width_inc__;

static double thunk_00_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * sin(*x) * (1 + cos(*x)) * jn(0, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_00_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * sin(*x) * (1 + cos(*x)) * jn(0, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_01_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_01_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_02_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * sin(*x) * (1 - cos(*x)) * jn(2, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_02_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * sin(*x) * (1 - cos(*x)) * jn(2, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_10_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x) * sin(*x)) * jn(0, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_10_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x) * sin(*x)) * jn(0, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_11_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 + 3. *cos(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_11_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 + 3. *cos(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_12_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 - cos(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_12_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 - cos(*x)) * jn(1, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_13_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x) * sin(*x)) * jn(2, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_13_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x) * sin(*x)) * jn(2, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_14_real(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return creal(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 - cos(*x)) * jn(3, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}
static double thunk_14_imag(double *x)
{
    double apodization_func = exp(-1. / (filling_factor__ * filling_factor__) * (sin(*x) * sin(*x)) / (sin(theta_max__) * sin(theta_max__)));
    return cimag(apodization_func * sqrt(cos(*x)) * (sin(*x) * sin(*x)) * (1 - cos(*x)) * jn(3, k__ * rho__ * sin(*x)) * (cos(k__ * z__ * cos(*x)) + sin(k__ * z__ * cos(*x)) * I));
}


static void custom_qagse(double (*thunk)(double *), double *result) {
  int      limit=50;
  int      full_output = 0;
  double   a=0, epsabs=1.49e-8, epsrel=1.49e-8;
  int      neval=0, ier=6, last=0, iord[limit];
  double   abserr=0.0;
  double   alist[limit], blist[limit], rlist[limit], elist[limit];
  double   *apointer=alist, *bpointer=blist, *rpointer=rlist, *epointer=elist;
  int      ret;
  int      *iordpointer=iord;

  DQAGSE(thunk, &a, &theta_max__, &epsabs, &epsrel, &limit, result, &abserr, &neval, &ier, apointer,
         bpointer, rpointer, epointer, iordpointer, &last);
}

static double complex I00(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_00_real, &result_real);
    custom_qagse(thunk_00_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I01(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_01_real, &result_real);
    custom_qagse(thunk_01_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I02(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_02_real, &result_real);
    custom_qagse(thunk_02_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I10(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_10_real, &result_real);
    custom_qagse(thunk_10_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I11(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_11_real, &result_real);
    custom_qagse(thunk_11_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I12(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_12_real, &result_real);
    custom_qagse(thunk_12_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I13(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_13_real, &result_real);
    custom_qagse(thunk_13_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static double complex I14(double rho, double z, double theta_max, double k, double filling_factor)
{
    double result_real, result_imag;
    double complex result;

    rho__ = rho;
    z__ = z;
    theta_max__ = theta_max;
    k__ = k;
    filling_factor__ = filling_factor;

    custom_qagse(thunk_14_real, &result_real);
    custom_qagse(thunk_14_imag, &result_imag);
    result = result_real + I * result_imag;

    return result;
}

static void _fields_00(double x, double y, double z, double focal_distance, double NA, double e_field, double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, int field, int surface, double d_surf, double r_surf, double complex *ret_val)
{

    double rho = sqrt(x * x + y * y);
    double phi = atan2(y, x);
    double theta_max = asin(NA / n_2);
    double k = 2. * M_PI / wavelength;
    double complex i00;
    double complex i01;
    double complex i02;
    double complex pre_factor_E, pre_factor_H;
    double complex mirror_field_3[3] = {0., 0., 0.};
    double complex mirror_field_6[6] = {0., 0., 0., 0., 0., 0.};

    int i;

    if(e_field == -1)
        e_field = sqrt(4. * power / (GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * M_PI * width_inc * width_inc));

    if(filling_factor == -1)
        filling_factor = width_inc / aperture_radius;

    i00 = I00(rho, z, theta_max, k, filling_factor);
    i01 = I01(rho, z, theta_max, k, filling_factor);
    i02 = I02(rho, z, theta_max, k, filling_factor);

    pre_factor_E = -1. * k * focal_distance * I / 2. * sqrt(n_1 / n_2) * e_field * (cos(-1. * k * focal_distance) + I * sin(-1. * k * focal_distance));

    if(field == 0)
    {
        ret_val[0] = pre_factor_E * (jones_vector[0] * (i00 + i02 * cos(2. * phi)) + jones_vector[1] * (-1 * i02 * sin(2. * phi + M_PI)));
        ret_val[1] = pre_factor_E * (jones_vector[0] * (i02 * sin(2. * phi)) + jones_vector[1] * (i00 + i02 * cos(2. * phi + M_PI)));
        ret_val[2] = pre_factor_E * (jones_vector[0] * (-2. * I * i01 * cos(phi)) + jones_vector[1] * (2. * I * i01 * cos(phi + M_PI / 2.)));

        if(surface == 1)
        {
            if(z <= d_surf)
            {
                _fields_00(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_3);
                ret_val[0] -= r_surf * mirror_field_3[0];
                ret_val[1] -= r_surf * mirror_field_3[1];
                ret_val[2] -= r_surf * mirror_field_3[2];
            };
        };
    }
    else if(field == 1)
    {
        pre_factor_H = pre_factor_E * GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * n_2;
        ret_val[0] = pre_factor_H * (jones_vector[0] * (i02 * sin(2. * phi)) + jones_vector[1] * (-1. * i00 + i02 * cos(2. * phi + M_PI)));
        ret_val[1] = pre_factor_H * (jones_vector[0] * (i00 - i02 * cos(2. * phi)) + jones_vector[1] * (i02 * sin(2. * phi + M_PI)));
        ret_val[2] = pre_factor_H * (jones_vector[0] * (-2. * I * i01 * cos(phi)) + jones_vector[1] * (-2. * I * i01 * cos(phi + M_PI / 2.)));

        if(surface == 1)
        {
            if(z <= d_surf)
            {

                _fields_00(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_3);
                ret_val[0] -= r_surf * mirror_field_3[0];
                ret_val[1] -= r_surf * mirror_field_3[1];
                ret_val[2] -= r_surf * mirror_field_3[2];
            };
        };

    }
    else if(field == 2)
    {
        pre_factor_H = pre_factor_E * GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * n_2;
        ret_val[0] = pre_factor_E * (jones_vector[0] * (i00 + i02 * cos(2. * phi)) + jones_vector[1] * (-1 * i02 * sin(2. * phi + M_PI)));
        ret_val[1] = pre_factor_E * (jones_vector[0] * (i02 * sin(2. * phi)) + jones_vector[1] * (i00 + i02 * cos(2. * phi + M_PI)));
        ret_val[2] = pre_factor_E * (jones_vector[0] * (-2. * I * i01 * cos(phi)) + jones_vector[1] * (2. * I * i01 * cos(phi + M_PI / 2.)));
        ret_val[3] = pre_factor_H * (jones_vector[0] * (i02 * sin(2. * phi)) + jones_vector[1] * (-1. * i00 + i02 * cos(2. * phi + M_PI)));
        ret_val[4] = pre_factor_H * (jones_vector[0] * (i00 - i02 * cos(2. * phi)) + jones_vector[1] * (i02 * sin(2. * phi + M_PI)));
        ret_val[5] = pre_factor_H * (jones_vector[0] * (-2. * I * i01 * cos(phi)) + jones_vector[1] * (-2. * I * i01 * cos(phi + M_PI / 2.)));

        if(surface == 1)
        {
            if(z <= d_surf)
            {
                _fields_00(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_6);
                ret_val[0] -= r_surf * mirror_field_6[0];
                ret_val[1] -= r_surf * mirror_field_6[1];
                ret_val[2] -= r_surf * mirror_field_6[2];
                ret_val[3] -= r_surf * mirror_field_6[3];
                ret_val[4] -= r_surf * mirror_field_6[4];
                ret_val[5] -= r_surf * mirror_field_6[5];
            };
        };

    };

}

static void _fields_doughnut_rad(double x, double y, double z, double focal_distance, double NA, double e_field, double power, double *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, int field, int surface, double d_surf, double r_surf, double complex *ret_val)
{

    double rho = sqrt(x * x + y * y);
    double phi = atan2(y, x);
    double theta_max = asin(NA / n_2);
    double k = 2. * M_PI / wavelength;
    double complex i11;
    double complex i12;
    double complex i10;
    double complex pre_factor_E, pre_factor_H;
    int i;
    double complex mirror_field_3[3] = {0., 0., 0.};
    double complex mirror_field_6[6] = {0., 0., 0., 0., 0., 0.};


    if(e_field == -1)
        e_field = sqrt(4. * power / (GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * M_PI * width_inc * width_inc));

    if(filling_factor == -1)
        filling_factor = width_inc / aperture_radius;

    i11 = I11(rho, z, theta_max, k, filling_factor);
    i12 = I12(rho, z, theta_max, k, filling_factor);
    i10 = I10(rho, z, theta_max, k, filling_factor);

    pre_factor_E = -1. * k * focal_distance * focal_distance * I / (2. * width_inc) * sqrt(n_1 / n_2) * e_field * (cos(-1. * k * focal_distance) + I * sin(-1. * k * focal_distance));

    if(field == 0)
    {
        ret_val[0] = pre_factor_E * I * (i11 - i12) * cos(phi);
        ret_val[1] = pre_factor_E * I * (i11 - i12) * sin(phi);
        ret_val[2] = pre_factor_E * (- 4. * i10);

        if(surface == 1)
        {
            if(z <= d_surf)
            {
                _fields_doughnut_rad(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_3);
                ret_val[0] -= r_surf * mirror_field_3[0];
                ret_val[1] -= r_surf * mirror_field_3[1];
                ret_val[2] -= r_surf * mirror_field_3[2];
            };
        };

    }
    else if(field == 1)
    {
        pre_factor_H = pre_factor_E * GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * n_2;
        ret_val[0] = pre_factor_H * (-1. * I) * (i11 + 3. * i12) * sin(phi);
        ret_val[1] = pre_factor_H * I * (i11 + 3. * i12) * cos(phi);
        ret_val[2] = 0;

        if(surface == 1)
        {
            if(z <= d_surf)
            {
                _fields_doughnut_rad(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_3);
                ret_val[0] -= r_surf * mirror_field_3[0];
                ret_val[1] -= r_surf * mirror_field_3[1];
                ret_val[2] -= r_surf * mirror_field_3[2];
            };
        };

    }
    else if(field == 2)
    {
        pre_factor_H = pre_factor_E * GSL_CONST_MKS_SPEED_OF_LIGHT * GSL_CONST_MKS_VACUUM_PERMITTIVITY * n_2;
        ret_val[0] = pre_factor_E * I * (i11 - i12) * cos(phi);
        ret_val[1] = pre_factor_E * I * (i11 - i12) * sin(phi);
        ret_val[2] = pre_factor_E * (- 4. * i10);
        ret_val[3] = pre_factor_H * (-1. * I) * (i11 + 3. * i12) * sin(phi);
        ret_val[4] = pre_factor_H * I * (i11 + 3. * i12) * cos(phi);
        ret_val[5] = 0;

        if(surface == 1)
        {
            if(z <= d_surf)
            {
                _fields_doughnut_rad(x, y, 2. * d_surf - z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, field, 0, 0, 0, mirror_field_6);
                ret_val[0] -= r_surf * mirror_field_6[0];
                ret_val[1] -= r_surf * mirror_field_6[1];
                ret_val[2] -= r_surf * mirror_field_6[2];
                ret_val[3] -= r_surf * mirror_field_6[3];
                ret_val[4] -= r_surf * mirror_field_6[4];
                ret_val[5] -= r_surf * mirror_field_6[5];
            };
        };

    };
}



static void first_derivative_efields(double x, double y, double z, double focal_distance, double NA, double e_field, double power, double complex *jones_vector, double wavelength, double n_1, double n_2, double filling_factor, double aperture_radius, double width_inc, double complex *grad_e_x, double complex *grad_e_y, double complex *grad_e_z, int field_kind, double delta, int surface, double d_surf, double r_surf) {
    int i;
    double complex val_1[3], val_2[3];

    if (field_kind == 0)
    {
    _fields_00(x - delta, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_00(x + delta, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_x[i] = -0.5 * (val_1[i] - val_2[i]) / delta;

    _fields_00(x, y - delta, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_00(x, y + delta, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_y[i] = -0.5 * (val_1[i] - val_2[i]) / delta;

    _fields_00(x, y, z - delta, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_00(x, y, (z + delta), focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_z[i] = -0.5 * (val_1[i] - val_2[i]) / delta;
    }
    else if (field_kind == 1)
    {
    _fields_doughnut_rad(x - delta, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_doughnut_rad(x + delta, y, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_x[i] = -0.5 * (val_1[i] - val_2[i]) / delta;

    _fields_doughnut_rad(x, y - delta, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_doughnut_rad(x, y + delta, z, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_y[i] = -0.5 * (val_1[i] - val_2[i]) / delta;

    _fields_doughnut_rad(x, y, z - delta, focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_1);
    _fields_doughnut_rad(x, y, (z + delta), focal_distance, NA, e_field, power, jones_vector, wavelength, n_1, n_2, filling_factor, aperture_radius, width_inc, 0, surface, d_surf, r_surf, val_2);

    for(i = 0; i <3; i++)
        grad_e_z[i] = -0.5 * (val_1[i] - val_2[i]) / delta;
    };
}


#endif /* __NUMERICMODULE_CONST__ */
