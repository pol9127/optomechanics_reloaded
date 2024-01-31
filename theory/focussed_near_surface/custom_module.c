#ifndef __CUSTOMMODULE_CONST__
#define __CUSTOMMODULE_CONST__


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <float.h>

#include "gsl_const_mks.c"
#include "numeric_module.c"

static double complex _r_s(double kx, double ky, double k1, double k2, double mu1, double mu2)
{
    double kz1, kz2;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        return (mu2 * kz1 - mu1 * kz2) / (mu2 * kz1 + mu1 * kz2);
    }
    else
    {
        return 0;
    };
}

static double complex _r_p(double kx, double ky, double k1, double k2, double eps1, double eps2)
{
    double kz1, kz2;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        return (eps2 * kz1 - eps1 * kz2) / (eps2 * kz1 + eps1 * kz2);
    }
    else
    {
        return 0;
    };
}

static double complex _t_s(double kx, double ky, double k1, double k2, double mu1, double mu2)
{
    double kz1, kz2;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        return (2 * mu2 * kz1) / (mu2 * kz1 + mu1 * kz2);
    }
    else
    {
        return 0;
    };
}

static double complex _t_p(double kx, double ky, double k1, double k2, double mu1, double mu2, double eps1, double eps2)
{
    double kz1, kz2;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        return (2 * eps2 * kz1) / (eps2 * kz1 + eps1 * kz2) * sqrt(mu2 * eps1 / (mu1 * eps2));
    }
    else
    {
        return 0;
    };
}

static double complex _r_s_membrane(double kx, double ky, double k1, double k2, double mu1, double mu2, double d)
{
    double kz2;
    double complex r_s;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        r_s = _r_s(kx, ky, k1, k2, mu1, mu2);
        return r_s * (1 - (cos(2 * kz2 * d) + I * sin(2 * kz2 * d))) / (1 - r_s * r_s * (cos(2 * kz2 * d) + I * sin(2 * kz2 * d)));
    }
    else
    {
        return 0;
    };
}

static double complex _r_p_membrane(double kx, double ky, double k1, double k2, double eps1, double eps2, double d)
{
    double kz2;
    double complex r_p;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        r_p = _r_p(kx, ky, k1, k2, eps1, eps2);
        return r_p * (1 - (cos(2 * kz2 * d) + I * sin(2 * kz2 * d))) / (1 - r_p * r_p * (cos(2 * kz2 * d) + I * sin(2 * kz2 * d)));
    }
    else
    {
        return 0;
    };
}

static double complex _t_s_membrane(double kx, double ky, double k1, double k2, double mu1, double mu2, double d)
{
    double kz2;
    double complex t_s_1, t_s_2, r_s;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        r_s = _r_s(kx, ky, k1, k2, mu1, mu2);
        t_s_1 = _t_s(kx, ky, k1, k2, mu1, mu2);
        t_s_2 = _t_s(kx, ky, k2, k1, mu2, mu1);
        return (t_s_1 * t_s_2 * (cos(kz2 * d) + I * sin(kz2 * d))) / (1 - r_s * r_s * (cos(2 * kz2 * d) + I * sin(2 * kz2 * d)));
    }
    else
    {
        return 0;
    };
}

static double complex _t_p_membrane(double kx, double ky, double k1, double k2, double mu1, double mu2, double eps1, double eps2, double d)
{
    double kz2;
    double complex t_p_1, t_p_2, r_p;
    if((kx * kx + ky * ky <= k1 * k1) & (kx * kx + ky * ky <= k2 * k2))
    {
        kz2 = sqrt(k2 * k2 - kx * kx - ky * ky);
        r_p = _r_s(kx, ky, k1, k2, eps1, eps2);
        t_p_1 = _t_p(kx, ky, k1, k2, mu1, mu2, eps1, eps2);
        t_p_2 = _t_p(kx, ky, k2, k1, mu2, mu1, eps1, eps2);
        return (t_p_1 * t_p_2 * (cos(kz2 * d) + I * sin(kz2 * d))) / (1 - r_p * r_p * (cos(2 * kz2 * d) + I * sin(2 * kz2 * d)));
    }
    else
    {
        return 0;
    };
}


static double _E_inc(double kx, double ky, double f, double w0, double k1, double E0)
{
    double kz1;
    if(kx * kx + ky * ky <= k1 * k1)
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        return E0 * exp(-f * f * (kx * kx + ky * ky) / (w0 * w0 * (kx * kx + ky * ky + kz1 * kz1)));
    }
    else
    {
        return 0;
    };
}

static void _E_inf(double kx, double ky, double f, double w0, double k1, double E0, double complex *ret_val)
{
    double Einc;
    double kz1;
    double prefac;
    if(kx * kx + ky * ky <= k1 * k1)
    {
        if(kx * kx + ky * ky < DBL_MIN)
        {
            ret_val[0] = E0;
            ret_val[1] = 0;
            ret_val[2] = 0;
        }
        else
        {
            kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
            Einc  = _E_inc(kx, ky, f, w0, k1, E0);
            prefac = sqrt(kz1 / k1) / (kx * kx + ky * ky);
            ret_val[0] = prefac * (ky * ky + kx * kx / k1 * kz1) * Einc;
            ret_val[1] = prefac * (kx * ky / k1 * kz1 - kx * ky) * Einc;
            ret_val[2] = -1 * prefac * kx / k1 * (kx * kx + ky * ky) * Einc;
        };
    }
    else
    {
        ret_val[0] = 0;
        ret_val[1] = 0;
        ret_val[2] = 0;
    };
}

static void _E_r_inf(double kx, double ky, double f, double w0, double k1, double k2, double E0, double z0, double mu1, double mu2, double eps1, double eps2, double d, double complex *ret_val)
{
    double Einc;
    double kz1;
    double prefac;
    double complex prefac2;
    double complex rs, rp;
    if(d < DBL_MIN)
    {
        rs = _r_s(kx, ky, k1, k2, mu1, mu2);
        rp = _r_p(kx, ky, k1, k2, eps1, eps2);
    }
    else
    {
        rs = _r_s_membrane(kx, ky, k1, k2, mu1, mu2, d);
        rp = _r_p_membrane(kx, ky, k1, k2, eps1, eps2, d);
    };

    if(kx * kx + ky * ky <= k1 * k1)
    {
        if(kx * kx + ky * ky < DBL_MIN)
        {
            ret_val[0] = E0 * (rs - rp) / 2 * (cos(2 * z0 * k1) + I * sin(2 * z0 * k1));
            ret_val[1] = -E0 * (rs + rp) / 2 * (cos(2 * z0 * k1) + I * sin(2 * z0 * k1));
            ret_val[2] = 0;
        }
        else
        {
            kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
            Einc  = _E_inc(kx, ky, f, w0, k1, E0);
            prefac = -sqrt(kz1 / k1) / (kx * kx + ky * ky);
            prefac2 = cos(2 * z0 * kz1) + I * sin(2 * z0 * kz1);
            ret_val[0] = prefac * (-ky * ky * rs + kx * kx * rp * kz1 / k1) * Einc * prefac2;
            ret_val[1] = prefac * (kx * ky * rs + kx * rp * ky * kz1 / k1) * Einc * prefac2;
            ret_val[2] = -1 * kx * rp / k1 * sqrt(kz1 / k1) * Einc * prefac2;
        };
    }
    else
    {
        ret_val[0] = 0;
        ret_val[1] = 0;
        ret_val[2] = 0;
    };
}

static void _E_r_integrand(double x, double y, double z, double kx, double ky, double f, double w0, double k1, double k2, double E0, double z0, double mu1, double mu2, double eps1, double eps2, double d, double complex *ret_val)
{
    double kz1;
    double complex prefac;

    if(kx * kx + ky * ky < k1 * k1)
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        _E_r_inf(kx, ky, f, w0, k1, k2, E0, z0, mu1, mu2, eps1, eps2, d, ret_val);
        prefac = -I * f / (2 * M_PI) * (cos(-k1 * f) + I * sin(-k1 * f)) * (cos(kx * x + ky * y - kz1 * z) + I * sin(kx * x + ky * y - kz1 * z)) / kz1;
        ret_val[0] *= prefac;
        ret_val[1] *= prefac;
        ret_val[2] *= prefac;
    }
    else
    {
        ret_val[0] = 0;
        ret_val[1] = 0;
        ret_val[2] = 0;
    };
}

static void _E_f_integrand(double x, double y, double z, double kx, double ky, double f, double w0, double k1, double E0, double complex *ret_val)
{
    double kz1;
    double complex prefac;

    if(kx * kx + ky * ky < k1 * k1)
    {
        kz1 = sqrt(k1 * k1 - kx * kx - ky * ky);
        _E_inf(kx, ky, f, w0, k1, E0, ret_val);
        prefac = -I * f / (2 * M_PI) * (cos(-k1 * f) + I * sin(-k1 * f)) * (cos(kx * x + ky * y + kz1 * z) + I * sin(kx * x + ky * y + kz1 * z)) / kz1;
        ret_val[0] *= prefac;
        ret_val[1] *= prefac;
        ret_val[2] *= prefac;
    }
    else
    {
        ret_val[0] = 0;
        ret_val[1] = 0;
        ret_val[2] = 0;
    };
}

static void _dblsimps(double complex ** factor_mat, double complex *** vector_mat, double delta_kx, double delta_ky, long kx_len, long ky_len, double complex *ret_val)
{
    long kx_itr;
    long ky_itr;
    long i;

    double complex ** _temp = (double complex **)malloc(kx_len * sizeof(double complex*));
    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        _temp[kx_itr] = (double complex *) malloc(3 * sizeof(double complex));
    };

    ret_val[0] = 0;
    ret_val[1] = 0;
    ret_val[2] = 0;

    for (kx_itr = 0; kx_itr < kx_len; kx_itr++)
    {
        _temp[kx_itr][0] = 0;
        _temp[kx_itr][1] = 0;
        _temp[kx_itr][2] = 0;

        for (ky_itr = 1; ky_itr < ky_len / 2; ky_itr++)
        {
            _temp[kx_itr][0] += factor_mat[kx_itr][2 * ky_itr - 2] * vector_mat[kx_itr][2 * ky_itr - 2][0];
            _temp[kx_itr][0] += 4 * factor_mat[kx_itr][2 * ky_itr - 1] * vector_mat[kx_itr][2 * ky_itr - 1][0];
            _temp[kx_itr][0] += factor_mat[kx_itr][2 * ky_itr] * vector_mat[kx_itr][2 * ky_itr][0];
            _temp[kx_itr][1] += factor_mat[kx_itr][2 * ky_itr - 2] * vector_mat[kx_itr][2 * ky_itr - 2][1];
            _temp[kx_itr][1] += 4 * factor_mat[kx_itr][2 * ky_itr - 1] * vector_mat[kx_itr][2 * ky_itr - 1][1];
            _temp[kx_itr][1] += factor_mat[kx_itr][2 * ky_itr] * vector_mat[kx_itr][2 * ky_itr][1];
            _temp[kx_itr][2] += factor_mat[kx_itr][2 * ky_itr - 2] * vector_mat[kx_itr][2 * ky_itr - 2][2];
            _temp[kx_itr][2] += 4 * factor_mat[kx_itr][2 * ky_itr - 1] * vector_mat[kx_itr][2 * ky_itr - 1][2];
            _temp[kx_itr][2] += factor_mat[kx_itr][2 * ky_itr] * vector_mat[kx_itr][2 * ky_itr][2];
        };
        _temp[kx_itr][0] *= delta_ky / 3;
        _temp[kx_itr][1] *= delta_ky / 3;
        _temp[kx_itr][2] *= delta_ky / 3;
    };
    for (kx_itr = 1; kx_itr < kx_len / 2; kx_itr++)
    {
        ret_val[0] += _temp[2 * kx_itr - 2][0] + 4 * _temp[2 * kx_itr - 1][0] + _temp[2 * kx_itr][0];
        ret_val[1] += _temp[2 * kx_itr - 2][1] + 4 * _temp[2 * kx_itr - 1][1] + _temp[2 * kx_itr][1];
        ret_val[2] += _temp[2 * kx_itr - 2][2] + 4 * _temp[2 * kx_itr - 1][2] + _temp[2 * kx_itr][2];
    };
    ret_val[0] *= delta_kx / 3;
    ret_val[1] *= delta_kx / 3;
    ret_val[2] *= delta_kx / 3;
    free(_temp);
}


#endif /* __CUSTOMMODULE_CONST__ */
