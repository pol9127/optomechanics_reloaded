# Calculates Allan Variance as escribed by Erik:
# https://sisyphous.ee.ethz.ch/dokuwiki/lib/exe/fetch.php?media=general:170729_gm_long_eh.pdf
# by Felix Tebbenjohanns May 2018

import numpy as np

def block_averages(data, tau_ind):
    N = int(data.shape[0] / tau_ind)
    tmp = np.average(data[:N*tau_ind].reshape(N, tau_ind), axis = 1)
    t = np.arange(N) * tau_ind
    return (t, tmp);
    

def allan_variance_fixed_tau(data, tau_ind):    
    tmp = block_averages(data, tau_ind)[1]
    return np.average((tmp[1:]-tmp[:-1])**2/2) # see Erik's presentation

    
def allan_variance (data, tau_ind_arr):
    avs = []
    for tau_ind in tau_ind_arr:
        av = allan_variance_fixed_tau(data, tau_ind)
        avs.append(av)
    return np.array(avs)