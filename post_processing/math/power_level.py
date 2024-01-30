import numpy as np

def rms(kind, amplitude):
    factors = {'sinusoidal': 1/np.sqrt(2), 'triangular': 1/np.sqrt(3)}
    return amplitude * factors[kind]

def power(kind, amplitude):
    return rms(kind, amplitude)**2/50

def dBm(pw):
    return 10*np.log10(pw/10**-3)

def dBm_inverse(dBm):
    return 10**(dBm/10 - 3)

def rms_inverse(kind, rms):
    factors = {'sinusoidal': 1/np.sqrt(2), 'triangular': 1/np.sqrt(3)}
    return rms / factors[kind]

def power_inverse(kind, power):
    return rms_inverse(kind, np.sqrt(power*50))
