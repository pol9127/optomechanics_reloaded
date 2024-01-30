#copied from http://www.earlevel.com/scripts/widgets/20131013/biquads2.js
import numpy as np
def calcBiquad(biquad_type, Fc, Fs, Q, gain): #, peakGain):
    #V = Math.power(10, np.abs(peakGain) / 20);
    K = np.tan(np.pi * Fc / Fs);
    if biquad_type == "one-pole lp":
        b1 = np.exp(-2.0 * np.pi * (Fc / Fs));
        a0 = 1.0 - b1;
        b1 = -b1;
        a1 = a2 = b2 = 0;
            
    elif biquad_type == "one-pole hp":
        b1 = -np.exp(-2.0 * np.pi * (0.5 - Fc / Fs));
        a0 = 1.0 + b1;
        b1 = -b1;
        a1 = a2 = b2 = 0;
            
    elif biquad_type == "lowpass":
        norm = 1 / (1 + K / Q + K * K);
        a0 = K * K * norm;
        a1 = 2 * a0;
        a2 = a0;
        b1 = 2 * (K * K - 1) * norm;
        b2 = (1 - K / Q + K * K) * norm;
        
    elif biquad_type == "highpass":
        norm = 1 / (1 + K / Q + K * K);
        a0 = 1 * norm;
        a1 = -2 * a0;
        a2 = a0;
        b1 = 2 * (K * K - 1) * norm;
        b2 = (1 - K / Q + K * K) * norm;
        
    elif biquad_type == "bandpass":
        norm = 1 / (1 + K / Q + K * K);
        a0 = K / Q * norm;
        a1 = 0;
        a2 = -a0;
        b1 = 2 * (K * K - 1) * norm;
        b2 = (1 - K / Q + K * K) * norm;
        
    elif biquad_type == "notch":
        norm = 1 / (1 + K / Q + K * K);
        a0 = (1 + K * K) * norm;
        a1 = 2 * (K * K - 1) * norm;
        a2 = a0;
        b1 = a1;
        b2 = (1 - K / Q + K * K) * norm;
        
    #elif biquad_type == "peak":
    #    if (peakGain >= 0):
    #        norm = 1 / (1 + 1/Q * K + K * K);
    #        a0 = (1 + V/Q * K + K * K) * norm;
    #        a1 = 2 * (K * K - 1) * norm;
    #        a2 = (1 - V/Q * K + K * K) * norm;
    #        b1 = a1;
    #        b2 = (1 - 1/Q * K + K * K) * norm;
    #    else:
    #        norm = 1 / (1 + V/Q * K + K * K);
    #        a0 = (1 + 1/Q * K + K * K) * norm;
    #        a1 = 2 * (K * K - 1) * norm;
    #        a2 = (1 - 1/Q * K + K * K) * norm;
    #        b1 = a1;
    #        b2 = (1 - V/Q * K + K * K) * norm;
    #        
    #elif biquad_type == "lowShelf":
    #    if (peakGain >= 0):
    #        norm = 1 / (1 + Math.SQRT2 * K + K * K);
    #        a0 = (1 + Math.sqrt(2*V) * K + V * K * K) * norm;
    #        a1 = 2 * (V * K * K - 1) * norm;
    #        a2 = (1 - Math.sqrt(2*V) * K + V * K * K) * norm;
    #        b1 = 2 * (K * K - 1) * norm;
    #        b2 = (1 - Math.SQRT2 * K + K * K) * norm;
    #    else:
    #        norm = 1 / (1 + Math.sqrt(2*V) * K + V * K * K);
    #        a0 = (1 + Math.SQRT2 * K + K * K) * norm;
    #        a1 = 2 * (K * K - 1) * norm;
    #        a2 = (1 - Math.SQRT2 * K + K * K) * norm;
    #        b1 = 2 * (V * K * K - 1) * norm;
    #        b2 = (1 - Math.sqrt(2*V) * K + V * K * K) * norm;
    #elif biquad_type == "highShelf":
    #    if (peakGain >= 0) :
    #        norm = 1 / (1 + Math.SQRT2 * K + K * K);
    #        a0 = (V + Math.sqrt(2*V) * K + K * K) * norm;
    #        a1 = 2 * (K * K - V) * norm;
    #        a2 = (V - Math.sqrt(2*V) * K + K * K) * norm;
    #        b1 = 2 * (K * K - 1) * norm;
    #        b2 = (1 - Math.SQRT2 * K + K * K) * norm;
    #    else:
    #        norm = 1 / (V + Math.sqrt(2*V) * K + K * K);
    #        a0 = (1 + Math.SQRT2 * K + K * K) * norm;
    #        a1 = 2 * (K * K - 1) * norm;
    #        a2 = (1 - Math.SQRT2 * K + K * K) * norm;
    #        b1 = 2 * (K * K - V) * norm;
    #        b2 = (V - Math.sqrt(2*V) * K + K * K) * norm;
    return [a0*gain,a1*gain,a2*gain,b1,b2]