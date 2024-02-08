# /usr/bin/python3
# -*- coding:utf-8 -*-

import math
from scipy.signal import butter, lfilter, iirnotch

# Biquad filter (ref: https://arachnoid.com/phase_locked_loop/resources/biquad_module.py)
class Biquad:
    # pretend enumeration
    LOWPASS, HIGHPASS, BANDPASS, NOTCH, PEAK, LOWSHELF, HIGHSHELF = range(7)

    def __init__(self, typ, freq, srate, Q=0.707, dbGain = 0):
        types = {
        Biquad.LOWPASS : self.lowpass,
        Biquad.HIGHPASS : self.highpass,
        Biquad.BANDPASS : self.bandpass,
        Biquad.NOTCH : self.notch,
        Biquad.PEAK : self.peak,
        Biquad.LOWSHELF : self.lowshelf,
        Biquad.HIGHSHELF : self.highshelf
        }
        assert(types.has_key(typ))
        freq = float(freq)
        self.srate = float(srate)
        Q = float(Q)
        dbGain = float(dbGain)
        self.a0 = self.a1 = self.a2 = 0
        self.b0 = self.b1 = self.b2 = 0
        self.x1 = self.x2 = 0
        self.y1 = self.y2 = 0
        # only used for peaking and shelving filter types
        A = math.pow(10, dbGain / 40)
        omega = 2 * math.pi * freq / self.srate
        sn = math.sin(omega)
        cs = math.cos(omega)
        alpha = sn / (2*Q)
        beta = math.sqrt(A + A)
        types[typ](A,omega,sn,cs,alpha,beta)
        # prescale constants
        self.b0 /= self.a0
        self.b1 /= self.a0
        self.b2 /= self.a0
        self.a1 /= self.a0
        self.a2 /= self.a0

    def lowpass(self,A,omega,sn,cs,alpha,beta):
        self.b0 = (1 - cs) /2
        self.b1 = 1 - cs
        self.b2 = (1 - cs) /2
        self.a0 = 1 + alpha
        self.a1 = -2 * cs
        self.a2 = 1 - alpha
        
    def highpass(self,A,omega,sn,cs,alpha,beta):
        self.b0 = (1 + cs) /2
        self.b1 = -(1 + cs)
        self.b2 = (1 + cs) /2
        self.a0 = 1 + alpha
        self.a1 = -2 * cs
        self.a2 = 1 - alpha
        
    def bandpass(self,A,omega,sn,cs,alpha,beta):
        self.b0 = alpha
        self.b1 = 0
        self.b2 = -alpha
        self.a0 = 1 + alpha
        self.a1 = -2 * cs
        self.a2 = 1 - alpha
        
    def notch(self,A,omega,sn,cs,alpha,beta):
        self.b0 = 1
        self.b1 = -2 * cs
        self.b2 = 1
        self.a0 = 1 + alpha
        self.a1 = -2 * cs
        self.a2 = 1 - alpha
        
    def peak(self,A,omega,sn,cs,alpha,beta):
        self.b0 = 1 + (alpha * A)
        self.b1 = -2 * cs
        self.b2 = 1 - (alpha * A)
        self.a0 = 1 + (alpha /A)
        self.a1 = -2 * cs
        self.a2 = 1 - (alpha /A)
        
    def lowshelf(self,A,omega,sn,cs,alpha,beta):
        self.b0 = A * ((A + 1) - (A - 1) * cs + beta * sn)
        self.b1 = 2 * A * ((A - 1) - (A + 1) * cs)
        self.b2 = A * ((A + 1) - (A - 1) * cs - beta * sn)
        self.a0 = (A + 1) + (A - 1) * cs + beta * sn
        self.a1 = -2 * ((A - 1) + (A + 1) * cs)
        self.a2 = (A + 1) + (A - 1) * cs - beta * sn
        
    def highshelf(self,A,omega,sn,cs,alpha,beta):
        self.b0 = A * ((A + 1) + (A - 1) * cs + beta * sn)
        self.b1 = -2 * A * ((A - 1) + (A + 1) * cs)
        self.b2 = A * ((A + 1) + (A - 1) * cs - beta * sn)
        self.a0 = (A + 1) - (A - 1) * cs + beta * sn
        self.a1 = 2 * ((A - 1) - (A + 1) * cs)
        self.a2 = (A + 1) - (A - 1) * cs - beta * sn
        
    # perform filtering function
    def __call__(self,x):
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y
        
    # provide a static result for a given frequency f
    def result(self,f):
        phi = (math.sin(math.pi * f * 2/(2.0 * self.srate)))**2
        return ((self.b0+self.b1+self.b2)**2 - \
        4*(self.b0*self.b1 + 4*self.b0*self.b2 + \
        self.b1*self.b2)*phi + 16*self.b0*self.b2*phi*phi) / \
        ((1+self.a1+self.a2)**2 - 4*(self.a1 + \
        4*self.a2 + self.a1*self.a2)*phi + 16*self.a2*phi*phi)

    def log_result(self,f):
        try:
            r = 10 * math.log10(self.result(f))
        except:
            r = -200
        return r

    # return computed constants
    def constants(self):
        return self.b0,self.b1,self.b2,self.a1,self.a2
    


# apply bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = lfilter(b,a, data)
    return filtered_data

# apply notch filter (a.k.a. bandstop filter)
def notch_pass_filter(data, center, interval=20, sr=4000, normalized=False):
    center = center / (sr/2) if normalized else center
    b, a = iirnotch(center, interval/interval, sr)
    filtered_data = lfilter(b, a, data)
    return filtered_data