# Spice NE5532 (opamp base) Wien bridge oscillator

import numpy as np
import random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0/fs
duration = 5.0
N = int(duration*fs)

R = 10000.0
C = 1.0e-8
Rmin, Rmax = 5000.0, 20000.0
Cmin, Cmax = 5e-9, 2e-8

f0 = 1.0/(2.0*np.pi*R*C)

A_ol_dc = 1e5
f_pole = 100.0
omega_p = 2*np.pi*f_pole
slew_rate = 9e6
soft_sat_alpha = 0.45
Vrail_pos, Vrail_neg = 15.0, -15.0
headroom = 1.5
Vout_max = Vrail_pos - headroom
Vout_min = Vrail_neg + headroom
vnoise_rms = 1e-7

target_amp = 6.0
agc_rate   = 0.0003
gain_min, gain_max = 2.95, 3.60
gain = 3.10

env = 0.0
env_alpha = (1.0 - np.exp(-dt*50.0))

y_out = np.zeros(N, dtype=np.float32)
v_ol_state = 0.0

bp = 1e-6
lp = 0.0
Q  = 1.0/np.sqrt(2.0)
D  = 1.0/Q * 1e-5

f_svf = 2.0*np.sin(np.pi*f0/fs)

def soft_saturation(v, vmax, vmin, alpha):
    center = 0.5*(vmax+vmin)
    span   = 0.5*(vmax-vmin)
    return center + span*np.tanh(alpha*(v-center)/span)

def limit_slew(v_target, v_prev, dt, SR):
    dv = v_target - v_prev
    dv_max = SR*dt
    dv = np.clip(dv, -dv_max, dv_max)
    return v_prev + dv

change_interval = 0.150
samples_per_change = int(change_interval*fs)

for n in range(N):
    if n % samples_per_change == 0:
        R = random.uniform(Rmin, Rmax)
        C = random.uniform(Cmin, Cmax)
        f0 = 1.0/(2.0*np.pi*R*C)
        f_svf = 2.0*np.sin(np.pi*f0/fs)
        
    noise = np.random.normal(0.0, vnoise_rms)

    hp = -lp - D*bp + noise
    bp += f_svf * hp
    lp += f_svf * bp

    v_wien = bp / 3.0

    Vdiff = gain * v_wien
    dv_ol = omega_p*(A_ol_dc*Vdiff - v_ol_state)*dt
    v_ol_state += dv_ol
    Vsat = soft_saturation(v_ol_state, Vout_max, Vout_min, soft_sat_alpha)
    Vout = limit_slew(Vsat, y_out[n-1] if n>0 else 0.0, dt, slew_rate)

    env += env_alpha * (abs(Vout) - env)
    gain += agc_rate * (target_amp - env)
    gain = np.clip(gain, gain_min, gain_max)

    y_out[n] = np.float32(Vout)

peak = np.max(np.abs(y_out))
    
if peak > 0:
    y_out *= (0.9/peak)

def float_to_pcm16(x):
    return np.int16(np.clip(x, -1.0, 1.0)*32767)

audio = float_to_pcm16(y_out)

write("wien_opamp_vco.wav", fs, audio)