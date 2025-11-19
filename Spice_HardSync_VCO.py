# Spice NE5532 (opamp RC relaxation) Hard-Sync oscillator

import numpy as np
import random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0/fs
duration = 5.0
N = int(duration*fs)

C_master = 3.3e-7
Rint_master = 8000.0
tau_master = Rint_master * C_master

C_slave = 3.3e-8
Rint_min, Rint_max = 5000.0, 20000.0
Rint_slave = (Rint_min + Rint_max) / 2.0
tau_slave = Rint_slave * C_slave

A_ol_dc = 1e5
f_pole = 500.0
omega_p = 2*np.pi*f_pole
slew_rate = 9e6
soft_sat_alpha = 0.45
vnoise_rms = 10e-6
Vrail_pos, Vrail_neg = 15.0, -15.0
headroom = 1.5
Vout_max = Vrail_pos - headroom
Vout_min = Vrail_neg + headroom

Vc_master = 0.0
Vc_slave = 0.0
dir_master = +1
dir_slave = +1
v_ol_state_master = 0.0
v_ol_state_slave  = 0.0
rng = np.random.default_rng(123)

y_out = np.zeros(N, dtype=np.float32)

change_interval = 0.150
samples_per_change = int(change_interval*fs)

use_square = False

def soft_saturation(v, vmax, vmin, alpha):
    center = 0.5*(vmax+vmin)
    span   = 0.5*(vmax-vmin)
    return center + span*np.tanh(alpha*(v-center)/span)

def limit_slew(v_target, v_prev, dt, SR):
    dv = v_target - v_prev
    dv_max = SR*dt
    dv = np.clip(dv, -dv_max, dv_max)
    return v_prev + dv

for n in range(N):
    if n % samples_per_change == 0:
        Rint_slave = random.uniform(Rint_min, Rint_max)
        tau_slave = Rint_slave * C_slave

    Vc_master += dir_master * dt / tau_master
    if Vc_master >= 1.0:
        Vc_master = 1.0
        dir_master = -1

        Vc_slave = -1.0
        dir_slave = +1
    elif Vc_master <= -1.0:
        Vc_master = -1.0
        dir_master = +1

        Vc_slave = -1.0
        dir_slave = +1

    Vc_slave += dir_slave * dt / tau_slave
    if Vc_slave >= 1.0:
        Vc_slave = 1.0
        dir_slave = -1
    elif Vc_slave <= -1.0:
        Vc_slave = -1.0
        dir_slave = +1

    Vdiff = Vc_slave + rng.normal(0.0, vnoise_rms)
    dv_ol = omega_p*(A_ol_dc*Vdiff - v_ol_state_slave)*dt
    v_ol_state_slave += dv_ol
    Vsat = soft_saturation(v_ol_state_slave, Vout_max, Vout_min, soft_sat_alpha)
    Vout_slave = limit_slew(Vsat, y_out[n-1] if n>0 else 0.0, dt, slew_rate)

    if use_square:
        y_out[n] = np.float32(Vout_slave)
    else:
        amp_scale = Vout_max * 0.8
        y_out[n] = np.float32(amp_scale * Vc_slave)

y_out *= 0.9/np.max(np.abs(y_out))

def float_to_pcm16(x):
    return np.int16(np.clip(x, -1.0, 1.0)*32767)

audio = float_to_pcm16(y_out)
write("hardsync_opamp_vco.wav", fs, audio)