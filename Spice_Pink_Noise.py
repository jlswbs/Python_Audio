# Spice pink noise (opamp base) generator

import numpy as np
import struct, random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0 / fs
duration = 5.0
N = int(duration * fs)

R = 10e3
C_min = 1e-9
C_max = 1e-6
C = (C_min + C_max) / 2.0
tau = R * C

change_interval = 0.150
samples_per_change = int(change_interval * fs)

A_ol_dc = 1e5
f_pole = 50.0
omega_p = 2.0 * np.pi * f_pole
slew_rate = 9e6
Vrail_pos, Vrail_neg = 15.0, -15.0
Vout_headroom = 1.5
Vout_max = Vrail_pos - Vout_headroom
Vout_min = Vrail_neg + Vout_headroom
soft_sat_alpha = 0.45

Vc = 0.0
Vout = 0.0
v_ol_state = 0.0
y_out = np.zeros(N, dtype=np.float32)

def soft_saturation(v, vmax, vmin, alpha):
    center = 0.5 * (vmax + vmin)
    span   = 0.5 * (vmax - vmin)
    x = (v - center) / span
    return center + span * np.tanh(alpha * x)

def limit_slew(v_target, v_prev, dt, SR):
    dv = v_target - v_prev
    dv_max = SR * dt
    dv = np.clip(dv, -dv_max, dv_max)
    return v_prev + dv

rng = np.random.default_rng(123)

for n in range(N):

    if n % samples_per_change == 0:
        C = random.uniform(C_min, C_max)
        tau = R * C

    inoise = rng.normal(0.0, 1.0e-9)

    Vc += (inoise * R - Vc) / tau * dt

    dv_ol = omega_p * (A_ol_dc * Vc - v_ol_state) * dt
    v_ol_state += dv_ol

    Vsat = soft_saturation(v_ol_state, Vout_max, Vout_min, soft_sat_alpha)
    Vout = limit_slew(Vsat, Vout, dt, slew_rate)

    y_out[n] = np.float32(Vout)

max_abs = np.max(np.abs(y_out))
norm = 0.9 / max_abs if max_abs > 0 else 1.0
y_out *= norm

def float_to_pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return np.int16(np.round(x * 32767.0))

audio = float_to_pcm16(y_out)

write("opamp_noise.wav", fs, audio)