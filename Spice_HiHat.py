# Spice Hi-Hat (opamp base) generator

import numpy as np
from scipy.io.wavfile import write
import random

random.seed(123)

fs = 44100
dt = 1.0 / fs
duration = 5.0
N = int(duration * fs)

R_noise = 10e3
C_min = 2e-7
C_max = 8e-9
C_noise = 4e-7
tau_noise = R_noise * C_noise

R_hpf = 3.3e3
C_hpf = 7.5e-9
tau_hpf = R_hpf * C_hpf

R_env = 10e3
C_env = 1e-6
tau_env = R_env * C_env
Venv = 0.0
charge_time = int(0.002 * fs)
charging = False
charge_start = 0

A_ol_dc = 5e4
f_pole = 500.0
omega_p = 2*np.pi*f_pole
slew_rate = 2e6
Vrail_pos, Vrail_neg = 15.0, -15.0
Vout_headroom = 1.5
Vout_max = Vrail_pos - Vout_headroom
Vout_min = Vrail_neg + Vout_headroom
soft_sat_alpha = 0.5

trigger_interval = 0.150
samples_per_trigger = int(trigger_interval * fs)

rng = np.random.default_rng(123)

Vc_noise = 0.0
x_prev_hpf = 0.0
y_prev_hpf = 0.0
v_ol_state = 0.0
Vout = 0.0
y_out = np.zeros(N, dtype=np.float32)

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
    if n % samples_per_trigger == 0:
        C_noise = random.uniform(C_min, C_max)
        tau_noise = R_noise * C_noise
        charging = True
        charge_start = n

    inoise = rng.normal(0.0, 1.0e-9)
    Vc_noise += (inoise*R_noise - Vc_noise)/tau_noise*dt

    RC = tau_hpf
    alpha = RC/(RC+1.0/fs)
    x_hpf = Vc_noise
    y_hpf = alpha*(y_prev_hpf + x_hpf - x_prev_hpf)
    x_prev_hpf = x_hpf
    y_prev_hpf = y_hpf

    dv_ol = omega_p*(A_ol_dc*y_hpf - v_ol_state)*dt
    v_ol_state += dv_ol
    Vsat = soft_saturation(v_ol_state, Vout_max, Vout_min, soft_sat_alpha)
    Vout = limit_slew(Vsat, Vout, dt, slew_rate)

    if charging and (n - charge_start) < charge_time:
        Icharge = 1e-6
    else:
        Icharge = 0.0
        charging = False

    dV = (-Venv / tau_env + Icharge / C_env) * dt
    Venv += dV

    y_out[n] = np.tanh(0.9 * Venv * Vout).astype(np.float32)

max_abs = np.max(np.abs(y_out))
if max_abs > 0:
    y_out *= (0.9/max_abs)

def float_to_pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return np.int16(np.round(x*32767.0))

audio = float_to_pcm16(y_out)

write("hihat_opamp.wav", fs, audio)