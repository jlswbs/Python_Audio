# Spice NE5532 (opamp RC relaxation) FM (frequency modulator) oscillator with varicap

import numpy as np
import random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0/fs
duration = 5.0
N = int(duration*fs)

RintA_min, RintA_max = 5000.0, 20000.0
Rint_A = 8000.0
C0_A   = 3.3e-8
Vj_A   = 0.8
m_A    = 0.5
Cmin_A = 0.5*C0_A
Cmax_A = 1.5*C0_A
bias_offset = 2.0
mod_gain    = 1.5

C_B = 3.3e-7
RintB_min, RintB_max = 5000.0, 20000.0
Rint_B = 12000.0

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

Vc_A = 0.0
Vc_B = 0.0
dir_A = +1
dir_B = +1
v_ol_state_A = 0.0
v_ol_state_B = 0.0
rng = np.random.default_rng(123)

y_out = np.zeros(N, dtype=np.float32)

change_interval = 0.150
samples_per_change = int(change_interval*fs)

mod_depth = 0.5

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

def varicap_C(Vctrl, C0, Vj, m, Cmin, Cmax):
    Veff = max(0.0, Vctrl)
    C = C0 / (1.0 + Veff / Vj)**m
    return float(np.clip(C, Cmin, Cmax))

for n in range(N):
    if n % samples_per_change == 0:
        Rint_A = random.uniform(RintA_min, RintA_max)
        Rint_B = random.uniform(RintB_min, RintB_max)

    tau_B = Rint_B * C_B
    Vc_B += dir_B * dt / tau_B
    if Vc_B >= 1.0:
        Vc_B = 1.0
        dir_B = -1
    elif Vc_B <= -1.0:
        Vc_B = -1.0
        dir_B = +1

    Vdiff_B = Vc_B + rng.normal(0.0, vnoise_rms)
    dv_ol_B = omega_p*(A_ol_dc*Vdiff_B - v_ol_state_B)*dt
    v_ol_state_B += dv_ol_B
    Vsat_B = soft_saturation(v_ol_state_B, Vout_max, Vout_min, soft_sat_alpha)
    Vout_B = limit_slew(Vsat_B, y_out[n-1] if n>0 else 0.0, dt, slew_rate)

    Vctrl = bias_offset + mod_gain * Vout_B / Vout_max
    C_A = varicap_C(Vctrl, C0_A, Vj_A, m_A, Cmin_A, Cmax_A)
    tau_A = Rint_A * C_A

    Vc_A += dir_A * dt / tau_A
    if Vc_A >= 1.0:
        Vc_A = 1.0
        dir_A = -1
    elif Vc_A <= -1.0:
        Vc_A = -1.0
        dir_A = +1

    Vdiff_A = Vc_A + rng.normal(0.0, vnoise_rms)
    dv_ol_A = omega_p*(A_ol_dc*Vdiff_A - v_ol_state_A)*dt
    v_ol_state_A += dv_ol_A
    Vsat_A = soft_saturation(v_ol_state_A, Vout_max, Vout_min, soft_sat_alpha)
    Vout_A = limit_slew(Vsat_A, y_out[n-1] if n>0 else 0.0, dt, slew_rate)

    if use_square:
        y_out[n] = np.float32(Vout_A)
    else:
        amp_scale = Vout_max * 0.8
        y_out[n] = np.float32(amp_scale * Vc_A)

y_out *= 0.9/np.max(np.abs(y_out))

def float_to_pcm16(x):
    return np.int16(np.clip(x, -1.0, 1.0)*32767)

audio = float_to_pcm16(y_out)
write("fmvari_opamp_vco.wav", fs, audio)