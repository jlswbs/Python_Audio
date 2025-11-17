# Spice NE5532 (opamp RC relaxation) PWM oscillator

import numpy as np
import struct, random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0 / fs
duration = 5.0
N = int(duration * fs)

C = 3.3e-7
Rint = 10000.0
tau = Rint * C

change_interval = 0.150
samples_per_change = int(change_interval * fs)

Vref_min = -0.8
Vref_max = +0.8
Vref = 0.0

Rint_min = 5000.0
Rint_max = 20000.0

Vhyst = 0.02

Vrail_pos = 15.0
Vrail_neg = -15.0
Vout_headroom = 1.5
Vout_max = Vrail_pos - Vout_headroom
Vout_min = Vrail_neg + Vout_headroom

A_ol_dc = 1e5
f_pole = 500.0
omega_p = 2.0 * np.pi * f_pole
slew_rate = 9e6
soft_sat_alpha = 0.45
vnoise_rms = 10e-6

Vc = 0.0
direction = +1
v_ol_state_cmp  = 0.0
Vout_pwm = Vout_max
rng = np.random.default_rng(123)

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
    if n % samples_per_change == 0:
        Vref = random.uniform(Vref_min, Vref_max)
        Rint = random.uniform(Rint_min, Rint_max)
        tau = Rint * C

    Vc += direction * dt / tau
    if Vc >= 1.0:
        Vc = 1.0
        direction = -1
    elif Vc <= -1.0:
        Vc = -1.0
        direction = +1
    Vtri = Vc

    Vref_eff = Vref + (Vhyst if Vout_pwm > 0 else -Vhyst)
    Vdiff = (Vtri - Vref_eff) + rng.normal(0.0, vnoise_rms)

    dv_ol_cmp = omega_p * (A_ol_dc * Vdiff - v_ol_state_cmp) * dt
    v_ol_state_cmp += dv_ol_cmp
    Vsat = soft_saturation(v_ol_state_cmp, Vout_max, Vout_min, soft_sat_alpha)
    Vout_pwm = limit_slew(Vsat, Vout_pwm, dt, slew_rate)

    y_out[n] = np.float32(Vout_pwm)

max_abs = np.max(np.abs(y_out))
if max_abs > 0:
    y_out *= (0.9/max_abs)

def float_to_pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return np.int16(np.round(x*32767.0))

audio = float_to_pcm16(y_out)
write("pwm_opamp_vco.wav", fs, audio)