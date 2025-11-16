# Spice TL074 (opamp RC relaxation) oscillator

import numpy as np
import struct, random
from scipy.io.wavfile import write

random.seed(123)

fs = 44100
dt = 1.0 / fs
duration = 5.0
N = int(duration * fs)

R1 = 10000.0
R2 = 10000.0
Vref = 0.0

C = 3.3e-7
Rint = 10000.0
tau = Rint * C

change_interval = 0.150
samples_per_change = int(change_interval * fs)

Rint_min = 5000.0
Rint_max = 20000.0

Vrail_pos = 15.0
Vrail_neg = -15.0
Vout_headroom = 1.5
Vout_max = Vrail_pos - Vout_headroom
Vout_min = Vrail_neg + Vout_headroom

A_ol_dc = 2e5

f_pole = 15.0
omega_p = 2.0 * np.pi * f_pole

slew_rate = 13e6

Rout = 60.0
Iout_limit = 0.010

V_offset = 0.003

I_b_pos = 65e-12
I_b_neg = 65e-12

Rin_pos = 1e12
Rin_neg = 1e12
Rid = 1e6

ICMR_min = -11.0
ICMR_max = 11.0

enable_noise = True
vnoise_rms = 20e-6

soft_sat_alpha = 0.45

Vout = Vout_max
Vc = 0.0
v_ol_state = 0.0

y_out = np.zeros(N, dtype=np.float32)
rng = np.random.default_rng(123)

def soft_saturation(v, vmax, vmin, alpha):
    center = 0.5 * (vmax + vmin)
    span = 0.5 * (vmax - vmin)
    if span <= 0:
        return np.clip(v, vmin, vmax)
    x = (v - center) / span
    return center + span * np.tanh(alpha * x)

def limit_slew(v_target, v_prev, dt, SR):
    dv = v_target - v_prev
    dv_max = SR * dt
    dv = np.clip(dv, -dv_max, dv_max)
    return v_prev + dv

def limit_output_current(v_unloaded, v_prev, Rout, Iout_lim):
    dv = v_unloaded - v_prev
    dv_max = Iout_lim * Rout
    dv = np.clip(dv, -dv_max, dv_max)
    return v_prev + dv

for n in range(N):
    if n % samples_per_change == 0:
        Rint = random.uniform(Rint_min, Rint_max)
        tau = Rint * C

    Vpos_in = (R2 * Vout + R1 * Vref) / (R1 + R2)
    Vpos_in += I_b_pos * (Rin_pos if np.isfinite(Rin_pos) else 0.0)

    Vneg_in = Vc + I_b_neg * (Rin_neg if np.isfinite(Rin_neg) else 0.0)

    Vcm = 0.5 * (Vpos_in + Vneg_in)
    A_ol = A_ol_dc
    extra_offset = 0.0
    if Vcm < ICMR_min or Vcm > ICMR_max:
        A_ol = max(1e3, A_ol_dc * 0.01)
        extra_offset = 1e-3 * np.sign(Vcm - np.clip(Vcm, ICMR_min, ICMR_max))

    vnoise = rng.normal(0.0, vnoise_rms) if enable_noise else 0.0
    Vdiff = (Vpos_in - Vneg_in) + V_offset + extra_offset + vnoise

    if np.isfinite(Rid) and Rid > 0:
        Vdiff *= (1.0 - dt / Rid)

    dv_ol = omega_p * (A_ol * Vdiff - v_ol_state) * dt
    v_ol_state += dv_ol

    Vsat = soft_saturation(v_ol_state, Vout_max, Vout_min, soft_sat_alpha)
    Vslew = limit_slew(Vsat, Vout, dt, slew_rate)
    Vout = limit_output_current(Vslew, Vout, Rout, Iout_limit)

    Vc += (Vout - Vc) / tau * dt
    y_out[n] = np.float32(Vc)

max_abs = np.max(np.abs(y_out))
norm = 0.9 / max_abs if max_abs > 0 else 1.0
y_out *= norm

def float_to_pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return np.int16(np.round(x * 32767.0))

audio = float_to_pcm16(y_out)

write("tl074_opamp_vco.wav", fs, audio)