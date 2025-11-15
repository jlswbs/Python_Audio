# NE5532 (opamp RC relaxation) oscillator and filter

import numpy as np
from scipy.io.wavfile import write

Vcc = 5.0
Vhigh = 13.0
Vlow = -13.0
GBW = 10e6
A0 = 200000
SR = 9e6

R_min = 5000.0
R_max = 20000.0
C = 1e-6

VTH = 3.1
VTL = 1.9

fs = 44100
Tmax = 5.0
dt = 1.0 / fs
N = int(Tmax * fs)

change_interval = 0.150
samples_per_change = int(change_interval * fs)

F_min = 200.0
F_max = 5000.0

y = np.zeros(N)
Vc = np.zeros(N)
out = np.zeros(N)

R_current = (R_min + R_max) / 2
fc = (F_min + F_max) / 2
alpha = dt / (1/(2*np.pi*fc) + dt)

fp = GBW / A0
tau = 1 / (2 * np.pi * fp)

Vout_int = Vlow
out_state = 1
Vc[0] = 0.0
out[0] = -1.0

def soft_clip(x, limit):
    return limit * np.tanh(x / limit)

for i in range(N):
    if i % samples_per_change == 0 and i > 0:
        R_current = np.random.uniform(R_min, R_max)
        fc = np.random.uniform(F_min, F_max)
        alpha = dt / (1/(2*np.pi*fc) + dt)

    dVc = (Vcc - Vc[i-1]) / (R_current * C) if out_state else (0 - Vc[i-1]) / (R_current * C)
    Vc[i] = Vc[i-1] + dVc * dt

    if out_state and Vc[i] >= VTH:
        out_state = 0
    elif not out_state and Vc[i] <= VTL:
        out_state = 1

    target = 1.0 if out_state else -1.0
    diff = target * A0
    dV = (diff - Vout_int) * (dt / tau)
    Vout_int += dV

    max_step = SR * dt
    prev_out = out[i-1] if i > 0 else out[0]
    Vout_int = np.clip(Vout_int, prev_out - max_step, prev_out + max_step)

    Vout_clipped = soft_clip(Vout_int, 1.0)
    Vout_real = (Vout_clipped + 1) * 0.5 * (Vhigh - Vlow) + Vlow
    out[i] = (Vout_real - Vlow) / (Vhigh - Vlow) * 2 - 1

    if i == 0:
        y[i] = alpha * out[i]
    else:
        y[i] = alpha * out[i] + (1 - alpha) * y[i-1]

audio = np.int16(y * 32767)
write("ne5532_oscillator.wav", fs, audio)