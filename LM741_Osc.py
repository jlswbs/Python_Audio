# LM741 (opamp RC relaxation) oscillator

import numpy as np
from scipy.io.wavfile import write

Vcc = 5.0
Vhigh = 4.0
Vlow  = 1.0

GBW = 1e6
A0 = 200000
SR = 0.5e6

R_min = 5000
R_max = 20000
C = 1e-6

VTH = 3.1
VTL = 1.9

fs = 44100
Tmax = 5.0
dt = 1 / fs
N = int(Tmax * fs)

change_interval = 0.150
samples_per_change = int(change_interval * fs)

fmin = 200.0
fmax = 5000.0

y = np.zeros(N)
Vc = np.zeros(N)
out = np.zeros(N)

Vc[0] = 0.0
out_state = 1

R_current = 10000
fc = np.random.uniform(fmin, fmax)
alpha = 0.0

Vout_int = 0.0

fp = GBW / A0
tau = 1 / (2 * np.pi * fp)

def soft_clip(x, limit):
    return limit * np.tanh(x / limit)

for i in range(1, N):

    if i % samples_per_change == 0:
        R_current = np.random.uniform(R_min, R_max)
        fc = np.random.uniform(fmin, fmax)
        alpha = dt / (1/(2*np.pi*fc) + dt)

    if out_state:
        dVc = (Vcc - Vc[i-1]) / (R_current * C)
    else:
        dVc = (0 - Vc[i-1]) / (R_current * C)

    Vc[i] = Vc[i-1] + dVc * dt

    if out_state and Vc[i] >= VTH:
        out_state = 0
    elif not out_state and Vc[i] <= VTL:
        out_state = 1

    target = 1.0 if out_state else -1.0

    Vin = target
    diff = Vin * A0

    dV = (diff - Vout_int) * (dt / tau)
    Vout_int += dV

    max_step = SR * dt
    Vout_int = np.clip(Vout_int,
                       out[i-1] - max_step,
                       out[i-1] + max_step)

    Vout = soft_clip(Vout_int, 1.0)

    Vout = (Vout + 1) * 0.5 * (Vhigh - Vlow) + Vlow

    out[i] = (Vout - Vlow) / (Vhigh - Vlow) * 2 - 1

    y[i] = alpha * out[i] + (1 - alpha) * y[i-1]

audio = np.int16(y * 32767)
write("lm741_relax_filtered.wav", fs, audio)
