# Lunetta (schmitt trigger) oscillator and filter

import numpy as np
from scipy.io.wavfile import write

Vcc = 5.0
VTH = 3.3
VTL = 1.7
R_min = 5000.0
R_max = 20000.0
C = 1e-6
fs = 44100
Tmax = 5.0
N = int(Tmax * fs)
dt = 1.0 / fs

y = np.zeros(N)
out = np.zeros(N)
Vc = 0.0
out_state = 1

change_interval = 0.150
samples_per_change = int(change_interval * fs)

F_min = 200.0
F_max = 5000.0

R = (R_min + R_max) / 2
tau = R * C
exp_factor = np.exp(-dt / tau)
fc = (F_min + F_max) / 2
alpha = dt / (1/(2*np.pi*fc) + dt)

for i in range(N):
    if i % samples_per_change == 0:
        R = np.random.uniform(R_min, R_max)
        tau = R * C
        exp_factor = np.exp(-dt / tau)
        fc = np.random.uniform(F_min, F_max)
        alpha = dt / (1/(2*np.pi*fc) + dt)

    if out_state and Vc >= VTH:
        out_state = 0
    elif not out_state and Vc <= VTL:
        out_state = 1

    target = Vcc if out_state else 0.0
    Vc = target + (Vc - target) * exp_factor
    y[i] = 1.0 if out_state else -1.0

    if i == 0:
        out[i] = alpha * y[i]
    else:
        out[i] = alpha * y[i] + (1 - alpha) * out[i-1]

audio = np.int16(out * 32767)
write("lunetta_oscillator.wav", fs, audio)