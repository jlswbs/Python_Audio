# NE555 (astable multivibrator) oscillator

import numpy as np
from scipy.io.wavfile import write

Vcc = 5.0
R1_min = 1000
R1_max = 10000
R2_min = 1000
R2_max = 10000
C = 1e-6

fs = 44100
Tmax = 5.0
dt = 1/fs
N = int(Tmax*fs)

change_interval = 0.150
samples_per_change = int(change_interval * fs)

fmin = 200.0
fmax = 5000.0
y = np.zeros(N)
alpha = 0.0

Vc = np.zeros(N)
out = np.zeros(N)
Vc[0] = 0.0
out_state = 1
R1 = np.random.uniform(R1_min, R1_max)
R2 = np.random.uniform(R2_min, R2_max)
fc = np.random.uniform(fmin, fmax)

VTH = 2/3 * Vcc
VTL = 1/3 * Vcc

for i in range(1, N):
    if i % samples_per_change == 0:
        R1 = np.random.uniform(R1_min, R1_max)
        R2 = np.random.uniform(R2_min, R2_max)
        fc = np.random.uniform(fmin, fmax)
        alpha = dt / (1/(2*np.pi*fc) + dt)

    if out_state:
        dVc = ((Vcc - Vc[i-1]) / ((R1 + R2) * C)) * dt
        Vc[i] = Vc[i-1] + dVc
        if Vc[i] >= VTH:
            out_state = 0
    else:
        dVc = ((0 - Vc[i-1]) / (R2 * C)) * dt
        Vc[i] = Vc[i-1] + dVc
        if Vc[i] <= VTL:
            out_state = 1

    out[i] = 1.0 if out_state else -1.0

    y[i] = alpha * out[i] + (1 - alpha) * y[i-1]

audio = np.int16(y * 32767)

write("ne555_oscillator_filtered.wav", fs, audio)
