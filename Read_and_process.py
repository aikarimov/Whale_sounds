import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import noisereduce as nr

Fs, y = wavfile.read('./91003005.wav')
#Fs, y = wavfile.read('./1.wav')
# Fs, y = wavfile.read('./62031001.wav')

# plot sample
fig, axs = plt.subplots(3, 1)
h = 1 / Fs
t = np.arange(0, len(y) * h, h)
N = len(y)
#M = round(N / 10)
M = N-1
axs[0].plot(t[1:M], y[1:M])
axs[0].set_ylabel('Audio')
axs[0].set_xlabel('Time [sec]')
axs[0].set_xlim(0, t[M])

# plot spectrogram
nperseg = 128
#f, tss, Sxx = signal.spectrogram(y[1:M], Fs)
fss, tss, Sxx = signal.stft(y[1:M], fs=Fs, nperseg=nperseg)
cf = axs[1].pcolormesh(tss, fss, 10 * np.log10(np.abs(Sxx)), shading='gouraud')

fig.colorbar(cf, ax=axs[1])
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')

#noise reduction
y0 = nr.reduce_noise(y[1:M],sr=Fs,stationary=True,n_fft=1024, n_std_thresh_stationary = 1.8,  time_mask_smooth_ms = 25)
f, tn, Sn = signal.stft(y0, fs=Fs, nperseg=nperseg) #noise

cf = axs[2].pcolormesh(tss, fss, 10 * np.log10(np.abs(Sn)) , shading='gouraud')
fig.colorbar(cf, ax=axs[2])
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_xlabel('Time [sec]')
axs[0].plot(t[1:M],y0)

wavfile.write('./2.wav',Fs,y0.astype(np.int16))

plt.show()