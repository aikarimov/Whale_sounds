import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import librosa

#Fs, y = wavfile.read('./91003005.wav')
Fs, y = wavfile.read('./1.wav')
# fs, y = wavfile.read('./62031001.wav')

# plot sample
fig, axs = plt.subplots(4, 1)
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
f, tss, Sxx = signal.spectrogram(y[1:M], Fs)
cf = axs[1].pcolormesh(tss, f, 10 * np.log10(Sxx), shading='gouraud')
fig.colorbar(cf, ax=axs[1])
axs[1].set_ylabel('Frequency [Hz]')
axs[1].set_xlabel('Time [sec]')

# construct filter
bands = (0, 1000, 1500, 29000, 30000, Fs / 2)  # freq. bands
desired = (0.1, 0.2, 1, 1, 0.0, 0.0)  # desired gain
fir_firls = signal.firls(173, bands, desired, fs=Fs)
z = signal.lfilter(fir_firls, [1], y[1:M])
plt.figure
axs[0].plot(t[1:M],z)
# plot spectrogram for filtered signal
f, tss, Sxx2 = signal.spectrogram(z, Fs)
cf = axs[2].pcolormesh(tss, f, 10 * np.log10(Sxx2), shading='gouraud')
fig.colorbar(cf, ax=axs[2])
axs[2].set_ylabel('Frequency [Hz]')
axs[2].set_xlabel('Time [sec]')
fig.tight_layout()

ysyn = librosa.griffinlim(Sxx)
#wavfile.write('./2.wav',Fs,z.astype(np.int16))

librosa.display.waveshow(ysyn, ax=axs[3])
axs[3].set_ylabel('Audio')
axs[3].set_xlabel('Time [sec]')
axs[3].set_xlim(0, 1.9)
plt.show()