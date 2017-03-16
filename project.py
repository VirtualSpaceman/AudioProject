from bob.bio import gmm
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from scipy.io import wavfile
import scipy
from python_speech_features import mfcc
from python_speech_features import sigproc
direct = "/home/levy/VidTIMIT/fadg0/audio/sa1.wav"

#axis = 0 (colunas) e axis = 1 (lihnas)
#Global variables
frame_length = 0.020 #milliseconds   it means (sample_rate x frame_length) samples.
frame_step = 0.010 #milliseconds     it means a new samples array after (frame_step x sample_rate) samples
NFFT = 2048  #Sanderson uses K = 2048 (FFT)
nfilt = 40

sample_rate, signal = wavfile.read(direct)

#emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
signal = signal/ (2. ** 15)

'''
timeArray = np.arange(0, len(signal), 1)
timeArray = timeArray / sample_rate
timeArray = timeArray * 1000  #scale to milliseconds
plt.plot(timeArray, signal)
plt.xlabel("Time(ms)")
plt.ylabel("Amplitude ")
plt.show()


'''

#enframing signal
frames = sigproc.framesig(signal, frame_len= sample_rate*frame_length, frame_step = sample_rate*frame_step)
ham_diag = np.diag(np.hamming(len(frames[0])))
frames = np.dot(frames, ham_diag)  #applying hamming window



#FFT
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum



#FilterBank on a Mel-Scale
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB



'''
zeros =  np.diag(np.hamming(640), k=0)
print("\n", zeros.shape)
frames


S = np.dot(frames, zeros)
print("\nS shape: ", np.transpose(S))
'''
