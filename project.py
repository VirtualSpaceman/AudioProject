import bob.io.audio
import bob.bio.gmm
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
from scipy.signal import spectrogram
from python_speech_features import  mfcc
from python_speech_features import fbank
from python_speech_features import sigproc
import os.path

people = 38
audio_by_person = 10
NFFT = 4095
nfilt = 40
sample_rate = 32000 # 32kHz = 32000
window_size = 0.020 * sample_rate # 20ms * sample_rate
window_step = 0.010 * sample_rate # 10ms * sample_rate
num_ceps = 37

def read_data():
    directory = "/home/levy/VidTIMIT/Person"
    for i in range(1, people+1):
        for j in range(1, audio_by_person+1):
            direc = directory + str(i) + "/audio/" + str(j) + ".wav"
            my_file = os.path.isfile(direc)
            print(my_file," ", str(i), "  ", str(j))


def pre_emphasis(signal):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal
#the signal can be emphatized or not
#window lenght, window step and sample rate(kHz)) must be in seconds.

def enframe_signal(signal, window_size, window_step, sample_rate):
    frames = sigproc.framesig(signal, window_size, window_step)
    n_frames, frame_lenght = frames.shape  #return N_frames x number of samples per window
    #applying hamming window
    hamming = np.diag(np.hamming(frame_lenght))
    frames = np.dot(frames, hamming)
    return frames

def magspec(frames, NFFT):
    #Compute the magnitude spectrum of each frame in frames.
    complex_spec = np.fft.rfft(frames, NFFT)
    print(complex_spec.shape)
    return np.absolute(complex_spec)

def power_spec(frames, NFFT):
    #Compute the power spectrum of each frame (frame as rows)
    return 1.0/ (sample_rate*window_size) * np.square(magspec(frames, NFFT))

def fft_and_filterbank(frames, NFFT, nfilt, sample_rate):
    mag_frames = magspec(frames, NFFT) #np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = power_spec(frames, NFFT)#((1.0 / window_size) * ((mag_frames) ** 2))  # Power Spectrum

    #Compute mel filterbank
    low_freq_mel = 0
    freqs = np.linspace(1.0, sample_rate / 2.0, NFFT/2 +1)
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center of frequency
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    plt.plot(freqs, fbank.T)
    plt.title("Filters")
    plt.grid(True)
    plt.show()
    filter_banks = np.dot(pow_frames, fbank.T) #Compute filterbank energies
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    #filter_banks = np.log(filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB

    plt.imshow(np.flipud(filter_banks.T))
    plt.title("Filter bank")
    plt.show()
    return filter_banks

#number 2-n_ceps are retained and the rest are descarted
def compute_mfccs(filter_banks, num_ceps):
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-N_ceps
    return mfcc
#---------------------------------------------------------------------------

def hertz_to_mel(freq):
    """Returns mel-frequency from linear frequency input.

    Parameter
    ---------
    freq : scalar or ndarray
        Frequency value or array in Hz.

    Returns
    -------
    mel : scalar or ndarray
        Mel-frequency value or ndarray in Mel

    """
    return 2595.0 * np.log10(1 + (freq/700.0))

def mel_to_hertz(mel):
    """Returns frequency from mel-frequency input.

    Parameter
    ---------
    mel : scalar or ndarray
        Mel-frequency value or ndarray in Mel

    Returns
    -------
    freq : scalar or ndarray
        Frequency value or array in Hz.

    """
    return 700.0 * (10**(mel/2595.0)) - 700.0

def melfrequencies_mel_filterbank(num_bands, freq_min, freq_max, num_fft_bands):
    """Returns centerfrequencies and band edges for a mel filter bank
    Parameters
    ----------
    num_bands : int
        Number of mel bands.
    freq_min : scalar
        Minimum frequency for the first band.
    freq_max : scalar
        Maximum frequency for the last band.
    num_fft_bands : int
        Number of fft bands.

    Returns
    -------
    center_frequencies_mel : ndarray
    lower_edges_mel : ndarray
    upper_edges_mel : ndarray

    """

    mel_max = hertz_to_mel(freq_max)
    mel_min = hertz_to_mel(freq_min)
    delta_mel = abs(mel_max - mel_min) / (num_bands + 1.0)
    frequencies_mel = mel_min + delta_mel*np.arange(0, num_bands+2)
    lower_edges_mel = frequencies_mel[:-2]
    upper_edges_mel = frequencies_mel[2:]
    center_frequencies_mel = frequencies_mel[1:-1]

    return center_frequencies_mel, lower_edges_mel, upper_edges_mel

def compute_melmat(num_mel_bands=12, freq_min=64, freq_max=8000,
                   num_fft_bands=513, sample_rate=16000):
    """Returns tranformation matrix for mel spectrum.

    Parameters
    ----------
    num_mel_bands : int
        Number of mel bands. Number of rows in melmat.
        Default: 24
    freq_min : scalar
        Minimum frequency for the first band.
        Default: 64
    freq_max : scalar
        Maximum frequency for the last band.
        Default: 8000
    num_fft_bands : int
        Number of fft-frequenc bands. This ist NFFT/2+1 !
        number of columns in melmat.
        Default: 513   (this means NFFT=1024)
    sample_rate : scalar
        Sample rate for the signals that will be used.
        Default: 44100

    Returns
    -------
    melmat : ndarray
        Transformation matrix for the mel spectrum.
        Use this with fft spectra of num_fft_bands_bands length
        and multiply the spectrum with the melmat
        this will tranform your fft-spectrum
        to a mel-spectrum.

    frequencies : tuple (ndarray <num_mel_bands>, ndarray <num_fft_bands>)
        Center frequencies of the mel bands, center frequencies of fft spectrum.

    """
    center_frequencies_mel, lower_edges_mel, upper_edges_mel =  \
        melfrequencies_mel_filterbank(
            num_mel_bands,
            freq_min,
            freq_max,
            num_fft_bands
    )

    len_fft = float(num_fft_bands) / sample_rate
    center_frequencies_hz = mel_to_hertz(center_frequencies_mel)
    lower_edges_hz = mel_to_hertz(lower_edges_mel)
    upper_edges_hz = mel_to_hertz(upper_edges_mel)
    freqs = np.linspace(1.0, sample_rate/2.0, num_fft_bands)
    melmat = np.zeros((num_mel_bands, num_fft_bands))

    for imelband, (center, lower, upper) in enumerate(zip(
            center_frequencies_hz, lower_edges_hz, upper_edges_hz)):

        left_slope = (freqs >= lower)  == (freqs <= center)
        melmat[imelband, left_slope] = (
            (freqs[left_slope] - lower) / (center - lower)
        )

        right_slope = (freqs >= center) == (freqs <= upper)
        melmat[imelband, right_slope] = (
            (upper - freqs[right_slope]) / (upper - center)
        )

    return melmat, (center_frequencies_mel, freqs)
#----------------------------------------------------------------------------

read_data()

'''
diretorio = "/home/levy/mcem0_sa1.wav"

sample_rate, signal = wavfile.read(diretorio)

frames = enframe_signal(signal, window_size, window_step, sample_rate)
filter_banks =  fft_and_filterbank(frames, NFFT, nfilt, sample_rate)
mfccs = compute_mfccs(filter_banks, 37)
mfccs -= (np.mean(mfccs, axis=0) + 1e-8) #mean normalization
plt.imshow(np.flipud(mfccs.T))
plt.title("MFCCS")
plt.show()
print(mfccs.shape)
'''

'''

mfcc_feat = mfcc(signal, 32000, 0.020, 0.01,nfilt= 17, nfft= 2048, numcep= 37)
print(mfcc_feat.shape)
plt.imshow(np.flipud(mfcc_feat.T), aspect= 'auto', interpolation= 'none')
plt.show()
'''


'''
melmat, (melfreq, fftfreq) = compute_melmat(17, 0, sample_rate/2, num_fft_bands= 2048, sample_rate= sample_rate)
plt.plot(fftfreq, melmat.T)
print()
print( "Linhas:",len(fftfreq) , " -  ", melmat.T.shape)
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("Amplitude")
plt.show()

'''
