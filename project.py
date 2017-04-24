import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.mixture import GaussianMixture
from python_speech_features import sigproc
from bob.learn import em

people = 5   #38
audio_by_person = 5   #7
NFFT = 4095
nfilt = 40
sample_rate = 32000 # 32kHz = 32000
frame_len = 0.020 * sample_rate # 20ms * sample_rate
frame_step = 0.010 * sample_rate # 10ms * sample_rate
num_ceps = 37
gmmList = []
gaussianMix = 12

def read_data():
    train = np.empty(shape=(0, num_ceps))
    directory = "/home/levy/VidTIMIT/Person"
    for i in range(1, people+1):
        teste = np.empty(shape=(0, num_ceps))
        for j in range(1, audio_by_person+1):
            direc = directory + str(i) + "/audio/" + str(j) + ".wav"
            rate, signal = wavfile.read(direc)
            mfccs = compute_mfccs(signal)
            teste = np.append(teste, mfccs, axis=0)
            train = np.append(train, mfccs, axis=0)
            print(mfccs.shape, "  /  ", i, " - ", j)
            kmeans = em.KMeansMachine(gaussianMix, num_ceps)
        em.train(em.KMeansTrainer(), kmeans, teste, max_iterations=500, convergence_threshold=1e-3)  # initialize the mean
        gmm = em.GMMMachine(gaussianMix, num_ceps)#Create a machine with (gaussianMix) Gaussian and feature dimensionality num_ceps
        gmm.means = kmeans.means
        em.train(em.ML_GMMTrainer(True, True, True), gmm, teste, max_iterations=500, convergence_threshold=1e-3)
        gmmList.append(gmm)
    return train
def pre_emphasis(signal):
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal
#the signal can be emphatized or not
#window lenght, window step and sample rate(kHz)) must be in seconds.

def enframe_signal(signal, window_size = frame_len, window_step = frame_step, sample_rate = sample_rate):
    frames = sigproc.framesig(signal, window_size, window_step)
    n_frames, frame_lenght = frames.shape  #return N_frames x number of samples per window
    #applying hamming window
    hamming = np.diag(np.hamming(frame_lenght))
    frames = np.dot(frames, hamming)
    return frames

def magspec(frames, NFFT):
    #Compute the magnitude spectrum of each frame in frames.
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def power_spec(frames, NFFT):
    #Compute the power spectrum of each frame (frame as rows)
    return 1.0/ (sample_rate*frame_len) * np.square(magspec(frames, NFFT))

def fft_and_filterbank(frames, NFFT = NFFT, nfilt = nfilt, sample_rate = sample_rate):
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

    '''plt.plot(freqs, fbank.T)
    plt.title("Filters")
    plt.grid(True)
    plt.show()'''
    filter_banks = np.dot(pow_frames, fbank.T) #Compute filterbank energies
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = np.log(filter_banks)
    #filter_banks = 20 * np.log10(filter_banks)  # dB

    '''plt.imshow(np.flipud(filter_banks.T))
    plt.title("Filter bank")
    plt.show()'''
    return filter_banks

#number 2-n_ceps are retained and the rest are descarted
def compute_mfccs(signal, window_size = frame_len, window_step = frame_step ,sample_rate = sample_rate,NFFT = NFFT,nceps = num_ceps):
    features = enframe_signal(signal) #Enframe and apply hamming window
    features = fft_and_filterbank(features)
    mfcc = dct(features, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-N_ceps
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


'''var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
print(var.pdf([1,0]))'''


list_mfccs = np.empty( shape= (0, num_ceps))
list_mfccs = read_data()
print(list_mfccs.shape)

kmeans = em.KMeansMachine(gaussianMix, num_ceps)
gmmMachine = em.GMMMachine(gaussianMix, num_ceps)
em.train(em.KMeansTrainer(), kmeans, list_mfccs, max_iterations=500, convergence_threshold= 1e-3)
gmmMachine.means = kmeans.means
# update means/variances/weights at each iteration

trainer = em.MAP_GMMTrainer(gmmMachine, update_means=True, update_variances=True, update_weights=True) # mean adaptation only
gmmAdapted = em.GMMMachine(gaussianMix, num_ceps) #Create a new machine for the MAP estimate
em.train(trainer, gmmAdapted, list_mfccs, max_iterations=500, convergence_threshold=1e-3)




diretorio = "/home/levy/VidTIMIT/Person3/audio/9.wav"
rate, signal = wavfile.read(diretorio)
test = compute_mfccs(signal)

'''
kmeans = em.KMeansMachine(16, num_ceps)
kmeansTrainer = em.KMeansTrainer()
em.train(kmeansTrainer, kmeans, test, max_iterations = 500, convergence_threshold = 1e-3) # Train the KMeansMachine


# TENTA INICIALIZAR OS PARAMATROS COM O KMEANS DO BOB E TESTAR COM O GMM DO SKLEARN
# FAZER O MAP (12-14) E PAG 20 DO SANDERSON. ALGORITMO MAP DO BOB. TENTAR CONSEGUIR AS MATRIZES DO MAP
#ME SINTO PERDIDO E ESPERO ESTAR FAZENDO A COISA CERTA. ESPERO QUE EU ESTEJA ENTENDENDO O QUE É PARA FAZER
#, POIS AS VEZES ME PERCO E  NÃO SEI ONDE ESTOU. ME AJUDA DEUS

gmm = em.GMMMachine(16 , num_ceps)
gmm.means = kmeans.means
trainer = em.ML_GMMTrainer(True, True, True) # update means/variances/weights at each iteration
em.train(trainer, gmm, test, max_iterations = 500, convergence_threshold = 1e-3)
print(gmm.get_gaussian(0).variance)
'''

'''
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
