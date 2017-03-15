import bob.io.audio
import numpy as np
import scipy
from python_speech_features import mfcc
from python_speech_features import sigproc

direct = "/home/levy/VidTIMIT/fadg0/audio/sa1.wav"


signal = bob.io.audio.reader(direct)
audio = signal.load()

frames = sigproc.framesig(sig= signal.load()[0], frame_len= 640, frame_step = 320)

print(frames.shape)
zeros =  np.diag(np.hamming(640), k=0)
print("\n", zeros.shape)

S = np.dot(frames, zeros)
print("\nS shape: ", np.transpose(S))
