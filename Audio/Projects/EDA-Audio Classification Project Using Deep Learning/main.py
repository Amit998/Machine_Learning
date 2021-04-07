import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import pandas as pd

# librosa.load()

metadata=pd.read_csv("D:/study/datasets/Audio/UrbanSound8K/metadata/UrbanSound8K.csv")



# metadata['class'].value_counts()


# filename='D:/study/datasets/Audio/Dog-barking-sound.wav'
filename2='D:/study/datasets/Audio/UrbanSound8K/audio/fold2/4201-3-1-0.wav'

# import wavio as wav
from scipy.io import wavfile as wav
wave_sample_rate, wave_audio=wav.read(filename2)

print(wave_sample_rate)

# data,sample_rate=librosa.load(filename)
# librosa.display.waveplot(data,sr=sample_rate)
# plt.figure(figsize=(14,5))
# ipd.audio(filename)
# plt.show()

# print(sample_rate)
# print(data)

# from scipy.io import wavfile as wav
# wave_sample_rate,wav_audio=wav.read(filename)
# 
# print(wave_sample_rate)