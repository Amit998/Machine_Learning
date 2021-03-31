import matplotlib.pyplot as plt
import IPython as ipd
import librosa
import librosa.display
import pandas as pd

# librosa.load()

metadata=pd.read_csv("")


metadata['class'].value_counts()


filename="D:/study/datasets/Audio/Dog-barking-sound.wav"

data,sample_rate=librosa.load(filename)
# librosa.display.waveplot(data,sr=sample_rate)
# plt.figure(figsize=(14,5))
# ipd.audio(filename)
# plt.show()

print(sample_rate)
print(data)

# from scipy.io import wavfile as wav
# wave_sample_rate,wav_audio=wav.read(filename)

# print(wave_sample_rate)