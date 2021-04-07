# import librosa
# filename2='D:/study/datasets/Audio/UrbanSound8K/audio/fold2/4201-3-1-0.wav'
# librosa_audio_data,librosa_sample_rate=librosa.load(filename2)

# # print(librosa_sample_rate)
# # print(librosa_audio_data)

# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(12,4))
# # plt.plot(librosa_audio_data)
# # plt.show()


# from scipy.io import wavfile as wav

# filename2='D:/study/datasets/Audio/UrbanSound8K/audio/fold2/4201-3-1-0.wav'
# # wave_sample_rate, wave_audio=wav.read(filename2)
# # print(wave_sample_rate)
# # print(wave_audio)

# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(12,4))
# # plt.plot(wave_audio)
# # plt.show()

# mfcc=librosa.feature.mfcc(y=librosa_audio_data,sr=librosa_sample_rate,n_mfcc=40)
# # print(mfcc.shape)
# print(mfcc)

from librosa.feature.spectral import mfcc
import pandas as pd
import os
import librosa
import numpy as  np
from tqdm import tqdm

audio_dataset_path="D:/study/datasets/Audio/UrbanSound8K/audio"
metadata=pd.read_csv("D:/study/datasets/Audio/UrbanSound8K/metadata/UrbanSound8K.csv")
print(metadata.head())




def features_extractor(filename):
    audio,sample_rate=librosa.load(filename,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features,axis=0)

    return mfccs_scaled_features

extracted_features=[]

for index_num,row in tqdm(metadata.iterrows()):
    file_name=os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])

extracted_features_df=pd.DataFrame(extracted_features,columns=["feature","class"])
(extracted_features_df.head())

x=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())



y=pd.array(pd.get_dummies(y))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics

num_labels=y.shape[1]

model=Sequential()
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')