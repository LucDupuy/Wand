import os
import re
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

"""
This function will extract useful features from each audio file that can be fed into the neural network.
It will also save them to a file to save computation time later on.
"""


def extractFeatures(files, path):
    mfccs = []
    augmented_mfccs = []

    if path == None:
        for _, file in enumerate(tqdm(files)):
            feature = librosa.feature.mfcc(y=file, n_mfcc=20, sr=44100)
            augmented_mfccs.append(feature)

        np.save('Datafiles/augmfccs.npy', augmented_mfccs, allow_pickle=True)

    else:

        for _, file in enumerate(tqdm(files)):
            data, sample_rate = librosa.load(path + file)
            feature = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20)
            mfccs.append(feature)

        np.save('Datafiles/mfccs.npy', mfccs, allow_pickle=True)


"""Splitting the data into the audio files and their matching labels"""


def getXandY(dir):
    files = os.listdir(dir)
    labels = []

    for _, file in enumerate(files):
        name = re.split(r'(^[^\d]+)', file)[1:][0]
        labels.append(name)

    x_train = np.load('Datafiles/mfccs.npy', allow_pickle=True)

    # Hot encode the y labels in order to pass into the network
    lb = LabelEncoder()
    labels = to_categorical(lb.fit_transform(labels))

    return x_train, labels


def augmentData(files, path):
    labels = []
    new_audio_clips = []

    for _, file in enumerate(tqdm(files)):
        name = re.split(r'(^[^\d]+)', file)[1:][0]
        audio, sr = librosa.load(path + file)
        noised_audio = audio + 0.005 * np.random.normal(0, 1, len(audio))
        noised_audio2 = audio + 0.002 * np.random.normal(0, 1, len(audio))
        noised_audio3 = audio + 0.01 * np.random.normal(0, 1, len(audio))
        shifted_audio = np.roll(audio, sr)
        pitched_audio = librosa.effects.pitch_shift(audio, sr, n_steps=-5)
        pitched_audio2 = librosa.effects.pitch_shift(audio, sr, n_steps=+5)
        pitched_audio3 = librosa.effects.pitch_shift(audio, sr, n_steps=+10)
        noise_pitch = librosa.effects.pitch_shift(noised_audio, sr, n_steps=-5)
        noise_pitch2 = librosa.effects.pitch_shift(noised_audio2, sr, n_steps=5)
        noise_pitch3 = librosa.effects.pitch_shift(noised_audio3, sr, n_steps=-5)

        new_audio_clips.append(noised_audio)
        new_audio_clips.append(noised_audio2)
        new_audio_clips.append(noised_audio3)
        new_audio_clips.append(shifted_audio)
        new_audio_clips.append(pitched_audio)
        new_audio_clips.append(pitched_audio2)
        new_audio_clips.append(pitched_audio3)
        new_audio_clips.append(noise_pitch)
        new_audio_clips.append(noise_pitch2)
        new_audio_clips.append(noise_pitch3)

        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)
        labels.append(name)

    lb = LabelEncoder()
    labels = to_categorical(lb.fit_transform(labels))

    np.save('Datafiles/auglabels.npy', labels, allow_pickle=True)

    return new_audio_clips