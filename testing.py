import os
import librosa
import keras
import numpy as np

SAMPLE_RATE = 44100

CLASSES = ["Alohomora", "Lumos", "Incendio"]
# CLASSES = ["Alohamora", "Incendio", "Lumos", "Nox", "Quietus", "Silencio", "Sonorus", "Wingardium Leviosa"]


path = "./Audio/Testing_audio/"
for file in os.listdir(path):

    data, sample_rate = librosa.load(path + file)
    features = librosa.feature.mfcc(y=data, n_mfcc=20, sr=SAMPLE_RATE)
    features = np.expand_dims(features, axis=0)
    features.reshape(1, -1)


    model = keras.models.load_model("Models/model")
    prediction = model.predict(features)
    predicted_class = CLASSES[np.argmax(prediction)]

    print("Actual: ", file)
    print("Predicted: ", predicted_class)
    print("*************************")



    # Why do testing features have different dimensions from the rest?
    # Check lengths of files