import os
import numpy as np
from model import lstm, trainModel, plot, evaluate
import math
import sys
from helperFunctions import extractFeatures, getXandY, augmentData
from tensorflow import keras


if __name__ == "__main__":

    GENERATE_BOOL = False
    TRAIN_BOOL = False

    if len(sys.argv) > 1:
        if sys.argv[1] == "generateData":
            GENERATE_BOOL = True
        else:
            GENERATE_BOOL = False

        if sys.argv[2] == "Train":
            TRAIN_BOOL = True
        else:
            TRAIN_BOOL = False
    elif sys.argv[1] == "Train":
        TRAIN_BOOL = True



    BATCH_SIZE = 32
    EPOCHS = 100

    base_dir = os.getcwd()
    audio_dir = os.path.join(base_dir, 'Audio/Main')

    total = len(os.listdir(audio_dir))
    audio_files = os.listdir(audio_dir)


    if GENERATE_BOOL:
        print("------------Data Augmentation------------")
        augmented_audio = augmentData(audio_files, "Audio/Main/")
        print("Done")
        print("-------------Extracting 20 MFCC features per audio file-------------")
        extractFeatures(audio_files, "Audio/Main/")
        print("Done")
        print("-------------Extracting 20 MFCC features per augmented audio file-------------")
        extractFeatures(augmented_audio, None)
        print("Done")


    x_data, y_labels = getXandY(audio_dir)
    x_augmented_data = np.load('Datafiles/augmfccs.npy', allow_pickle=True)
    augmented_labels = np.load('Datafiles/auglabels.npy', allow_pickle=True)


    x_data = np.concatenate((x_data, x_augmented_data), axis=0)
    y_labels = np.concatenate((y_labels, augmented_labels), axis=0)



    SPLIT_NUM = math.floor(0.9*len(x_data))
    x_train = x_data[:SPLIT_NUM]
    y_train = y_labels[:SPLIT_NUM]

    x_test = x_data[SPLIT_NUM:]
    y_test = y_labels[SPLIT_NUM:]

    #print(y_train.shape)
    #exit()

    history = []
    if TRAIN_BOOL:
        print("-------------Compiling the model-------------")
        model, LR = lstm(input_shape=x_train[0].shape)
        print("Done")
        print("-------------Training the model-------------")
        history = trainModel(model, x_train, y_train, EPOCHS, BATCH_SIZE, LR=LR)
        print("Finished training the model, now saving to file")
        print("Evaluating the model")
        plot(history)


    model = keras.models.load_model('Models/model')
    evaluate(model, x_test, y_test)