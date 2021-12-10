from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import absl.logging
import tensorflow as tf

tf.keras.backend.clear_session()
absl.logging.set_verbosity(absl.logging.ERROR)

LR = 0.000005
optimizer = Adam(lr=LR)
optimizer_SGD = SGD(learning_rate=LR)


def lstm(input_shape):
    model = Sequential([
        LSTM(4000, input_shape=input_shape),
        Dropout(0.9),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model, LR


def trainModel(model, x, y, EPOCHS, batch_size, LR):
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=30)
    checkpoint = ModelCheckpoint(filepath='Models/model', monitor='val_loss', save_best_only=True)
    history = model.fit(x=x, y=y, epochs=EPOCHS, batch_size=batch_size, shuffle=True, validation_split=0.3,
                        callbacks=[es, checkpoint])
   # model.save("Models/model{}".format(LR))

    return history


def plot(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def evaluate(model, x_test, y_test):
    testing_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', testing_score[0])
    print('Test accuracy:', testing_score[1])

# pad audio to be same length? for f in *.wav; do ffmpeg -y -i "$f" -ss 0 -to 3 "${f%}";done
# normalize audio?


# figure out input for modeel? Maybe change how we record?

# ls -v | cat -n | while read n f; do mv -n "$f" "Lumos$n.wav"; done

#CHANGED ALL THE PATHS TO NEW AUDIO