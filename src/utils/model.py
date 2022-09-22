
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

def load_model(loss_function, optimizer, metrics):

    classifier = Sequential()

    classifier.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)))
    classifier.add(MaxPooling2D(2,2))
    classifier.add(Conv2D(64, (3,3), activation='relu'))

    classifier.add(Flatten())
    classifier.add(Dense(128, activation='relu'))
    classifier.add(Dense(3, activation='softmax'))

    classifier.compile(loss=loss_function, optimizer=optimizer, metrics=metrics)

    classifier.summary()

    return classifier

