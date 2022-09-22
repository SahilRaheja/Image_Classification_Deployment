import numpy as np

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

import warnings
warnings.filterwarnings("ignore")

classifier = Sequential()

classifier.add(Conv2D(32, (3,3), activation='relu', input_shape=(64,64, 3)))
classifier.add(MaxPooling2D(2,2))
classifier.add(Conv2D(64, (3,3), activation='relu'))

classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(3, activation='softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

classifier.summary()

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=0.2, zoom_range=0.2,horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('flower_photos/train', target_size=(64, 64), batch_size=32) #, class_mode='binary'
test_set = test_datagen.flow_from_directory('flower_photos/validation', target_size=(64, 64), batch_size=32) #, class_mode='binary'

model = classifier.fit_generator(training_set, steps_per_epoch=2000, epochs = 2, validation_data=test_set, validation_steps=1000)

classifier.save('artifacts/model.h5')
print('Model saved successfully')

print(training_set.class_indices)


