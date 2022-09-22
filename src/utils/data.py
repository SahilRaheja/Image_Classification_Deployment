
from keras.preprocessing.image import ImageDataGenerator

def load_data(train_path, test_path, shear_range, zoom_range, horizontal_flip, target_size, batch_size):
    train_datagen = ImageDataGenerator(rescale = 1./255, shear_range=shear_range, zoom_range=zoom_range,horizontal_flip=horizontal_flip)    
    test_datagen = ImageDataGenerator(rescale = 1./255) 

    training_set = train_datagen.flow_from_directory(train_path, target_size=target_size, batch_size=batch_size)
    test_set = train_datagen.flow_from_directory(test_path, target_size=target_size, batch_size=batch_size)

    print("Data Loaded Successfully")
    print(training_set.class_indices)
    
    return training_set, test_set

