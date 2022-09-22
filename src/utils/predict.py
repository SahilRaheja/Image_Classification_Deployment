import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# test_image = image.load_img(r'C:\Users\SAHIL\Downloads\sunflower.jpg', target_size=(64, 64))

class Flower:

    def __init__(self,filename):
        self.filename = filename

    def prediction(self):
        model = load_model('artifacts/model.h5')

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)


        # {'daisy': 0, 'roses': 1, 'sunflowers': 2}

        if result[0][0] == 1.0:
            prediction = 'daisy'
            return [{'Prediction': prediction}]

        elif result[0][1] == 1.0:
            prediction = 'roses'
            return [{'Prediction': prediction}]

        elif result[0][2] == 1.0:
            prediction = 'sunflower'
            return [{'Prediction': prediction}]

        else:
            prediction = 'Unknown Image'
            return [{'Prediction': prediction}]
    

