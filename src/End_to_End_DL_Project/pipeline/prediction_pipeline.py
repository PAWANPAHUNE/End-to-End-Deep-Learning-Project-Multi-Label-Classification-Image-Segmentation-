import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # 1. Load the model
        model = load_model(os.path.join("model", "model.h5"))

        # 2. Preprocess and explicitly SCALE the image matrix
        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        # CRITICAL FIX: Match the training normalization step
        test_image = test_image / 255.0  

        # 3. Process index mapping
        result = np.argmax(model.predict(test_image), axis=1)
        predicted_idx = result[0]

        class_mapping = {
            0: 'Adenocarcinoma',
            1: 'Large Cell Carcinoma',
            2: 'Normal',
            3: 'Squamous Cell Carcinoma'
        }

        prediction = class_mapping.get(predicted_idx, "Unknown Class")
        return [{"image": prediction}]