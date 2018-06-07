from keras.models import load_model

IMAGE_LENGTH = 128
MODEL_PATH = 'model.hdf5'

classifier = load_model(MODEL_PATH)

from keras.preprocessing import image
import numpy as np
from glob import glob

labels = ['nsfw', 'sfw']
files = glob('*.jpeg')

predictions = []
for file in files:
    img = image.load_img(file, target_size=(IMAGE_LENGTH, IMAGE_LENGTH, 3))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255. 
    pred_prob = classifier.predict(img_tensor)[0][0]
    pred_class = classifier.predict_classes(img_tensor)[0][0]
    predictions.append((file, labels[pred_class], pred_prob))
for prediction in predictions:
    print("{0} is {1} with {2:.2f}% likelihood".format(prediction[0], prediction[1], prediction[2] * 100))