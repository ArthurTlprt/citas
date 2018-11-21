from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from skimage.io import imread
from skimage.transform import resize

h5_path = 'brightfield.h5'
folder_path = 'dataset/'

def predict_from_directory(folder_path):
    images_path = glob(folder_path+'clean/*')
    images_path += glob(folder_path+'infected/*')
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(192, 256, 3))
    classifier = load_model(h5_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
    folder_path,
    target_size=(192, 256),
    class_mode='categorical',
    shuffle=False)

    f_test = base_model.predict_generator(test_generator)
    y_pred = classifier.predict(f_test)
    return images_path, y_pred

def predict(x):
    if len(x.shape) == 3:
        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(192, 256, 3))
    classifier = load_model(h5_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow(x)
    f_test = base_model.predict_generator(test_generator)
    y_pred = classifier.predict(f_test)
    return y_pred
