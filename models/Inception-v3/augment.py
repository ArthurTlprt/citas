from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import h5py
import numpy as np
import matplotlib.pyplot as plt

bs=8

train_datagen = ImageDataGenerator(
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        rotation_range=45)

labels = ['clean', 'tr4']

for l in labels:
    train_generator = train_datagen.flow_from_directory(
    '../../dataset/dataset_3/fluorescent/'+l,
    target_size=(192, 256),
    batch_size=bs,
    class_mode='categorical',
    save_to_dir='../../augmented_3/fluorescent/'+l)
    for i in range(20):
        x, y = train_generator.next()
