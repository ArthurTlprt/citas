from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(192, 256, 3))

train_datagen = ImageDataGenerator(rescale=1./255)

# sets = ['train', 'test']
# paths = {'train': '../../augmented_x/', 'test': '../../dataset/balanced/test/'}
labels = ['clean', 'infected']


#for s in sets:
for l in labels:
    bs = len(glob('../../augmented_2/'+l+'/*'))

    train_generator = train_datagen.flow_from_directory(
    '../../augmented_2/'+l,
    target_size=(192, 256),
    batch_size=bs,
    class_mode='categorical')

    # extract features
    f = base_model.predict_generator(train_generator)


    np.save('features/f_{}_3.npy'.format(l), f)
    print("saved")
