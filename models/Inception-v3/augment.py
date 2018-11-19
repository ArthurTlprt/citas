from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


bs=60

train_datagen = ImageDataGenerator(
	zoom_range=0.2,
	vertical_flip=True,
	horizontal_flip=True,
	rotation_range=45)

labels = ['clean', 'tr4']
im_types = ['brightfield', 'darkfield', 'fluorescent']


def augment_type(im_type):
	for l in labels:
		train_generator = train_datagen.flow_from_directory(
		'../../dataset/box_1/'+im_type+'/'+l,
		target_size=(192, 256),
		batch_size=bs,
		class_mode='categorical',
		save_to_dir='../../augmented_box_1/'+im_type+'/'+l)
		for i in range(10):
			x, y = train_generator.next()

p = Pool(8)
p.map(augment_type, im_types)
