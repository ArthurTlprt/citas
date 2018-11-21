from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Model
import h5py
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from multiprocessing import Pool

im_types = ['brightfield', 'darkfield', 'fluorescent']

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(192, 256, 3))

train_datagen = ImageDataGenerator(rescale=1./255)

labels = ['clean', 'tr4']
bs = 50

#dataset/box_1/test/brightfield
def feature_im_type(im_type):
	for l in labels:
		print(glob('../../dataset/box_1/test/'+im_type+'/'+l))
		train_generator = train_datagen.flow_from_directory(
		'../../augmented_box_1/'+im_type+'/'+l,
		target_size=(192, 256),
		batch_size=bs,
		class_mode='categorical')

		# extract features
		f = base_model.predict_generator(train_generator)
		print(f.shape)
		np.save('features/dataset_box_1/{}/f_{}.npy'.format(im_type, l), f)

		train_generator = train_datagen.flow_from_directory(
		'../../dataset/box_1/test/'+im_type+'/'+l,
		target_size=(192, 256),
		batch_size=len(glob('../../dataset/box_1/test/'+im_type+'/'+l+'/1/*.jpg')),
		class_mode='categorical')
		# extract features
		f = base_model.predict_generator(train_generator)
		print(f.shape)
		np.save('features/dataset_box_1/{}/ftest_{}.npy'.format(im_type, l), f)
		print("saved")


if __name__ == '__main__':
	#p = Pool(8)
	#p.map(feature_im_type, im_types)

	# for im_type in im_types:
	# 	feature_im_type(im_type)
	feature_im_type('fluo_brighter')
