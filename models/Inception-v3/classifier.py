from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

f_shape = (4, 6, 2048)

types = ['brightfield/', 'darkfield/', 'fluo_brighter/']

def mix_types(path):
	x_l_train, y_l_train, x_l_test, y_l_test = ([] for i in range(4))
	# getting all the type
	for t in types:
		f_path = path+t
		x_train_t, y_train_t, x_test_t, y_test_t = shuffle_classes(f_path+'f_clean.npy', f_path+'f_tr4.npy', f_path+'ftest_clean.npy', f_path+'ftest_tr4.npy')
		x_l_train.append(x_train_t)
		y_l_train.append(y_train_t)
		x_l_test.append(x_test_t)
		y_l_test.append(y_test_t)
	# computing the number of images to know the shapes
	n_train = 0
	for p in x_l_train:
		n_train += p.shape[0]
	n_test = 0
	for p in x_l_test:
		n_test += p.shape[0]
	# initializing big empty x and y arrays to contain the mixed types
	shape = (n_train, 4, 6, 2048)
	x_train = np.zeros(shape)
	y_train = np.zeros((n_train, 2))
	shape = (n_test, 4, 6, 2048)
	x_test = np.zeros(shape)
	y_test = np.zeros((n_test, 2))

	train_thres = test_thres = 0
	for i, t in enumerate(x_l_train):
		i_train = x_l_train[i].shape[0]
		i_test = x_l_test[i].shape[0]

		x_train[train_thres:train_thres+i_train] = x_l_train[i]
		y_train[train_thres:train_thres+i_train] = y_l_train[i]
		train_thres += i_train
		x_test[test_thres:test_thres+i_test] = x_l_test[i]
		y_test[test_thres:test_thres+i_test] = y_l_test[i]
		test_thres = i_test

	return x_train, y_train, x_test, y_test


def shuffle_classes(path_clean, path_infected, path_test_clean, path_test_tr4):
	# load .npy, merge two classes, hot encod label y and then shuffle the data

	# load 2 classes
	f_clean = np.load(path_clean, 'r')
	f_infected = np.load(path_infected, 'r')
	# define new shape of the concatenate classes for x
	x_shape = ((f_clean.shape[0]+f_infected.shape[0], f_clean.shape[1], f_clean.shape[2], f_clean.shape[3]))
	# define new array to get to concatenate the 2 classes
	x_train = np.zeros(x_shape)
	x_train[0:f_clean.shape[0]] = f_clean
	x_train[f_clean.shape[0]:] = f_infected
	# concatenate and then hot encod labels
	y_train = np.zeros(x_shape[0])
	y_train[f_clean.shape[0]:] = 1.
	y_train = to_categorical(y_train, num_classes=2)
	# shuffle the x and y the same
	indexes = np.arange(x_shape[0])
	np.random.shuffle(indexes)
	x_train = x_train[indexes]
	y_train = y_train[indexes]
	# loading test data
	ftest_clean = np.load(path_test_clean, 'r')
	ftest_tr4 = np.load(path_test_tr4, 'r')
	# then create the arrays to store the features and the labels
	x_test = np.zeros((ftest_clean.shape[0]+ftest_tr4.shape[0], ftest_clean.shape[1], ftest_clean.shape[2], ftest_clean.shape[3]))
	x_test[:ftest_clean.shape[0]] = ftest_clean
	x_test[ftest_clean.shape[0]:] = ftest_tr4
	y_test = np.zeros(ftest_clean.shape[0]+ftest_tr4.shape[0])
	y_test[ftest_clean.shape[0]:] = 1.
	y_test = to_categorical(y_test, num_classes=2)

	return x_train, y_train, x_test, y_test

def get_model():
	model = Sequential()
	model.add(GlobalAveragePooling2D(input_shape=f_shape))
	# Then output the 2 probabilities
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))
	# the learning rate is low because the model is little
	adam = Adam(lr=0.0001)
	model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])
	return model

def train_on_features(f_path, snapshot_name):
	#x_train, y_train, x_test, y_test = shuffle_classes(f_path+'f_clean.npy', f_path+'f_tr4.npy', f_path+'ftest_clean.npy', f_path+'ftest_tr4.npy')
	x_train, y_train, x_test, y_test = mix_types(f_path)
	model = get_model()
	check = ModelCheckpoint('snapshots/'+snapshot_name+'.h5', monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
	hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[check])

	y_pred = model.predict(x_test)

	np.save('predictions/'+snapshot_name+'_pred.npy', y_pred)
	np.save('predictions/'+snapshot_name+'_test.npy', y_test)


if __name__ == '__main__':
    #train_on_features('features/dataset_box_1/brightfield/', 'brightfield')
    #train_on_features('features/dataset_box_1/darkfield/', 'darkfield')
    #train_on_features('features/dataset_box_1/fluorescent/', 'fluorescent')
	#train_on_features('features/dataset_box_1/fluo_brighter/', 'fluo_brighter')
	train_on_features('features/dataset_box_1/', 'mixed')
