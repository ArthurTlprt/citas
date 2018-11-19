from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

f_shape = (4, 6, 2048)

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
    model.add(Dense(2, activation='softmax'))
	# the learning rate is low because the model is little
    adam = Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])
    return model

def train_on_features(f_path, snapshot_name):
    x_train, y_train, x_test, y_test = shuffle_classes(f_path+'f_clean.npy', f_path+'f_tr4.npy', f_path+'ftest_clean.npy', f_path+'ftest_tr4.npy')
    model = get_model()
    hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
    model.save('snapshots/'+snapshot_name+'.h5')

    y_pred = model.predict(x_test)

    np.save('predictions/'+snapshot_name+'_pred.npy', y_pred)
    np.save('predictions/'+snapshot_name+'_test.npy', y_test)


if __name__ == '__main__':
    #train_on_features('features/dataset_box_1/brightfield/', 'brightfield')
    #train_on_features('features/dataset_box_1/darkfield/', 'darkfield')
    train_on_features('features/dataset_box_1/fluorescent/', 'fluorescent')
