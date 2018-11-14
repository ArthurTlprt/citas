from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

f_shape = (4, 6, 2048)

def shuffle_classes(path_clean, path_infected):
    # load .npy, merge two classes, hot encod label y and then shuffle the data

    # load 2 classes
    f_clean = np.load(path_clean, 'r')
    f_infected = np.load(path_infected, 'r')
    # define new shape of the concatenate classes for x
    x_shape = ((f_clean.shape[0]+f_infected.shape[0], f_clean.shape[1], f_clean.shape[2], f_clean.shape[3]))
    # define new array to get to concatenate the 2 classes
    x = np.zeros(x_shape)
    x[0:f_clean.shape[0]] = f_clean
    x[f_clean.shape[0]:] = f_infected
    # concatenate and then hot encod labels
    y = np.zeros(x_shape[0])
    y[f_clean.shape[0]:] = 1.
    y = to_categorical(y, num_classes=2)
    # shuffle the x and y the same
    indexes = np.arange(x_shape[0])
    np.random.shuffle(indexes)
    x = x[indexes]
    y = y[indexes]

    l = int(x.shape[0]*0.8)
    x_train = x[:l]
    y_train = y[:l]
    x_test = x[l:]
    y_test = y[l:]
    return x_train, y_train, x_test, y_test

def get_model():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=f_shape))
    # let's add a fully-connected layer
    model.add(Dense(2, activation='softmax'))

    adam = Adam(lr=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])
    return model

def train_on_features(f_path, snapshot_name):
    x_train, y_train, x_test, y_test = shuffle_classes(f_path+'f_clean.npy', f_path+'f_tr4.npy')
    model = get_model()
    hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
    model.save('snapshots/'+snapshot_name+'.h5')

    y_pred = model.predict(x_test)

    np.save('predictions/'+snapshot_name+'_pred.npy', y_pred)
    np.save('predictions/'+snapshot_name+'_test.npy', y_test)


if __name__ == '__main__':
    #train_on_features('features/dataset_3/brightfield/', 'brightfield')
    #train_on_features('features/dataset_3/darkfield/', 'darkfield')
    #train_on_features('features/dataset_3/fluorescent/', 'fluorescent')
