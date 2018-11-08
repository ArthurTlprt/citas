from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

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

    return x, y


# x_test, y_test = shuffle_classes('f_test_clean.npy', 'f_test_infected.npy')
# print(x_test.shape)
# print(y_test.shape)
# x_train, y_train = shuffle_classes('f_train_clean.npy', 'f_train_infected.npy')

x, y = shuffle_classes('features/f_clean_2.npy', 'features/f_infected_2.npy')

l = int(x.shape[0]*0.8)
print(l)
x_train = x[:l]
y_train = y[:l]
x_test = x[l:]
y_test = y[l:]



model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=x_train[0].shape))
# let's add a fully-connected layer
model.add(Dense(2, activation='softmax'))

adam = Adam(lr=0.0001)
model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test))
model.save('best.h5')

y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

np.save('y_pred.npy', y_pred)
np.save('y_test.npy', y_test)
