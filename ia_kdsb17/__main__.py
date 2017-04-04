import argparse
import logging
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D
from keras.optimizers import SGD

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Simple 2D CNN for Kaggle Data Science Bowl 2017')
parser.add_argument("train_data", help="The numpy binary file of training data")
parser.add_argument("test_data", help="The numpy binary file of test data")
parser.add_argument("labels", help="The numpy binary file pf labels")
args = parser.parse_args()

train_data = np.load(args.train_data, mmap_mode='r')
test_data = np.load(args.test_data, mmap_mode='r')
labels = np.load(args.labels, mmap_mode='r')

l, rows, cols = train_data.shape

train_data = train_data[..., np.newaxis]
train_data = train_data.reshape((l, rows, cols, 1))
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(rows, cols, 1), data_format='channels_last'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd)

model.fit(train_data, labels, batch_size=1, epochs=10)
