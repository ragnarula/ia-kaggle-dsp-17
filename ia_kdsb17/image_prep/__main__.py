import argparse
import os
import pandas as pd
import numpy as np
import ia_kdsb17.image_prep.helpers as imhelpers
from numpy.lib.format import open_memmap
import matplotlib.pyplot as plt
from ia_kdsb17.common import is_writeable

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Input data dir")
parser.add_argument("labels_csv", help="Labels CSV file")
parser.add_argument("output_dir", help="Dir to write the numpy arrays")

args = parser.parse_args()

if not is_writeable(args.output_dir):
    print('{} is not writeble'.format(args.data_dir))
    exit(1)

labels = pd.read_csv(args.labels_csv, index_col=0)

train_size, train, test_size, test = imhelpers.cropped_averaged_scaled(args.data_dir, 0.25, labels)
print(train_size, test_size)
one_patient = next(train)
# print(one_patient)
# fig = plt.figure()
#
# one_image = one_patient[1]
# # one_image.shape
# plt.imshow(one_image)
# plt.show()


print(one_patient[1].shape)

h, w = one_patient[1].shape

train_file = os.path.join(args.output_dir, "train.npy")
labels_file = os.path.join(args.output_dir, "labels.npy")
test_file = os.path.join(args.output_dir, "test.npy")

# train_dat = np.memmap(train_file, dtype='float32', mode='w+', shape=(train_size, h, w))
# train_labels = np.memmap(labels_file, dtype='int', mode='w+', shape=(train_size,))
# test_dat = np.memmap(test_file, dtype='float32', mode='w+', shape=(test_size, h, w))

train_dat = open_memmap(train_file, dtype='float32', mode='w+', shape=(train_size, h, w))
train_labels = open_memmap(labels_file, dtype='int', mode='w+', shape=(train_size,))
test_dat = open_memmap(test_file, dtype='float32', mode='w+', shape=(test_size, h, w))

for i, image in enumerate(train):
    train_dat[i, :, :] = image[1]
    train_labels[i] = labels.get_value(image[0], 'cancer')

for i, image in enumerate(test):
    test_dat[i, :, :] = image[1]

del train_dat
del train_labels
del test_dat
#
# # print(labels.get_value('0015ceb851d7251b8f399e39779d1e7d', 'cancer'))