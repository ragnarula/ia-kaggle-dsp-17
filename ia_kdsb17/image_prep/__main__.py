import argparse
import os
import pandas as pd
import numpy as np
import config
from itertools import chain
from numpy.lib.format import open_memmap
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

image_params = config.image_params
train_size, train, test_size, test = config.image_prep_function(args.data_dir, labels, **image_params)

print("Training dataset size: {}".format(train_size))
print("Test dataset size: {}".format(test_size))
one_patient = next(train)

h, w = one_patient[1].shape
print("Image rows: {} cols: {}".format(h, w))

train = chain([one_patient], train)

train_file = os.path.join(args.output_dir, "train_data.npy")
labels_file = os.path.join(args.output_dir, "labels.npy")
test_file = os.path.join(args.output_dir, "test_data.npy")
test_id_file = os.path.join(args.output_dir, "test_id.csv")

train_dat = open_memmap(train_file, dtype='float32', mode='w+', shape=(train_size, h, w))
train_labels = open_memmap(labels_file, dtype='int', mode='w+', shape=(train_size, 2))
test_dat = open_memmap(test_file, dtype='float32', mode='w+', shape=(test_size, h, w))
test_ids = []

for i, image in enumerate(train):
    print("Writing training sample {}".format(i), end='\r')
    train_dat[i, :, :] = image[1]
    label = labels.get_value(image[0], 'cancer')
    if label == 0:
        train_labels[i] = np.array([1, 0])
    else:
        train_labels[i] = np.array([0, 1])
print("", flush=True)

for i, image in enumerate(test):
    print("Writing test sample {}".format(i), end='\r')
    test_dat[i, :, :] = image[1]
    test_ids.append(image[0])
print("", flush=True)

del train_dat
del train_labels
del test_dat

test_id_df = pd.DataFrame({'id': test_ids})
test_id_df.to_csv(test_id_file, index=False)
#
# # print(labels.get_value('0015ceb851d7251b8f399e39779d1e7d', 'cancer'))