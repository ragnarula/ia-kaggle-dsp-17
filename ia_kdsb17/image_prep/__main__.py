import argparse
import os
import pandas as pd
import numpy as np
import config
import tables
from itertools import chain
from numpy.lib.format import open_memmap
from ia_kdsb17.common import is_writeable

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="Input data dir")
parser.add_argument("labels_csv", help="Labels CSV file")
parser.add_argument("output_dir", help="Dir to write the numpy arrays")

args = parser.parse_args()

if not is_writeable(args.output_dir):
    print('{} is not writeable'.format(args.data_dir))
    exit(1)

labels = pd.read_csv(args.labels_csv, index_col=0)

image_params = config.image_params
train, test = config.image_prep_function(args.data_dir, labels, **image_params)

one_patient = next(train)

h, w = one_patient[1].shape
print("Image rows: {} cols: {}".format(h, w))

train = chain([one_patient], train)

train_file = os.path.join(args.output_dir, "train_data.hdf5")
labels_file = os.path.join(args.output_dir, "labels.hdf5")
test_file = os.path.join(args.output_dir, "test_data.hdf5")
test_id_file = os.path.join(args.output_dir, "test_id.csv")

# train_dat = open_memmap(train_file, dtype='float32', mode='w+', shape=(-1, 1, h, w))
# train_labels = open_memmap(labels_file, dtype='int', mode='w+', shape=(-1, 2))
# test_dat = open_memmap(test_file, dtype='float32', mode='w+', shape=(-1, 1, h, w))

tables_file = os.path.join(args.output_dir, 'image_data.h5')

image_data_table = tables.open_file(tables_file, mode='w+')
image_atom = tables.Float32Atom()
int_atom = tables.Int8Atom()

train_data = image_data_table.create_earray(image_data_table.root, 'train_data', atom=image_atom, shape=(0, 1, h, w))
train_labels = image_data_table.create_earray(image_data_table.root, 'train_labels', atom=int_atom, shape=(0, 2))
test_data = image_data_table.create_earray(image_data_table.root, 'test_data', atom=image_atom, shape=(0, 1, h, w))
test_ids = []

for i, image in enumerate(train):
    print("Writing training sample {}".format(i), end='\r')
    train_data.append(image[1].reshape(1, 1, h, w))
    label = labels.get_value(image[0], 'cancer')
    if label == 0:
        train_labels.append(np.array([1, 0]).reshape(1, 2))
    else:
        train_labels.append(np.array([0, 1]).reshape(1, 2))
print("", flush=True)

for i, image in enumerate(test):
    print("Writing test sample {}".format(i), end='\r')
    test_data.append(image[1].reshape(1, 1, h, w))
    test_ids.append(image[0])
print("", flush=True)

image_data_table.close()

test_id_df = pd.DataFrame({'id': test_ids})
test_id_df.to_csv(test_id_file, index=False)
#
# # print(labels.get_value('0015ceb851d7251b8f399e39779d1e7d', 'cancer'))