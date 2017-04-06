import argparse
import os
import pandas as pd
import numpy as np
import config
import tables
from itertools import chain
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
stream = config.image_prep_function(args.data_dir, labels, **image_params)

one_patient = next(stream)

c, h, w = one_patient[2].shape
print("Image channels: {} rows: {} cols: {}".format(c, h, w))

stream = chain([one_patient], stream)

tables_file = os.path.join(args.output_dir, 'image_data.h5')

image_data_table = tables.open_file(tables_file, mode='w')
image_atom = tables.Float32Atom()
int_atom = tables.Int8Atom()
id_atom = tables.StringAtom(len(one_patient[1]))

train_data = image_data_table.create_earray(image_data_table.root, 'train_data', atom=image_atom, shape=(0, 1, h, w))
train_labels = image_data_table.create_earray(image_data_table.root, 'train_labels', atom=int_atom, shape=(0, 2))
test_data = image_data_table.create_earray(image_data_table.root, 'test_data', atom=image_atom, shape=(0, 1, h, w))
test_ids = image_data_table.create_earray(image_data_table.root, 'test_ids', atom=id_atom, shape=(0,))

test_sample = 1
train_sample = 1

for item in stream:
    dataset, idd, image = item

    if dataset == 'train':
        print("Writing training sample {}".format(train_sample), end='\r')
        train_sample += 1

        train_data.append(image.reshape(1, c, h, w))
        label = labels.get_value(idd, 'cancer')

        if label == 0:
            train_labels.append(np.array([1, 0]).reshape(1, 2))
        else:
            train_labels.append(np.array([0, 1]).reshape(1, 2))
    if dataset == 'test':
        print("Writing test sample {}".format(test_sample), end='\r')

        test_data.append(image.reshape(1, c, h, w))
        test_ids.append(idd)

print("", flush=True)


image_data_table.close()

#
# # print(labels.get_value('0015ceb851d7251b8f399e39779d1e7d', 'cancer'))