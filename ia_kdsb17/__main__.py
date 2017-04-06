import argparse
import logging
import os
import datetime
import numpy as np
import pandas as pd
import tables
from ia_kdsb17.common import is_writeable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Simple 2D CNN for Kaggle Data Science Bowl 2017')
parser.add_argument("image_data", help="The hdf5 file of image data")
parser.add_argument("test_ids", help="CSV of IDs for the test set")
parser.add_argument("results_dir", help="Directory to where results CSV will be written, must be writeable")

args = parser.parse_args()

if not is_writeable(args.results_dir):
    print("{} is not writeable".format(args.results_dir))
    exit(1)

results_dir = os.path.join(args.results_dir, "results_{}".format(datetime.datetime.utcnow()).replace(' ', '_'))

# train_data = np.load(args.train_data, mmap_mode='r')
# test_data = np.load(args.test_data, mmap_mode='r')
# labels = np.load(args.labels, mmap_mode='r').astype('int')
test_ids = pd.read_csv(args.test_ids)

image_data_table = tables.open_file(args.image_data, mode='r')
print(image_data_table)
train_data = image_data_table.root.train_data[:, :, :, :]
test_data = image_data_table.root.test_data[:, :, :, :]
labels = image_data_table.root.train_labels[:, :]

l, c, rows, cols = train_data.shape

# train_data = train_data[..., np.newaxis]
# train_data = train_data.reshape((l, rows, cols, 1))


x_train, x_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.15, random_state=0)

input_shape = (c, rows, cols)

model = config.model_function(input_shape)
model.fit(x_train, y_train, **config.model_params)

val_predictions = model.predict(x_val).astype('int')

test_predictions = model.predict(test_data).astype('int')

accuracy = accuracy_score(y_val, val_predictions)

if not os.path.exists(results_dir):
    print('Creating results dir')
    os.mkdir(results_dir, mode=0o755)
else:
    print('Results path already existed')


def to_probability_of_cancer(values):
    return list(map(lambda x: x[1], values.tolist()))

y_val_prob = to_probability_of_cancer(y_val)
pred_val_prob = to_probability_of_cancer(val_predictions)

val_results_df = pd.DataFrame({'actual': y_val_prob,
                               'predicted': pred_val_prob})

val_results_df.to_csv(os.path.join(results_dir, "validation_results.csv"), index=False)

test_ids['prediction'] = to_probability_of_cancer(test_predictions)
test_ids.to_csv(os.path.join(results_dir, "test_predictions.csv"))

model.save(os.path.join(results_dir, "model.h5"))

print(val_results_df.head())
print("Validation score: {}".format(config.validation_function(y_val_prob, pred_val_prob)))
print("Results written to {}".format(results_dir))
