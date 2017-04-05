import argparse
import logging
import os
import datetime
import numpy as np
import pandas as pd
from ia_kdsb17.common import is_writeable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Simple 2D CNN for Kaggle Data Science Bowl 2017')
parser.add_argument("train_data", help="The numpy binary file of training data")
parser.add_argument("labels", help="The numpy binary file pf labels")
parser.add_argument("test_data", help="The numpy binary file of test data")
parser.add_argument("test_ids", help="CSV of IDs for the test set")
parser.add_argument("results_dir", help="Directory to where results CSV will be written, must be writeable")

args = parser.parse_args()

if not is_writeable(args.results_dir):
    print("{} is not writeable".format(args.results_dir))
    exit(1)

results_dir = os.path.join(args.results_dir, "results_{}".format(datetime.datetime.utcnow()).replace(' ', '_'))

train_data = np.load(args.train_data, mmap_mode='r')
test_data = np.load(args.test_data, mmap_mode='r')
labels = np.load(args.labels, mmap_mode='r').astype('int')
test_ids = pd.read_csv(args.test_ids)

l, rows, cols = train_data.shape

train_data = train_data[..., np.newaxis]
train_data = train_data.reshape((l, rows, cols, 1))


x_train, x_val, y_train, y_val = train_test_split(train_data, labels, test_size=0.15, random_state=0)

input_shape = (rows, cols, 1)

model = config.model_function(input_shape)
model.fit(x_train, y_train, **config.model_params)

val_predictions = model.predict(x_val).astype('int')

l, rows, cols = test_data.shape
test_data = test_data[..., np.newaxis]
test_data = test_data.reshape((l, rows, cols, 1))

test_predictions = model.predict(test_data).astype('int')

accuracy = accuracy_score(y_val, val_predictions)

if not os.path.exists(results_dir):
    print('Creating results dir')
    os.mkdir(results_dir, mode=0o755)
else:
    print('Results path already existed')


def to_indexes(values):
    return list(map(lambda x: x.index(1), values.tolist()))

val_results_df = pd.DataFrame({'actual': to_indexes(y_val), 'predicted': to_indexes(val_predictions)})
val_results_df.to_csv(os.path.join(results_dir, "validation_results.csv"))

test_ids['prediction'] = to_indexes(test_predictions)
test_ids.to_csv(os.path.join(results_dir, "test_predictions.csv"))

model.save(os.path.join(results_dir, "model.h5"))

print(val_results_df.head())
print("Validation Accuracy: {}".format(accuracy))
