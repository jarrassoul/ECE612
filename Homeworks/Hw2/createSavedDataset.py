#createSavedDataset.py
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime

# Load the dataset
data_dict = pd.read_pickle(r"C:\Users\ya3be\Downloads\appml-assignment1-dataset-v2.pkl")
X = data_dict['X']
y = data_dict['y']
print(X.columns.tolist())
# Create bins for CAD change classification
bins = np.linspace(-0.001, 0.001, 21)
y_binned = np.digitize(y, bins)

# Extract time-based features
X['date'] = pd.to_datetime(X['date'])
weekday = X['date'].dt.dayofweek
hour = X['date'].dt.hour
month = X['date'].dt.month

# Get ticker values (excluding date column)
tickers = X.drop('date', axis=1).values

# Create TFRecord writer
writer = tf.io.TFRecordWriter('dataset.tfrecords')

# Create Example objects
for i in range(len(X)):
    feature = {
        'tickers': tf.train.Feature(float_list=tf.train.FloatList(value=tickers[i].astype(float))),
        'weekday': tf.train.Feature(int64_list=tf.train.Int64List(value=[weekday[i]])),
        'hour': tf.train.Feature(int64_list=tf.train.Int64List(value=[hour[i]])),
        'month': tf.train.Feature(int64_list=tf.train.Int64List(value=[month[i]])),
        'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[y_binned[i]]))
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())

writer.close()
print("dataset created succesfully", data_dict)
