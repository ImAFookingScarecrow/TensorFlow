import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Data import dataPointsX, dataPointsY

dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() # returns a list of all unique feature names from the column
    print(vocabulary)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32)) # adds numerical data as a column of floats


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create a tf.data.Dataset object with data and it's labels
        if shuffle:
            ds = ds.shuffle(1000) # shuffle the data
        ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process for number of epochs
        return ds # return a batch of the dataset
    return input_function # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train) # call the input_function returned to us tp get a dataset
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

print(type(y_train))

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, model_dir="modelDir")

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

result = list(linear_est.predict(eval_input_fn))
