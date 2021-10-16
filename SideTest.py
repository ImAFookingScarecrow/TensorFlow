import tensorflow as tf
import pandas as pd
import numpy as np

length = 20 # the length of each binary string
test_size = 30

filee = open("randomtext.txt", "r")
filef = open("nonrandomtext.txt", "r")
randomData = filee.read().split('\n')
nonrandomData = filef.read().split('\n')
filee.close()
filef.close()

trainRandomData = randomData[:-(test_size+1)]
testRandomData = randomData[-(test_size+1):-1]
randomExamples = len(randomData)
trainNonrandomData = nonrandomData[:-(test_size+1)]
testNonrandomData = nonrandomData[-(test_size+1):-1]
nonrandomExamples = len(nonrandomData)

feature_columns = []

def split(array):
    list = {}
    for i in range(0, length):
        for line in array:
            list[str(i)] = pd.Series([line[i] for line in array])
    return list


data = split(trainRandomData + trainNonrandomData)
testData = split(testRandomData + testNonrandomData)

for i in range(0, length):  # returns a list of all unique feature names from the column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(str(i), ['0', '1']))


def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((data_df, label_df)) # create a tf.data.Dataset object with data and it's labels
        if shuffle:
            ds = ds.shuffle(1000) # shuffle the data
        ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process for number of epochs
        return ds # return a batch of the dataset
    return input_function # return a function object for use

label = pd.Series(['1' for i in trainRandomData] + ['0' for i in trainNonrandomData])
testLabel = pd.Series(['1' for i in range(0, test_size)] + ['0' for i in range(0, test_size)])

print(type(label[1]))

train_input_fn = make_input_fn(data, label)
eval_input_fn = make_input_fn(testData, testLabel, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

result = list(linear_est.predict(eval_input_fn))



