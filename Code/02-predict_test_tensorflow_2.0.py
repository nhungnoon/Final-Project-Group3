from __future__ import absolute_import, division, print_function, unicode_literals
import functools
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
import os
tf.random.set_seed(42)
np.random.seed(42)

# data prep
os.system("wget https://dataml2.s3.amazonaws.com/sign_mnist_test.csv")
test = pd.read_csv('sign_mnist_test.csv')

labels = test['label'].values
test.drop('label', axis=1, inplace=True)
images = (test.values)/255
X = tf.reshape(images, (-1, 28, 28, 1), )
X = tf.dtypes.cast(X, tf.float32)
Y = tf.convert_to_tensor(labels)
test_ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(32)

# define model
class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(25, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# # metrics and such
model = CNN()
loss_ = SparseCategoricalCrossentropy()
optimizer = Adam()
test_loss = Mean(name='test_loss')
test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')

# load trained model
model.load_weights('model')

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

for test_images, test_labels in test_ds:
  test_step(test_images, test_labels)

to_print = 'Test Loss: {}, Test Accuracy: {}'
print(to_print.format(test_loss.result(),
                      test_accuracy.result()*100))