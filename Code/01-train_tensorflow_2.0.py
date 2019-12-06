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

# DATA PREP
os.system("wget https://dataml2.s3.amazonaws.com/sign_mnist_train.csv")
data = pd.read_csv('sign_mnist_train.csv')
print(data.shape)
labels = data['label'].values
data.drop('label', axis=1, inplace=True)
images = (data.values)/255
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)

x_train, x_test = tf.reshape(x_train, (-1, 28, 28, 1), ), tf.reshape(x_test, (-1, 28, 28, 1))
x_train, x_test = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_test, tf.float32)
y_train, y_test = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_test)

train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# DEFINE MODEL
EPOCHS = 20
DROPOUT = 0.25

class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.drop = DROPOUT
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(25, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = tf.nn.dropout(self.d1(x), self.drop)
    return self.d2(x)

# Create an instance of the model
model = CNN()
loss_ = SparseCategoricalCrossentropy()
optimizer = Adam()
train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
test_loss = Mean(name='test_loss')
test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def training(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def testing(images, labels):
  predicts= model(images)
  t_loss = loss_(labels, predicts)

  test_loss(t_loss)
  test_accuracy(labels, predicts)

# TRAINING
for epoch in range(EPOCHS):
  for train_images, train_labels in train:
    training(train_images, train_labels)

  for test_images, test_labels in test:
    testing(test_images, test_labels)

  to_print = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(to_print.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  model.save_weights('model', save_format='tf')