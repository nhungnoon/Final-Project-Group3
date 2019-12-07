from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
from keras.utils.np_utils import to_categorical
from tensorflow.compat.v2.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean, SparseCategoricalAccuracy
import os
tf.random.set_seed(42)
np.random.seed(42)

# DATA PREP
# load data and split labels from the csv file
os.system("wget https://dataml2.s3.amazonaws.com/sign_mnist_train.csv")
data = pd.read_csv('sign_mnist_train.csv')
labels = data['label'].values
data.drop('label', axis=1, inplace=True)
images = data.values/255 # normalize the data

# split data into train and validation tests
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)

# reshape data and convert labels to tensor
x_train, x_val = tf.reshape(x_train, (-1, 28, 28, 1), ), tf.reshape(x_val, (-1, 28, 28, 1))
x_train, x_val = tf.dtypes.cast(x_train, tf.float32), tf.dtypes.cast(x_val, tf.float32)
y_train, y_val = tf.convert_to_tensor(y_train), tf.convert_to_tensor(y_val)

train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

# DEFINE MODEL - 2 layers with con2d, batchnorm, and max pool, activation functions: relu and softmax
EPOCHS = 10
DROPOUT = 0.5

class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.convnorm1 = BatchNormalization()
    self.pool1 = MaxPool2D(pool_size = (2, 2), strides = (2,2))

    self.conv2 = Conv2D(64, (2, 2), padding= 'same', strides=(1, 1), activation='relu')
    self.convnorm2 = BatchNormalization()
    self.pool2 = MaxPool2D(pool_size = (2, 2), strides = (2,2))

    self.flatten = Flatten()
    self.drop = DROPOUT
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(25, activation='softmax')

  def call(self, x):
    x = self.pool1(self.convnorm1(self.conv1(x)))
    x= self.pool2(self.convnorm2(self.conv2(x)))
    x = self.flatten(x)
    x = tf.nn.dropout(self.d1(x), self.drop)
    return self.d2(x)

# Create an instance of the model
model = CNN()
loss_ = SparseCategoricalCrossentropy()
optimizer = Adam()
train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')
val_loss = Mean(name='val_loss')
val_accuracy = SparseCategoricalAccuracy(name='val_accuracy')

# define functions for training and testing
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

  val_loss(t_loss)
  val_accuracy(labels, predicts)

# TRAINING
epochs, train_l, val_l, train_a, val_a = [], [], [], [], []
for epoch in range(EPOCHS):
  for train_images, train_labels in train:
    training(train_images, train_labels)

  for val_images, val_labels in val:
    testing(val_images, val_labels)

  to_print = 'Epoch {}, Loss: {}, Accuracy: {}, Valid Loss: {}, Valid Accuracy: {}'
  print(to_print.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        val_loss.result(),
                        val_accuracy.result()*100))
  train_l.append(train_loss.result())
  train_a.append(train_accuracy.result())
  val_l.append(val_loss.result())
  val_a.append(val_accuracy.result())
  epochs.append(epoch)

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  val_loss.reset_states()
  val_accuracy.reset_states()

  model.save_weights('model', save_format='tf')

plt.figure(figsize=(24,8))
plt.subplot(1,2,1)
plt.plot(epochs,val_a, label="validation_accuracy", c="red")
plt.plot(epochs, train_a, label="training_accuracy", c="green")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, val_l, label="validation_loss", c="red")
plt.plot(epochs, train_l, label="training_loss", c="green")
plt.grid(True)
plt.legend()

plt.suptitle("ACCURACY / LOSS",fontsize=18)

plt.show()