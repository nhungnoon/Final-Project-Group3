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
import PIL
from PIL import Image
tf.random.set_seed(42)
np.random.seed(42)

# loading data and image preprocessing
images = os.listdir('/home/ubuntu/project/Project/images/')
# print(images)
root='/home/ubuntu/project/Project/images/'
data=[]
for i in images:
    img = np.asarray(Image.open(os.path.join(root, i)).convert("L").resize((28,28)))
    img = np.expand_dims(img, axis=-1)
    data.append(img)
print(len(data))
X= np.asarray(data)
print(X.shape)

images = (X)/255
X = tf.dtypes.cast(X, tf.float32)

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
loss_object = SparseCategoricalCrossentropy()
optimizer = Adam()
test_loss = Mean(name='test_loss')
test_accuracy = SparseCategoricalAccuracy(name='test_accuracy')

# load model
model.load_weights('model')
predictions = model(X)
print(predictions)
