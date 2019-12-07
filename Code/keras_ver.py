# data source
# https://www.kaggle.com/datamunge/sign-language-mnist
# code reference
# https://www.kaggle.com/ranjeetjain3/deep-learning-using-sign-langugage

# import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.initializers import glorot_uniform
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Retrieve the train and test data
os.system("wget https://dataml2.s3.amazonaws.com/sign_mnist_train.csv")
os.system("wget https://dataml2.s3.amazonaws.com/sign_mnist_test.csv")

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# set seed and the initial weight 
SEED = 42
np.random.seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# %% --------------------------------------- Data --------------------------------------------------------------------
# load the data 
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')


# create train images from the values
train_imgs = train.iloc[:,1:].values.reshape((train.iloc[:,1:].shape[0],28,28))
train_imgs = np.expand_dims(train_imgs, axis=-1)
# get the labels from the first column
train_labels = train.iloc[:, :1].values


# create test images from the values
test_imgs = test.iloc[:,1:].values.reshape((test.iloc[:,1:].shape[0],28,28))
test_imgs = np.expand_dims(test_imgs, axis=-1)
# get the labels from the first column
test_labels = test.iloc[:, :1].values

# categorize the labels
train_labels, test_labels = to_categorical(train_labels, num_classes=25), to_categorical(test_labels, num_classes=25)

# split train set into train and validation set
x_train, x_test2, y_train, y_test2 = train_test_split(train_imgs, train_labels, test_size=0.30)

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------

# define number of classes, learning rate, image shape, and number of epochs
n_classes = 25
learning_rate = 0.001
image_shape = (28, 28, 1)
n_epochs = 20

# %% ----------------------------------- Model Set-up --------------------------------------------------------------

# initilize a model
model = Sequential()

# add the first layer
model.add(Conv2D(32, (2, 2), padding= 'same', strides =  (1, 1), activation='relu', input_shape=image_shape,
kernel_initializer=weight_init))

# add the second layer
model.add(Conv2D(32, (2, 2), padding= 'same', strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# add the 3rd layer
model.add(Conv2D(64, (2, 2), padding= 'same', strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# add the 4th layer
model.add(Conv2D(96, (2, 2), padding= 'same', strides=(1, 1), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# flatten layer to ensure size are not mismatch
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

# last layer
model.add(Dense(n_classes, activation='softmax'))

# %% ----------------------------------- Train Model --------------------------------------------------------------


# create checkpoint and choose to save the best model
model_c = ModelCheckpoint('model_group3.hdf5', monitor='val_loss', mode='auto', verbose=1, save_best_only=True)

# construct the image generator for data augmentation
img_aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, horizontal_flip=True, shear_range=0.2,
                             vertical_flip=True)

# compile model
model.compile(optimizer=Adam(learning_rate), loss=categorical_crossentropy, metrics=[categorical_accuracy])


# using fit_generator for model
fit_model = model.fit_generator(img_aug.flow(x_train, y_train, batch_size=200),
	validation_data=(x_test2, y_test2), steps_per_epoch=len(x_train) // 200, epochs=n_epochs,
	 verbose=1, shuffle=True, callbacks=[model_c])

# %% ----------------------------------- The Accuracy for Test set --------------------------------------------------------------
print("Final accuracy on test set:", 100*model.evaluate(test_imgs, test_labels)[1], "%")


# %% ----------------------------------- Visualize the loss for the model --------------------------------------------------------------
# summarize history for loss
plt.plot(fit_model.history['categorical_accuracy'])
plt.plot(fit_model.history['val_categorical_accuracy'])
plt.title('Keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(fit_model.history['loss'])
plt.plot(fit_model.history['val_loss'])
plt.title('Keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# End
