To run Keras:
The file keras_train_predict_combine.py can be downloaded and run in one setting. The file includes:
+ Retrieve train and test data
+ Train model using the Sequential API Keras
+ Evaluate model on test set
+ Visualization of loss and accuracy over epochs

To run Tensorflow:
1. The file 01-train_tensorflow_2.0.py should be run first to create model. The file includes:
+ Retrieve train data
+ Train model using tf.keras.Model
+ Save model
+ Visualization of loss and accuracy over epochs

2. The file 02-predict_test_tensorflow_2.0.py will be run after to predict the accuracy of test set based on the saved model. The file includes:
+ Retrieve test data
+ Define the model structure
+ Load trained parameters 
+ Test on the test data 
