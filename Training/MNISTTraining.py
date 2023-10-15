import tensorflow as tf
import pandas as pd 
import numpy as np 

data_train = pd.read_csv('~\Desktop\Lessons\Machine Learning\mnist_train.csv')
data_test = pd.read_csv('~\Desktop\Lessons\Machine Learning\mnist_test.csv')

y_train = data_train["label"]
y_test = data_test['label']

x_train = data_train.iloc[:,1:]
x_train = x_train / 255

x_test = data_test.iloc[:,1:] / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = tf.keras.Sequential()

model.add(tf.keras.Input(784,))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, 'softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(x_train, y_train, 32, 30)

model.save_weights('trained_32_64_64_10_adam.h5')
#Epoch 30/30 1875/1875 [==============================] - 2s 851us/step - loss: 0.0187 - accuracy: 0.9936