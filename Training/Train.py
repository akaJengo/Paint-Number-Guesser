import tensorflow as tf
import pandas as pd
import numpy as np 

#Model paramters must be the same prior to loading trained weights 
model = tf.keras.Sequential()

model.add(tf.keras.Input(784,))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dense(10, 'softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', 
            metrics=['accuracy'])

#Function to load the weights, You may need to change the pathing to your saved models
model.load_weights('./trained_32_10_adam_cc.h5')

#Loading the test data
data_test = pd.read_csv('~\Desktop\Lessons\Machine Learning\mnist_test.csv')

y_test = data_test['label']
y_test = tf.keras.utils.to_categorical(y_test)

x_test = data_test.iloc[:,1:] / 255

#Using our trained model to predict the first number 
#The prediction array will be the actual output of the output neuron
prediction = model.predict(x_test[:1]) #Use the first number to predict 
#prediction = model.predict(x_test[:5]) #Use the first 5 to predict

#Argmax gives us the index with the highest value (Finds the greatest value from the output neuron and prints the index)
print("Predicted number is: ",np.argmax(prediction)) 
print("Actual number is: ",np.argmax(y_test[:1]))
