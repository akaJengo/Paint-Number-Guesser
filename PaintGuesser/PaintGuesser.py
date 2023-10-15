import tkinter as tk
import tensorflow as tf
import pandas as pd
import numpy as np 



'''
Author: Micheal, Wigaloo
Description:

'''


#Constants
WIDTH, HEIGHT = [280,280]
SIZE = 280
PAINT_SiZE = 1
updates = 0 
pixelArray = np.zeros((28,28), dtype='int')

#Framework
window = tk.Tk()
window.title("Paint Guesser")

#The Canvas
Canvas = tk.Canvas(window, width=WIDTH, height=HEIGHT, bg='white')
Canvas.pack() #It like initializes the frame or canvas or whatever


model = tf.keras.Sequential()
model.add(tf.keras.Input(784,))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, 'softmax'))

model.compile(optimizer='adam', loss = 'categorical_crossentropy', 
              metrics=['accuracy'])

model.load_weights('trained_32_64_64_10_adam.h5')

#Painting
def Paint(event):
    global updates
    x,y = event.x //PAINT_SiZE, event.y // PAINT_SiZE #Main part that allows what pixel can be drawn by using the modulo

    pixelArray[int(x/10),int(y/10)] = 254
    #pixelArray[y * WIDTH + x] = 0  # Set pixel to black in the array
    Canvas.create_rectangle( #This function allows us to create the paint and the bottom is jus filling out the properties  
        x * PAINT_SiZE,
        y * PAINT_SiZE,
        (x + 1) * PAINT_SiZE,
        (y + 1) * PAINT_SiZE,
        fill="black",
        outline="black"
    )

    df = pd.DataFrame(pixelArray)
    cleaned = df.to_numpy() / 255
    cleaned = cleaned.T
    cleaned = np.reshape(cleaned, (-1,784))

    if updates % 400 == 0:
        prediction = model.predict(cleaned, verbose=0)
        print(np.argmax(prediction))

    updates += 1

Canvas.pack()

#enter.pack()



Canvas.bind("<B1-Motion>", Paint)



''' This allows the funciton to work as bind will allow you to say 
which function you want to call when B1 motion or left button (m1) is clicked and moved at
the same time.

'''

window.mainloop()






