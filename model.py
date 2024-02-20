import tensorflow as tf
import tensorflow.keras.layers as layers
from keras import Sequential
from keras.layers import *

# model = Sequential()
# model.add(Conv2D(kernel_size=3, filters=32, padding='same', input_shape=(32,32,3)))
# model.add(Activation('relu'))
# model.add(Conv2D(kernel_size=3, filters=32, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding = 'same'))

# model.add(Conv2D(kernel_size=3, filters=64, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(kernel_size=3, filters=64, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding = 'same'))

# model.add(Conv2D(kernel_size=3, filters=128, padding='same'))
# model.add(Activation('relu'))
# model.add(Conv2D(kernel_size=3, filters=128, padding='same'))
# model.add(Activation('relu'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding = 'same'))

# model.add(Flatten())


# model.add(Dense(256, Activation('relu'))) 
# model.add(Dense(10, Activation('softmax')))

# model.compile(metrics = ["acc"],
#               loss = "sparse_categorical_crossentropy",
#               optimizer = "adam")

    
if __name__=="__main__":
    model = u_net()
    model.build(input_shape=(None, 256, 256, 3))
    print(model.summary())
    print("model is ready to use.")