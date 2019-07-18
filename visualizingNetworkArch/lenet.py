from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        #- First set of Layers
        #- CONV => RELU => POOL

        #_ 20 Filters ?| (5 x 5) 
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu')) #? ReLu instead of Tanh, its way better now !
        #_ (2 X 2) filter size | (2 x 2) strides
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #- Another set of Layers
        #- CONV => RELU => POOL

        #_ 50 Filters
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        #- First and only set of
        #- FC => RELU 
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        #- Softmax Classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model