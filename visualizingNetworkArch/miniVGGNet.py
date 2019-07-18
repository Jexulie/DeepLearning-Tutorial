from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D,Conv2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        #- Index of the Channel Dimension
        chanDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        
        #- First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #_ probability of p = 0.25
        #_ which means a node from POOL layer will randomly disconnect from the next layer with a probability of 25% during training, helps to reduce the effects of overfitting.

        #- Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        #- first and only set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5)) #_ probability to 50%

        #- softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
