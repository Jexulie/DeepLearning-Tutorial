from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Flatten, Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        #- Init Model along with the input shape to be "channels_last"
        model = Sequential()
        inputShape = (height, width, depth)

        # if [keras.json] -> channels_first option is being used...
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        #- Define CONV => RELU layer
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        #- Softmax classifier
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
