from keras.preprocessing.image import img_to_array
from SimplePreprocessor import SimplePreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader

class ImageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)

# sp = SimplePreprocessor(32, 32)
# iap = ImageToArrayPreprocessor()
# sdl = SimpleDatasetLoader(preprocessors=[sp, iap])

# (data, labels) = sdl.load(imagePaths, verbose=500)

#* Load Image from Disk -> Resize to 32 x 32 -> Channels Ordered -> Output Image

