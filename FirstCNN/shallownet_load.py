from ImageToArrayPreprocessor import ImageToArrayPreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
from SimplePreprocessor import SimplePreprocessor
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

TARGET_NAMES=["Bulbasaur", "Charmander", "Squirtle"]

print("[INFO] sampling images...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]

#* init the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()


#* load the dataset from disk then scale the raw pixel intensities
#* to the range [0, 1] AKA Normalization !!
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

#* load the pre-trained network
print("[INFO] loading pre-trained network")
model = load_model(args["model"])

print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(TARGET_NAMES[preds[i]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)