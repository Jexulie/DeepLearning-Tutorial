from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19, imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-m", "--model", type=str, default="vgg16")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
}


if args["model"] not in MODELS.keys():
    raise AssertionError("The --model cli arg should be a key in MODELS dict..")


#- init input shape of pre-trained models, which is (224x224)px
inputShape = (224, 224)

if args["model"] in ("inception", "xception"):
    inputShape = (299, 299)
    preprocess = preprocess_input


#- load network weights from disk
print("[INFO] loading {}...".format(args["model"]))
Network = MODELS[args["model"]]
model = Network(weights="imagenet")

#- Inception or Xception it is ...


#- load image -> resizing | output -> np.array
print("[INFO] loading and pre-processing image...")
image = load_img(args["image"], target_size=inputShape)
image = img_to_array(image)


#- pre-process the image using the appropriate function based on the model that has been loaded (i.e. mean subtraction, scaling, etc...)
image = np.expand_dims(image, axis=0)
image = preprocess(image)


#- classify img
print("[INFO] classifying image with '{}'...".format(args["model"]))
preds = model.predict(image)
P = imagenet_utils.decode_predictions(preds)

#- loop over the preds and display the rank-5 preds + probbilities
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

#- load image via OpenCV
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)