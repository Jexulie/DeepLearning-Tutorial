import matplotlib
matplotlib.use("Agg")

from miniVGGNet import MiniVGGNet
from trainingMonitor import TrainingMonitor

from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import os, argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

print("[INFO] loading CIFAR-10 Dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

#_ Color Normalization
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

#_ Binarize Y 
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#_ init classes
labelNames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

#- Init and compile model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

#- define set of callbacks to be passed to the model during training
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

#- train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1, callbacks=callbacks) #- add callbacks to training
