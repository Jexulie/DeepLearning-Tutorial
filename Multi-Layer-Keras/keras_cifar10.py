from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())


#! LOAD DATA
print("[INFO] loading CIFAR-10 data ...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
#* reshape data
#* CIFAR-10 from 8-bit int to floating point
#* data -> 32 x 32 x 3
#* trainX shape -> (50000, 32, 32, 3) | testX shape -> (10000, 32, 32, 3)
#* flatten -> 32 x 32 x 3 = 3,072 entries.
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))
#* data shape is now;
#* trainX -> (50000, 3072) | testX -> (10000, 3072)

#! LABELIZE
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

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

#! STOCHASTIC GRADIENT DESCENT MODEL - LAYERS
model = Sequential()
#* Layers
#! NETWORK ARCHITECTURE [3072 - 1024 - 512 - 10]
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network ...")
sgd = SGD(0.01)

#! LOSS CATEGORICAL CROSSENTROPY AND FIT MODEL / TRAIN
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

#! PREDICTION
print("[INFO] evaluating network...")
pred = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1), target_names=labelNames))

#! PLOT LOSS GRAPH
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
