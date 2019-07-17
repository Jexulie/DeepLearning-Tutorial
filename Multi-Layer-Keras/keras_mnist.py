from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

#! LOAD DATA
print("[INFO] loading MNIST dataset ...")
dataset = datasets.fetch_openml('mnist_784', version=1, cache=True)

#! NORMALIZE
data = dataset.data.astype("float") / 255.0

#! SPLIT TEST-TRAIN DATA
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size=0.25)

#! LABELIZE
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

#! STOCHASTIC GRADIENT DESCENT MODEL - LAYERS
model = Sequential()
#* Layers
#! NETWORK ARCHITECTURE [784 - 256 - 128 - 10]
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))

print("[INFO] training network ...")
sgd = SGD(0.01)

#! LOSS CATEGORICAL CROSSENTROPY AND FIT MODEL / TRAIN
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

#! PREDICTION
print("[INFO] evaluating network...")
pred = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))


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