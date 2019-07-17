import matplotlib
matplotlib.use("Agg")

from miniVGGNet import MiniVGGNet

from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras import backend as K
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt
import numpy as np


def step_decay(epoch):
    #- Init Base Learning Rate, Drop Factor and Epochs to drop every
    initAlpha = 0.01
    factor = 0.5
    dropEvery = 5

    #- out formula
    return initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))


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

#- define set of callbacks to be passed to the model during training
callbacks = [LearningRateScheduler(step_decay)]


#- Init and compile model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#- train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1, callbacks=callbacks) #- add callbacks to training

#- evaluate the network
print("[INFO] evaluating network...")
preds = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), 
    target_names=labelNames))

#- plot results
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history['loss'], label="train_loss")
plt.plot(np.arange(0, 40), H.history['val_loss'], label="val_loss")
plt.plot(np.arange(0, 40), H.history['acc'], label="train_acc")
plt.plot(np.arange(0, 40), H.history['val_acc'], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('Cifar-10-lr-decay-Acc-Loss.png')
plt.show()