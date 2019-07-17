import matplotlib
matplotlib.use("Agg")

from miniVGGNet import MiniVGGNet

from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np


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
#_ learning-rate = 0.01
#_ momentum-term = 0.9
#_ Nestrov accelerated gradient to SGD optimizer set to True

#_ decay -> slowly reduce the learning rate over time | helps to reducing overfitting and obtaining higher classification accuracy, the smaller the learning rate is the smaller the weight updates will be.

#_ common decay calc -> divide the initial learning rate by total number of epochs
opt = SGD(lr=0.01, decay=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

#- train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)

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
plt.savefig('miniVGGNet-Cifar-10-Acc-Loss.png')
plt.show()

#_ without batch normalization,
#_ loss starts to increase path epoch 30, indicates that the network is overfitting the training data.

#_ 1) Batch Normalization can lead to a faster, more stable convergence with higher accuracy.
#_ 2) However, the advantages will come at the expense of training time - batch normalization will require more "wall time" to train the network, even though the network will obtain higher accuracy in less epochs.


#_ #_ Results
#_ INFO] evaluating network...
#_               precision    recall  f1-score   support
#_ 
#_     airplane       0.78      0.68      0.73      1000
#_   automobile       0.85      0.75      0.80      1000
#_         bird       0.61      0.43      0.51      1000
#_          cat       0.50      0.48      0.49      1000
#_         deer       0.54      0.64      0.59      1000
#_          dog       0.61      0.55      0.58      1000
#_         frog       0.57      0.89      0.70      1000
#_        horse       0.81      0.66      0.73      1000
#_         ship       0.76      0.83      0.80      1000
#_        truck       0.75      0.78      0.77      1000
#_ 
#_     accuracy                           0.67     10000
#_    macro avg       0.68      0.67      0.67     10000
#_ weighted avg       0.68      0.67      0.67     10000
#_ 