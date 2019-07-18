from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.optimizers import SGD
from lenet import LeNet
from utils.capcha_helper import preprocess
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse, cv2, os

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True)
ap.add_argument('-m', '--model', required=True)
args = vars(ap.parse_args())

data = []
labels = []

for imagePath in paths.list_images(args["dataset"]):
    img = cv2.imread(imagePath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = preprocess(image, 28, 28)
    img = img_to_array(img)
    data.append(img)
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model..")
model = LeNet.build(width=28, height=28, depth=1, classes=9)
opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training network..")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=15, verbose=1)

print("[INFO] evaluating network..")
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=lb.classes_))

print("[INFO] serializing model..")
model.save(args["model"])

plt.style.use("ggplot")
plt.figure()
plt.plot(np.array(0, 15), H.history['loss'], label='train_loss')
plt.plot(np.array(0, 15), H.history['val_loss'], label='val_loss')
plt.plot(np.array(0, 15), H.history['acc'], label='acc')
plt.plot(np.array(0, 15), H.history['val_acc'], label='val_acc')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()