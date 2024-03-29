from keras.preprocessing.image import img_to_array
from keras.models import load_model
from utils.capcha_helper import preprocess
from imutils import contours, paths
import numpy as np
import argparse, imutils, cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
ap.add_argument('-m', '--model', required=True)
args = vars(ap.parse_args())

print("[INFO] loading pre-trained network...")
model = load_model(args['model'])

imgPaths = list(paths.list_images(args['input']))
imgPaths = np.random.choice(imgPaths, size=(10,), replace=False)

for imgPaths in imgPaths:
    #- load image and convert it into grayscale
    #- pad the image
    image = cv2.imread(imgPaths)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_REPLICATE)

    #- threshold the image to reveal the digits
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU[1])

    #- find contours in the img, keep only the four largest ones, then sort them from left to right
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]
    cnts = contours.sort_contours(cnts)[0]

    #- init the output img as a grayscale img with 3 channels among with the output preds
    output = cv2.merge([gray] * 3)
    predictions = []

    #- loop over contours
    for c in cnts:
        #- compute the bounding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

        #- pre-process the ROI and classify it then classify it
        roi = predictions(roi, 28, 28)
        roi = np.expand_dims(img_to_array(roi), axis=0) / 255.0
        pred = model.predict(roi).argmax(axis=1)[0] + 1
        predictions.append(str(pred))

        #- draw the prediction on the output image
        cv2.rectangle(output, (x - 2, y - 2), (x + w+ 4, y + h + 4), (0, 255, 0), 1)

        cv2.putText(output, str(pred), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    print("[INFO] captcha: {}".format("".join(predictions)))
    cv2.imshow("Output", output)
    cv2.waitKey()
    