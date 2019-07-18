from imutils import paths
import argparse, imutils, cv2, os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
ap.add_argument('-a', '--annot', required=True)
args = vars(ap.parse_args())

imPaths = list(paths.list_images(args["input"]))
counts = {}

for (i, imPaths) in enumerate(imPaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imPaths)))

    try:
        #- padding the image to prevent to miss numbers which are touching the border.
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8,8,8,8, cv2.BORDER_REPLICATE)

        #- convert bg to black, numbers to white color
        #- common way to work on ML|DL|AI
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

        #- find contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            if key == ord("'"):
                print("[INFO] ignoring character")
                continue
            
            key = chr(key).upper()
            dirPath = os.path.sep.join([args['annot'], key])

            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])
            cv2.imwrite(p, roi)

            counts[key] = count + 1

    except KeyboardInterrupt:
        print("[INFO] Manually leaving script")
        break
    
    except:
        print("[INFO] skipping image...")