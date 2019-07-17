from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2

def convolve(image, K):
    #* Shape of image and kernel
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    #* Output placeholder
    pad = (kW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float")

    #* X - Y sliding loop
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # print(f'y:{y} | x:{x}')
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            k = (roi * K).sum()
            output[y - pad, x - pad]
        
    #* to keep color values between [0-255]
    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype("uint8")
    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

#! Filters

smallBlurRatio = 7
largeBlurRatio = 21

smallBlur = np.ones((smallBlurRatio, smallBlurRatio), dtype="float") * (1.0 / (smallBlurRatio * smallBlurRatio))

largeBlur = np.ones((largeBlurRatio, largeBlurRatio), dtype="float") * (1.0 / (largeBlurRatio * largeBlurRatio))


sharpen = np.array((
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
), dtype="int")

#!########################################
#! Kernels

#! Laplacian kernel used to detect edge-like regions

laplacian = np.array((
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
), dtype="int")

#! Sobels used to detect edge-like regions on x | y axis

sobelX = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
), dtype="int")

SobelY = np.array((
    [-1, -2, -2],
    [0, 0, 0],
    [1, 2, 1]
), dtype="int")

#!########################################

emboss = np.array((
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
), dtype="int")


#!#######################################

kernelBank = (
    ("small_blur", smallBlur),
    ("large_blur", largeBlur),
    ("sharpen", sharpen),
    ("laplacian", laplacian),
    ("sobel_x", sobelX),
    ("sobel_y", SobelY),
    ("emboss", emboss),
)

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

    cv2.imwrite(f"./filtered-images/{kernelName} - Original - gray.png", gray)
    cv2.imwrite("./filtered-images/{} - convole.png".format(kernelName), convolveOutput)
    cv2.imwrite("./filtered-images/{} - opencv.png".format(kernelName), opencvOutput)
    cv2.waitKey(0)
    cv2.destroyAllWindows()