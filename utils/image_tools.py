import cv2
import matplotlib
import matplotlib.pyplot as plt
import time
from google.colab.patches import cv2_imshow

import numpy as np

def rescaleImage(image, colormap_min=0, colormap_max=4096):
    image_clipped = np.where(image > colormap_max, colormap_max, image)
    image_clipped = (image_clipped - colormap_min)/colormap_max*255
    image_clipped = image_clipped.astype(dtype='uint8')
    return image_clipped


def adjustGamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def plotBBoxes(image, detections):
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for detection in detections:
        # format detection & GT  [ xTopLeft, yTopLeft, xBottomRight, yBottomRight]
        if len(detection) == 4:
            cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (255, 255, 255),
                1
            )
        
        elif len(detection) > 4:
            if detection[4] == 0:
                cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 255, 255),
                2
            )
            elif detection[4] == 1:
                cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 0, 255),
                2
            )
            elif detection[4] == 2:
                cv2.rectangle(
                image,
                (int(detection[0]), int(detection[1])),
                (int(detection[2]), int(detection[3])),
                (0, 255, 0),
                2
            )  
            
    return image




def showImage(image):
    # Changed to work on google colab
    #cv2.imshow("Image", image)
    cv2_imshow(image)
    # cv2.waitKey(1)

def adjustImage(image):
    max = np.percentile(image, 95)
    image = rescaleImage(image, colormap_max=max)
    image = adjustGamma(image, 2)
    return image

def preProcess(image):
    max = np.percentile(image, 95)
    image = rescaleImage(image, colormap_max=max)
    # image = cv2.medianBlur(image, 10)
    return image


def computeDarkImage (images):
    # Compute Dark Image
    print('Computing Dark image...')
    t = time.time()
    aux = np.zeros([images.shape[0], images.shape[1], 30])
    nFrames = images.shape[2]
    for i in range(30):
        aux[:, :, i] = (images[:, :, nFrames - i - 1])
    darkImage1 = np.median(aux, axis=2)

    for i in range(30):
        aux[:, :, i] = (images[:, :, i])
    darkImage2 = np.median(aux, axis=2)

    darkImage = np.minimum(darkImage1, darkImage2)

    max = np.percentile(darkImage, 95)

    darkImage = np.where(darkImage > max, max, darkImage)

    elapsed = time.time() - t
    print('Elapsed time computing dark image: {}'.format(elapsed))

    return darkImage

def computeDarkImageOld(images):
    print('Computing Dark image...')
    t = time.time()
    images = np.zeros([images.shape[0], images.shape[1], 30])
    nFrames=images.shape[2]
    for i in range(30):
        images[:,:,i] = (images[:, :, nFrames -i-1])
    darkImage = np.median(images, axis=2)
    elapsed = time.time() - t
    print('Elapsed time computing dark image: {}'.format(elapsed))
    return darkImage
