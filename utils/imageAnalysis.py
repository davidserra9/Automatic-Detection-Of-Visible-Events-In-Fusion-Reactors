
import image_tools
import numpy as np
import numpy.ma as ma
import cv2
from skimage import feature
import time

def preProcessAnalyze(image):
    return image


def process_image( image):
        """  Apply  following operations to backgrouns substracted image to improve moving object detection:
                - Thresholding to remove shadows in some models
                - Morphological opening with a 5x5 circular structuring element to reduce noise
                - Morphological closing with a 10x10 circular structuring element to fill objects
                - Apply roi.jog mask
        Returns:
            Processed image
        """

        ret, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return image

def get_features(image):
    """  Given a binarized image, find connected components as detections
        Applies a non maxima suppression algorithm to merge similar detections
    Returns:
        Detections
    """

    #PreProces Image
    max = np.percentile(image, 95)
    image = image_tools.rescaleImage(image, colormap_max=max)

    #apply TopHat
    size = 80
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    image = cv2.medianBlur(tophat, 5)

    # apply Otsu
    ret, binarizedImage = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #process image
    ret, image = cv2.threshold(binarizedImage, 130, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    detections = []

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    area = stats[0][4]
    width =  stats[0][2]
    height =  stats[0][3]
    formFactor = width/height
    fillFactor = area/(width*height)
    centroidX = centroids[0][0]
    centroidY = centroids[0][1]



    features = [area, width, height, formFactor, fillFactor, centroidX, centroidY]

    return features

def get_intensity_features(image, croppedImage, ring):
    """  Given a binarized image, find connected components as detections
        Applies a non maxima suppression algorithm to merge similar detections
    Returns:
        Detections
    """
    image = preProcessAnalyze(image)
    croppedImage = preProcessAnalyze(croppedImage)
    #7-> Mean Intensity ROI
    meanROI = np.mean(croppedImage)
    #8-> Mean Intensity Image
    meanImage = np.mean(image)
    #9-> Intensity Histogram
    intHist, aux = np.histogram(croppedImage, bins = 256, range = (0,4095))
    # normalize the histogram
    intHist = intHist.astype("float")
    intHist /= (intHist.sum() + 1e-7)
    #10-> Historgrama LBP
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    numPoints = 8
    radius = 2
    lbp = feature.local_binary_pattern(croppedImage, numPoints,
                                       radius, method="uniform")
    (LBPHist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))

    # normalize the histogram
    LBPHist = LBPHist.astype("float")
    LBPHist /= (LBPHist.sum() + 1e-7)
    #11-> Intensitat amb una corona al voltant del HotSpot
    meanRing = np.mean(ring)
    #12-> Histograma corona
    ringHist, aux = np.histogram(ring, bins = 256, range = (0,256))
    ringHist = ringHist.astype("float")
    ringHist /= (ringHist.sum() + 1e-7)

    features = [meanROI, meanImage]

    for value in intHist:
        features.append(value)

    for value in LBPHist:
        features.append(value)

    features.append(meanRing)

    for value in ringHist:
        features.append(value)

    return features


def get_ring_features(input_image, detection):

    aux_image = np.zeros(input_image.shape)

    aux_image[int(detection[1]): int(detection[3]), int(detection[0]): int(detection[2])] = 1

    # PreProces Image
    max = np.percentile(input_image, 95)
    input_image = image_tools.rescaleImage(input_image, colormap_max=max)
    masked_image = np.where(aux_image == 1, input_image, 0)


    #apply TopHat
    size = 80
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    tophat = cv2.morphologyEx(masked_image, cv2.MORPH_TOPHAT, kernel)
    tophat_image = cv2.medianBlur(tophat, 5)

    # apply Otsu
    ret, binarizedImage = cv2.threshold(tophat_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #process image
    ret, image = cv2.threshold(binarizedImage, 130, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    #obtain ring
    ringSize = int((detection[3]-detection[1]+detection[2]-detection[0])/2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ringSize, ringSize))
    aux = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    ringMask = aux - image
    ringMask = (255*np.ones(ringMask.shape) - ringMask)/255
    ring = ma.array(input_image, mask= ringMask)

    features = [ring.mean()]

    return features


def extractCharacteristics(image, detection):

    #Characteristics:
    #0-> Binarized AREA
    #1-> Width
    #2-> Height
    #3-> FORM FACTOR
    #4-> Fill Factor
    #5-> centroidX
    #6-> centroidY
    #7-> Mean Intensity ROI
    #8-> Mean Intensity Image
    #9-> Intensity Histogram
    #10-> Historgrama LBP
    #11-> Intensitat amb una corona al voltant del HotSpot
    #12-> Histograma corona

    characteristics = []
    croppedImage = image[int(detection[1]): int(detection[3]),
            int(detection[0]): int(detection[2])]
    if croppedImage.size == 0:
        print ('found')
    else:
        features = get_features(croppedImage)

        for feature in features:
            characteristics.append(float(feature))


        ring =[0,0,0,0,0]
        intensity_features = get_intensity_features (image, croppedImage, ring)

        for feature in intensity_features:
            characteristics.append(float(feature))

        ring_features = get_ring_features (image, detection)


        for feature in ring_features:
            characteristics.append(float(feature))

    return characteristics



def compareHistograms(im1,im2, method = "Chi-Squared"):

    OPENCV_METHODS = {
        "Correlation": cv2.HISTCMP_CORREL,
        "Chi-Squared": cv2.HISTCMP_CHISQR,
        "Intersection": cv2.HISTCMP_INTERSECT,
        "Hellinger": cv2.HISTCMP_BHATTACHARYYA
    }

    #preprocess images
    im1 = preProcessAnalyze(im1)
    im2 = preProcessAnalyze(im2)
    # Intensity Histogram
    H1, aux = np.histogram(im1, bins = 256, range = (0,4095))
    #Normalize the histogram
    H1 = H1.astype("float")
    H1 /= (H1.sum() + 1e-7)
    H1 = H1.astype("float32")
    # Intensity Histogram
    H2, aux = np.histogram(im2, bins=256, range=(0, 4095))
    # Normalize the histogram
    H2 = H2.astype("float")
    H2 /= (H2.sum() + 1e-7)
    H2 = H2.astype("float32")

    result = cv2.compareHist(H1, H2, OPENCV_METHODS[method])

    return result