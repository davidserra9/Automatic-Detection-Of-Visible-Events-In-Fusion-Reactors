from backgroundSubstractor import background_substractor
import numpy as np
import cv2
import math
import time
MIN_AREA = 300
MAX_AREA = 500000

class hotSpotDetector():
    def __init__(self, image):
        self.prevImage = image
        method = "MOG2"
        sigma_thr = "3"
        rho = "0.1"
        self.bckg_subs = background_substractor(method, sigma_thr=sigma_thr, rho=rho)

    # Malisiewicz et al.
    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(
                                                       overlap > overlapThresh)[
                                                       0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    def process_image(self, image):
        """  Apply  following operations to backgrouns substracted image to improve moving object detection:
                - Thresholding to remove shadows in some models
                - Morphological opening with a 5x5 circular structuring element to reduce noise
                - Morphological closing with a 10x10 circular structuring element to fill objects
                - Apply roi.jog mask
        Returns:
            Processed image
        """

        ret, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return image

    def get_detections(self, image):
        """  Given a binarized image, find connected components as detections
            Applies a non maxima suppression algorithm to merge similar detections
        Returns:
            Detections
        """

        detections = []

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        for i in range(len(stats)):
            if stats[i][4] > MIN_AREA and stats[i][4] < MAX_AREA:
                candidate = [stats[i][0], stats[i][1], stats[i][0] + stats[i][2],
                     stats[i][1] + stats[i][3]]
                if self.checkCandidate(candidate):
                    detections.append(candidate)

        detections = self.non_max_suppression_fast(np.array(detections), 0)

        return detections

    def get_local_detections(self, image):
        """  Given a binarized image, find connected components as detections
            Applies a non maxima suppression algorithm to merge similar detections
        Returns:
            Detections
        """

        detections = []

        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

        for i in range(len(stats)):
            if stats[i][4] > MIN_AREA and stats[i][4] < MAX_AREA:
                candidate = [stats[i][0], stats[i][1], stats[i][0] + stats[i][2],
                     stats[i][1] + stats[i][3]]
                detections.append(candidate)

        detections = self.non_max_suppression_fast(np.array(detections), 0)

        return detections


    def checkCandidate(self, detection):
        inMask = True
        x1 = detection[0]
        y1 = detection[1]
        x2 = detection[2]
        y2 = detection[3]
        if (x2> 100 and x1<1200) and (x1> 100 and x2<1200):
            return True
        return False

    def applyOtsu(self, image):

        ret, binarizedImage = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binarizedImage

    def applyTopHat(self, image):

        size = 80
        #size = 80
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        result = cv2.medianBlur(tophat, 5)
        return result


    def binarizeImageDetect(self, inputImage):
        image = self.applyTopHat(inputImage)
        image = self.applyOtsu(image)
        binarizedImage = self.process_image(image)
        detections = self.get_detections(binarizedImage)
        refinedDetections = []
        increaseFactor = 0.5
        for detection in detections:
            x1 = detection[0]
            y1 = detection[1]
            x2 = detection[2]
            y2 = detection[3]
            w = x2-x1
            h = y2-y1
            x1 = int(x1 - increaseFactor*w)
            y1 = int(y1 - increaseFactor*h)
            x2 = int(x2 + increaseFactor*w)
            y2 = int(y2 + increaseFactor*h)
            croppedImage = inputImage[y1: y2, x1: x2]
            auxImage = inputImage[y1: y2, x1: x2]
            #croppedImage = self.applyTopHat(croppedImage)
            binarizedLocalImage = self.applyOtsu(croppedImage)
            #binarizedLocalImage = self.process_image(croppedImage)
            localDetections = self.get_local_detections(binarizedLocalImage)
            for localDetection in localDetections:
                globalDetection = [localDetection[0]+x1,
                localDetection[1]+y1,
                localDetection[2]+x1,
                localDetection[3]+y1]
                refinedDetections.append(globalDetection)
        return refinedDetections, binarizedImage

    def binarizeImage(self, image):
        image = self.applyTopHat(image)
        image = self.applyOtsu(image)
        return image

    def maskImage(self, image):

        h= image.shape[0]
        w = image.shape[1]
        for i in range(h):
            for j in range(w):
                dist = math.sqrt((i-h/2)**2 + (j-w/2)**2)
                if dist > 550:
                    image[i,j] = 0


        return image

    def detectHotSpot(self,image,backgroundSubstraction):

        detections = []
        self.prevImage = image

        image = self.maskImage(image)

        localBinarizaton = False
        if not localBinarizaton:
            if backgroundSubstraction:
                image = self.bckg_subs.apply(image)
                binarizedImage = self.process_image(image)
            else:
                image = self.binarizeImage(image)
                binarizedImage = self.process_image(image)
            detections = self.get_detections(binarizedImage)
        else:
            detections, binarizedImage = self.binarizeImageDetect(image)

        #detections = []

        #threshold = 0.5
        threshold = 0.8
        detections = self.non_max_suppression_fast(np.array(detections), threshold)
        return detections, binarizedImage
