import cv2
import matplotlib
import matplotlib.pyplot as plt

#matplotlib.use('Agg')

import numpy as np
from image_tools import rescaleImage, adjustGamma

MIN_AREA = 100
MAX_AREA = 500000


class background_substractor():
    """  Wrapper class for applying background substraction models
    """

    def __init__(self, method, sigma_thr=3, rho=0.01, colour_conversion='gray'):

        self.method = method
        if method == 'MOG2':
            self.backSub = cv2.createBackgroundSubtractorMOG2()
        elif method == 'LSBP':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorLSBP()
        elif method == 'GMG':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorGMG()
        elif method == 'KNN':
            self.backSub = cv2.createBackgroundSubtractorKNN()
        elif method == 'GSOC':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorGSOC()
        elif method == 'CNT':
            self.backSub = cv2.bgsegm.createBackgroundSubtractorCNT()
        else:
            self.backSub = cv2.bgsegm.createBackgroundSubtractorMOG()

    def apply(self, image):

        """  Apply background substraction to an image according to the model initialized
        Returns:
            Background substracted image
        """

        return (self.backSub.apply(image))

    def getBackgroungImage(self, image):
        return (self.backSub.getBackgroundImage())


def process_image(image):
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


def get_detections(image):
    """  Given a binarized image, find connected components as detections
        Applies a non maxima suppression algorithm to merge similar detections
    Returns:
        Detections
    """

    detections = []

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    for i in range(len(stats)):
        if stats[i][4] > MIN_AREA and stats[i][4] < MAX_AREA:
            detections.append(
                [stats[i][0], stats[i][1], stats[i][0] + stats[i][2],
                 stats[i][1] + stats[i][3]])

    detections = non_max_suppression_fast(np.array(detections), 0)

    return detections


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
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


def add_detections_gt(image, frame_detections, gtExtractor, frame):
    """  Add bounding boxes to GT and detections on a given image
    Returns:
        Image with rectangles corresponding to GT and detections
    """

    # print detections
    for detection in frame_detections:
        # format detection & GT  [frame, ID, xTopLeft, yTopLeft, xBottomRight, yBottomRight, class]
        cv2.rectangle(
            image,
            (int(detection[0]), int(detection[1])),
            (int(detection[2]), int(detection[3])),
            (0, 0, 255),
            2
        )
    # print gt
    for gtBBOX in gtExtractor.gt:
        if gtBBOX[0] == frame:
            cv2.rectangle(
                image,
                (int(gtBBOX[2]), int(gtBBOX[3])),
                (int(gtBBOX[4]), int(gtBBOX[5])),
                (0, 255, 0),
                2
            )
    return image


def get_frame_bounding_box(detections, frame):
    """  Filter bounding boxes by frame
    Returns:
        List with all bounding boxes corresponding to a given frame
    """

    frame_detections = []
    for j in range(len(detections)):
        if detections[j][0] == frame:
            frame_detections.append(
                [detections[j][2], detections[j][3], detections[j][4],
                 detections[j][5]])

    return frame_detections





def analyze_sequence(method, loader, sigma_thr=3, rho=0.01, write_video = False):
    """  Analyze the video sequence with a given method and color conversion
    """

    bckg_subs = background_substractor(method, sigma_thr=sigma_thr, rho=rho)

    if write_video:
        video_name = 'video_' + method +  '.avi'
        video = cv2.VideoWriter(video_name,
                                cv2.VideoWriter_fourcc('M', 'P', '4', 'S'), 20,
                                #cv2.VideoWriter_fourcc('H', '2', '6', '4'), 10,
                               (1920, 1080))

    detections = []

    frames = loader.nFrames()
    #frames = 100

    for i in range(frames):
        # load the image
        #image = cv2.imread(str(frame_path))
        image = loader.loadImage(i)
        initialImage = rescaleImage(image)
        initialImage = adjustGamma(initialImage, 2)

        image = bckg_subs.apply(image)
        image = process_image(image)

        frame_detections = get_detections(image)

        for detection in frame_detections:
            detections.append(
                [i, 0, detection[0], detection[1], detection[2], detection[3],
                 1])

        image = np.concatenate((initialImage, image), axis=1)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(1)
        if write_video:
            video.write(image)


def run():
    test = True

    methods = [
        'MOG2',
        'LSBP',
        'GMG',
        'GSOC',
        'CNT',
        'MOG',
    ]


    if not test:
        for method in methods:
            analyze_sequence(method)
    else:
        analyze_sequence('MOG2', None, sigma_thr=3)


if __name__ == '__main__':
    verbose = False
    run()