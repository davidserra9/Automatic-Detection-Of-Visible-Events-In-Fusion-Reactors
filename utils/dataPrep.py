# Custom imports
from hdf5Loader import hdf5Loader
from JSONLoader import loadData
from paths import HDF5_FOLDER, JSON_FOLDER, MAX_SQUARE_LEN, PROJECT_ROOT
from sequenceLinker import seq2num, num2seq
import math
import cv2

def hdf5Initializer(hdf5Names):
    """
    Function to load all the sequence loaders
    :param hdf5Names: list with all the filenames
    :return: list with all de hdf5Loaders
    """
    loaders = []
    for n in hdf5Names:
        loaders.append(hdf5Loader(HDF5_FOLDER + n + '.h5'))

    return loaders


def detectionLoader(jsonNamesList, tracks=False):
    """
    Function to load detections which are in .json files
    :param jsonNamesList: list with all the filenames
    :param tracks:
        if False:   load all the detections
        if True:    load only one detections for each track. Take as the
                    thumbnail the frame of the middle of the track
    """
    detections = []
    labels = []

    # FRAME BY FRAME DETECTIONS
    # Add all the detectons of the file
    if not tracks:
        for seq in jsonNamesList:
            # detections_i = loadData(JSON_FOLDER + seq + '.h5.json')
            detections_i = loadData(PROJECT_ROOT + '/data/experiment/' + seq + '.h5.json')
            seqNum = seq2num(seq)
            for d in detections_i:
                detections.append([seqNum, d[0], d[1], d[2], d[3], d[4], d[5], 0])
                labels.append(d[6])

    # TRACK DETECTIONS
    # Add one detection per track (the middle one)
    else:
        for seq in jsonNamesList:

            # Detections of each sequence of the list
            detections_i = loadData(JSON_FOLDER + seq + '.h5.json')
            # Number of the sequence
            seqNum = seq2num(seq)

            # List with all the detections with the same trackID
            sameTrackID_detections = []
            trackID = 0

            for d in detections_i:
                if d[1] == trackID:
                    sameTrackID_detections.append(d)
                else:
                    detections.append([seqNum,
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][0],
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][1],
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][2],
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][3],
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][4],
                                       sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][5],
                                       0])

                    labels.append(sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][6])
                    trackID = d[1]
                    sameTrackID_detections = []
                    sameTrackID_detections.append(d)

            # Add the final track detection
            detections.append([seqNum,
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][0],
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][1],
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][2],
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][3],
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][4],
                               sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][5],
                               0])

            labels.append(sameTrackID_detections[(round(len(sameTrackID_detections) / 2))][6])
            trackID = d[1]
            sameTrackID_detections = []
            sameTrackID_detections.append(d)

    return detections, labels


def sampleTrackDetections(trackDetections, n):
    """
    Function capable of taking n equidistant frames from the entire track
    detections
    :param trackDetections: list with only the detections of one track
    :param n: number of frames to take
    :return: list of detectons with len: n
    """
    # Distance between samples
    delta = len(trackDetections) / (n - 1)

    sample_detections = []

    for i in range(n):
        # Obtain the frameNum of the nth sample
        fNum = math.floor(delta * i)

        # If the last sample exceeds the list limits, reduce one
        if fNum == len(trackDetections):
            fNum = fNum - 1
        sample_detections.append(trackDetections[fNum])

    return sample_detections


def labelCounter(labels):
    """
    Function capable of counting and printing the labels
    :param labels: list of labels
    """
    count_NHS = 0
    count_HS = 0
    count_AN = 0
    for l in labels:
        if l == 0:
            count_NHS = count_NHS + 1
        elif l == 1:
            count_HS = count_HS + 1
        elif l == 2:
            count_AN = count_AN + 1

    print('NHS: {}'.format(count_NHS))
    print('HS: {}'.format(count_HS))
    print('AN: {}'.format(count_AN))
    print('TOTAL: {}'.format(count_NHS + count_HS + count_AN))

def trackLoader(detections):
    """
    :param detections: all the detections
        [seqNum, frameNum, trackID, xmin, ymin, xmax, ymax, repNum]
    :return: list of all the trackIDs [seqNum, trackID, repNum]
    """
    trackIDs = []
    for d in detections:
        if not [d[0], d[2], d[7]] in trackIDs:
            trackIDs.append([d[0], d[2], d[7]])
    
    return trackIDs


def crop_detection(image, detection):
    """
    Function to crop the detection in a square of the max length side and then
    resize it to a square MAX_SQUARE_LEN x MAX_SQUARE_LEN"
    :param image: frame of the sequenc > np.ndarray(1024, 1280)
    :param detection: Bounding Box of the detection > [xmin, ymin, xmax, ymax]
    :return: squared image np.ndarray(MAX_SQUARE_LEN, MAX_SQUARE_LEN)
    """

    # Find the min point of the square detection
    x_diff = detection[2] - detection[0]
    y_diff = detection[3] - detection[1]
    max_len = max(x_diff, y_diff)
    xmin = ((x_diff - max_len) / 2) + detection[0]
    ymin = ((y_diff - max_len) / 2) + detection[1]

    # If image limits exceeded
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if (xmin + max_len) > image.shape[1]:
        xmin = image.shape[1] - max_len
    if (ymin + max_len) > image.shape[0]:
        ymin = image.shape[0] - max_len

    xmin = int(xmin)
    ymin = int(ymin)
    max_len = int(max_len)
    # Crop and resize the image
    crop_image = image[ymin:(ymin + max_len), xmin:(xmin + max_len)]
    resized_image = cv2.resize(crop_image, (MAX_SQUARE_LEN, MAX_SQUARE_LEN), interpolation=cv2.INTER_NEAREST)

    return resized_image