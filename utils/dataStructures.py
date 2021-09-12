import torch
from torch.utils.data import Dataset
import numpy as np
import random
import math
from torch.utils.data import DataLoader
import cv2

# Custom imports
from dataPrep import sampleTrackDetections, crop_detection, trackLoader, detectionLoader, hdf5Initializer


class Dataset(Dataset):
    def __init__(self, loaders, labels, detections, n=5,
                 dataAug='N', pixRange=5, degRange=5):
        'Initialization'
        self.loaders = loaders  # List of hdf5Loaders
        self.trackIDs = trackLoader(detections)  # List of all the trackIDs [seqNum, trackID, repNum]
        self.labels = labels  # List of labels
        self.detections = detections  # List of detections
        self.n = n  # Number of frames per track
        self.dataAug = dataAug  # Data Augmentation option
        # 'N': no, 'Y': yes, 'P' percentage
        self.pixRange = pixRange  # Pixels change for Data Aug
        self.degRange = degRange  # Degree change for Data Aug

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.trackIDs)

    def __getitem__(self, index):
        """
        Having all the hdf5Loaders, all the detections and all the labels,
        this function returns a tensor of size torch.Size([n, 3, 224, 224]) from
        the index passed as parameter. The function:
        1. searches for the track in position index
        2. obtains n equispaced frames f the track
        3, crops the detection in each frame
        4. resizes the image into a square of 224
        5. triples the image to have three channels
        6. normalize the images into the range 0-1
        7. transforms the images to tensors

        :return: tensor with torch.Size([n, 3, 224, 224])
        """
        # With the selected index obtain the sequence and trackID of the sample
        seqNum = self.trackIDs[index][0]
        trackID = self.trackIDs[index][1]
        repNum = self.trackIDs[index][2]

        # Get the frame samples of the whole track
        # First, obtain all the detections of the the track in question
        # and the label of the track
        track_detections = []
        for d, l in zip(self.detections, self.labels):
            if d[0] == seqNum and d[2] == trackID and d[7] == repNum:
                track_detections.append(d)
                y = l

            # Then obtain the equispaced samples of the track
        sample_detections = sampleTrackDetections(track_detections, self.n)

        # Load the images for all the sample detections
        # Get the BBoxes for all the sample detections
        # Crop and resize the detection from the image
        # Triple the image to have 3 channels
        # Normalize the image between 0 and 1
        # Transform the image to Tensor
        # Transform the list of images to Tensor

        first_d = True

        if self.dataAug == 'Y':
            rotation = random.uniform(-self.degRange, self.degRange)

            xDisp = random.uniform(-self.pixRange, self.pixRange)
            yDisp = random.uniform(-self.pixRange, self.pixRange)

            xSign = 1
            ySign = 1

        elif self.dataAug == 'P':
            rotation = random.uniform(-self.degRange, self.degRange)

            detMean = np.mean(np.array(sample_detections), axis=0)

            xDisp = int((detMean[5] - detMean[3]) * 0.1)
            yDisp = int((detMean[6] - detMean[4]) * 0.1)
            xSign = 1 if random.random() < 0.5 else -1
            ySign = 1 if random.random() < 0.5 else -1

        for d in sample_detections:

            image = self.loaders[(seqNum - 1)].loadImage(d[1])

            if self.dataAug == 'Y' or self.dataAug == 'P':
                deltaX, deltaY = d[5] - d[3], d[6] - d[4]
                (cX, cY) = (d[3] + deltaX // 2, d[4] + deltaY // 2)
                M = cv2.getRotationMatrix2D((cX, cY), rotation, 1.0)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                BBox = [d[3] + (xSign * xDisp), d[4] + (ySign * yDisp),
                        d[5] + (xSign * xDisp), d[6] + (ySign * yDisp)]
                # print('Pre-BBox',BBox)

                if BBox[0] < 0: BBox[0] = 0
                if BBox[1] < 0: BBox[0] = 0
                if BBox[2] > image.shape[1]: BBox[2] = image.shape[1] - 1
                if BBox[3] > image.shape[0]: BBox[3] = image.shape[0] - 1
                # print('Post-BBox',BBox)
            else:
                BBox = [d[3], d[4], d[5], d[6]]

            resized_img = crop_detection(image, BBox)
            resized_img = np.stack((resized_img,) * 3, axis=-1)
            resized_img = np.moveaxis(resized_img, -1, 0)
            resized_img = ((1 / 4095) * resized_img).astype(np.float16)
            resized_img = torch.Tensor(resized_img)
            resized_img = torch.unsqueeze(resized_img, 0)
            if first_d == True:
                resized_images = resized_img
                first_d = False
            else:
                resized_images = torch.cat((resized_images, resized_img), 0)

        return resized_images, y


def splitDataset(seqs, trainPercentage=0.8):
    """
    Function to split the dataset in training and validation with the percentage
    passed as a parameter. This function maintain the proportion of the labels
    in both of the new datasets.
    :param seqs: list of sequences names to load them
    :param trainPercentage: float with the percentage of the dataset which will
    go to training
    :return train_detections: detections of the training datasetç
    :return train_labels: labels of the training dataset
    :return validation_detections: detections of the validation dataset
    :return validation_labels: labels of the validation dataset
    """
    print("Splitting the dataset into 2 sets of {:.1f}/{:.1f}".format(trainPercentage,
                                                                 1 - trainPercentage))
    # Load 1 detection and 1 label per track
    detections, labels = detectionLoader(seqs, tracks=True)

    # List of the detections of each label
    d_NHS = []
    d_HS = []
    d_AN = []

    # Classify the detections by labels
    for detection, label in zip(detections, labels):
        if label == 0:
            d_NHS.append(detection)
        elif label == 1:
            d_HS.append(detection)
        elif label == 2:
            d_AN.append(detection)

    # The 80% of each class has to go to the train_dataset
    # The other 20% to the validation_dataset
    num_train_NHS = math.floor(trainPercentage * len(d_NHS))
    num_train_HS = math.floor(trainPercentage * len(d_HS))
    num_train_AN = math.floor(trainPercentage * len(d_AN))

    # Shuffle the list and take the first 80% of the detecions and the last 20%
    random.seed(42)
    random.shuffle(d_NHS)
    random.shuffle(d_HS)
    random.shuffle(d_AN)

    # Training dataset and labels
    train_detections = []
    train_labels = []

    train_detections = d_NHS[:num_train_NHS] + d_HS[:num_train_HS] + d_AN[:num_train_AN]
    train_labels = np.concatenate((np.zeros(num_train_NHS, dtype=int),
                                np.ones(num_train_HS, dtype=int),
                                2 * np.ones(num_train_AN, dtype=int)))
    train_labels = train_labels.tolist()

    # Validation dataset and labels
    validation_detections = []
    validation_labels = []

    validation_detections = d_NHS[num_train_NHS:] + d_HS[num_train_HS:] + d_AN[num_train_AN:]
    validation_labels = np.concatenate((np.zeros(len(d_NHS) - num_train_NHS, dtype=int),
                                np.ones(len(d_HS) - num_train_HS, dtype=int),
                                2 * np.ones(len(d_AN) - num_train_AN, dtype=int)))
    validation_labels = validation_labels.tolist()

    # Shuffle both datasets
    z = list(zip(train_detections, train_labels))
    random.shuffle(z)
    train_detections, train_labels = zip(*z)

    z = list(zip(validation_detections, validation_labels))
    random.shuffle(z)
    validation_detections, validation_labels = zip(*z)

    return train_detections, train_labels, validation_detections, validation_labels


def balanceDataset(detections, labels, shuffle=True):
    """
    Function to split the dataset without data augmentation. The function
    repeats the classes with the least number of appearances to level them with
    the class which has the most.
    :param detections: detections
    :param labels: labels
    :param shuffle: if true, shuffle the balanced dataset
    :return detections_balanced: balanced detections
    :returm labels_balanced: balanced detections
    """
    print('Balancing the train dataset...')
    # List of the detections of each label
    d_NHS = []
    d_HS = []
    d_AN = []

    # Classify the detections by labels
    for detection, label in zip(detections, labels):
        if label == 0:
            d_NHS.append(detection)
        elif label == 1:
            d_HS.append(detection)
        elif label == 2:
            d_AN.append(detection)

    # Find the maximum number of detections of the same label
    # and from which label it is
    max_len = max(len(d_NHS), len(d_HS), len(d_AN))

    detections_balanced = []
    labels_balanced = []

    for i in range(math.floor(max_len / len(d_NHS))):
        for d in d_NHS:
            detections_balanced.append([d[0], d[1], d[2], d[3], d[4], d[5], d[6], i])
            labels_balanced.append(0)

    for i in range(math.floor(max_len / len(d_HS))):
        for d in d_HS:
            detections_balanced.append([d[0], d[1], d[2], d[3], d[4], d[5], d[6], i])
            labels_balanced.append(1)

    for i in range(math.floor(max_len / len(d_AN))):
        for d in d_AN:
            detections_balanced.append([d[0], d[1], d[2], d[3], d[4], d[5], d[6], i])
            labels_balanced.append(2)

    if shuffle:
        z = list(zip(detections_balanced, labels_balanced))
        random.shuffle(z)
        detections_balanced, labels_balanced = zip(*z)

    return detections_balanced, labels_balanced


def getDataLoaders(loaders,
                   train_detections, train_labels,
                   validation_detections, validation_labels,
                   train_bs, val_bs,
                   dataAug='N', pixRange=5, degRange=5):
    """
    Funciton to create the train and validation dataloaders
    :param loaders: list of the sequence hdf5Loaders
    :param train_detections: list of train detections
    :param train_labels: list of train labels
    :param validation_detections: list of validation detections
    :param validation_labels: list of validation labels
    :train_bs: batch size of the train DataLoader
    :val_bs: batch size of the validation DataLoader
    :return train_dataloader: train DataLoader
    :return validation_dataloader: validation DataLoader
    """
    print('Creating the DataLoaders...')

    train_dataset = Dataset(loaders, train_labels,
                            train_detections, dataAug=True,
                            pixRange=pixRange, degRange=degRange)
    validation_dataset = Dataset(loaders, validation_labels,
                                 validation_detections)

    # batch sizes are the number of tracks in one batch. This means that the
    # real number of images loaded at the same time is the batch size multiplied
    # by n, e.g. n=5, batch_size=7 -> images per batch = 7 * 5 = 35
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=train_bs,
                                  num_workers=2)

    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=val_bs,
                                       num_workers=2)

    return train_dataloader, validation_dataloader