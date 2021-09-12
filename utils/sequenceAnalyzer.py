import h5py
import os
import numpy as np
from tqdm.notebook import tqdm
import time
import math
from image_tools import plotBBoxes, showImage, adjustImage, preProcess, computeDarkImage
from hotSpotDetector import hotSpotDetector
import cv2
from JSONLoader import saveData, loadData
import tracker as tr
import joblib

class sequenceAnalyzer():

    def __init__(self, path, T1, T4, T4e):
        print('Loading sequence file...')
        self.f = h5py.File(path, 'r', driver='core')
        self.name = os.path.basename(path)
        t = time.time()
        self.images = self.f['ROIP']['ROIP1']['ROIP1Data'][()]
        self.timestamps = self.f['ROIP']['ROIP1']['ROIP1W7XTime'][()]
        elapsed = time.time() - t
        self.f.close()
        print('Elapsed time reading: {}'.format(elapsed))


        self.frameT1 = self.estimateFrame(T1)
        self.frameT4 = self.estimateFrame(T4)
        self.frameT4e = self.estimateFrame(T4e)
        self.darkImage = self.computeDarkImage()
        #self.darkImage = computeDarkImage(self.images)
        self.darkImage = self.computeDarkImage_old()
        return

    def computeDarkImage(self):
        # Compute Dark Image
        print('Computing Dark image...')
        t = time.time()
        aux = np.zeros([self.images.shape[0], self.images.shape[1], 30])
        nFrames = self.images.shape[2]
        for i in range(30):
            aux[:, :, i] = (self.images[:, :, nFrames - i - 1])
        darkImage1 = np.median(aux, axis=2)

        for i in range(30):
            aux[:, :, i] = (self.images[:, :, i])
        darkImage2 = np.median(aux, axis=2)

        darkImage = np.minimum(darkImage1, darkImage2)

        max = np.percentile(darkImage, 95)

        darkImage = np.where(darkImage > max, max, darkImage)

        elapsed = time.time() - t
        print('Elapsed time computing dark image: {}'.format(elapsed))

        return darkImage


    def computeDarkImage_old(self):
        print('Computing Dark image...')
        t = time.time()
        images = np.zeros([self.images.shape[0], self.images.shape[1], 30])
        nFrames=self.images.shape[2]
        for i in range(30):
            images[:,:,i] = (self.images[:, :, nFrames -i-1])
        darkImage = np.median(images, axis=2)

        #max = np.percentile(darkImage, 95)
        #darkImage = np.where(darkImage > max, max, darkImage)

        elapsed = time.time() - t
        print('Elapsed time computing dark image: {}'.format(elapsed))
        return darkImage

    def estimateFrame(self, time):

        closest = min(self.timestamps, key=lambda x: abs(int(x) - int(time)))
        frame = np.where(self.timestamps == closest)[0][0]
        return frame

    def processSequence(self, jsonPath, modelPath, showBinarized = False,  trackDetections = False, classifyDetections = False):
        global imagestacked
        print('Processing Sequence image...')
        t = time.time()
        frames = self.images.shape[2]
        image = self.images[:, :, 0] - self.darkImage
        detector = hotSpotDetector(image)
        detections = []
        tracker_sa = tr.objectTracker(maxDisappeared=50, maxDistance=10, classify = classifyDetections)

        for i in tqdm(range(frames)):
            frameImage = self.images[:, :, i] - self.darkImage
            image = preProcess(frameImage)
            if (i >self.frameT1 and i< self.frameT4):

                backgroundSubstraction = False
                frame_detections, binarizedImage = detector.detectHotSpot(image, backgroundSubstraction)
                image = adjustImage(image)

                if trackDetections:
                    aux_detections = []
                    for detection in frame_detections:
                        aux_detections.append(
                            [i, 0, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]),
                             1])

                    tracker_sa.update(i, aux_detections, frameImage)
                    image = tracker_sa.plotTracks(image)
                    binarizedImage = tracker_sa.plotTracks(binarizedImage)


                if showBinarized and i == 100:
                    print('Image returned from the frame {}'.format(350))
                    binarizedImage = plotBBoxes(binarizedImage, frame_detections)
                    image = plotBBoxes(image, frame_detections)
                    image = np.hstack((image, binarizedImage))
                    imagestacked = cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)))

                    #self.showImage(binarizedImage)
                else:
                    image = plotBBoxes(image, frame_detections)
                    #showImage(image)

                for detection in frame_detections:
                    detections.append(
                        [i, 0, int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3]),
                         1])

        elapsed = time.time() - t
        modelFilename = modelPath + 'tracker.joblib'
        # joblib.dump(tracker_sa, modelFilename)
        
        jsonFilename = jsonPath + self.name + '.json'
        # saveData(jsonFilename, detections)

        print('Elapsed time calibrating: {}'.format(elapsed))
        return imagestacked