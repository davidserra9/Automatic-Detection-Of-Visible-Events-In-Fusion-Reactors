import cv2
from centroidtracker import CentroidTracker
import trackableobject
from imageAnalysis import extractCharacteristics
import numpy as np
from JSONLoader import saveData
from paths import PROJECT_ROOT

class objectTracker():

    def __init__(self, maxDisappeared=50, maxDistance=10, classify = False):
        self.ct = CentroidTracker(maxDisappeared=maxDisappeared, maxDistance=maxDistance)
        self.trackers = []
        self.trackableObjects = {}
        self.frameID = 0
        self.lastObjects = []
        self.classify = classify
        #ToDo Implement file writing??


    def update (self,frameID,frameDetections, image):
        self.frameID = frameID
        rects = []

        for detection in frameDetections:
            # compute the (x, y)-coordinates of the bounding box
            # for the object [detection[2], detection[3], detection[4], detection[5]]
            startY = int(detection[3])
            startX = int(detection[2])
            endX = int(detection[4])
            endY = int(detection[5])
            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

        (objects, bboxes) = self.ct.update(rects)

        det_impr = []
        
        if len(frameDetections)>0:
            for (trackID, bbox) in zip(objects.keys(), bboxes.values()):
                print('FrameDetections',frameDetections)
                det_impr.append([frameDetections[0][0], trackID, bbox[0], bbox[1], bbox[2], bbox[3], 0])
            saveData(PROJECT_ROOT + '/data/experiment/Frames/' + str(frameDetections[0][0]) + '.json', det_impr)

        
        # loop over the tracked objects
        newObjects = []
        for (objectID, centroid) in objects.items():

            detection = bboxes[objectID]
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)

            imgCrop = image[int(detection[1]): int(detection[3]),
                           int(detection[0]): int(detection[2])]

            # if there is no existing trackable object, create one
            result = [0]
            characteristics = []
            if self.classify:
                frame_detection = [detection[0], detection[1], detection[2], detection[3]]
                characteristics = extractCharacteristics(image, frame_detection)
                #print(self.clf)
                result = self.clf.predict(characteristics)
            if to is None:
                frame_detection = [detection[0], detection[1], detection[2], detection[3]]
                to = trackableobject.TrackableObject(objectID, centroid, self.frameID, imgCrop, result[0], frame_detection, characteristics, image)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                frame_detection = [detection[0], detection[1], detection[2], detection[3]]
                to.update(frameID, centroid, imgCrop, result[0], frame_detection, characteristics, image)

            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to
            newObjects.append([objectID, centroid[0], centroid[1], int(result[0])])

            """"
            # Write bbox to file
            xmin = bboxes[objectID][0]
            ymin = bboxes[objectID][1]
            xmax = bboxes[objectID][2]
            ymax = bboxes[objectID][3]
            fout_pos.write('{} {} {} {} {} {}\n'.format(frameID, objectID, int(round(xmin, 0)),
                                                        int(round(ymin, 0)), int(round(xmax - xmin, 0)),
                                                        int(round(ymax - ymin, 0))))
            """
        self.lastObjects = newObjects

    def plotTracks(self, image):

        if len(image.shape)==2:

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        for element in self.lastObjects:
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            objectID = element[0]
            centroid = element[1:3]
            result = element[3]
            text = "ID {}".format(objectID)
            if result == 0: #0--> False Positive
                color = (0,255,0)
            else:# 1 --> HotSpot
                color = (0,0,255)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, color, -1)
        return image


    def saveTracks(self, filename):
        # write tracks
        trackFile= open(filename, 'w')
        trackID = 0
        for objectID in sorted(self.trackableObjects.keys()):
            startFrame = self.trackableObjects[objectID].startFrame
            endFrame = self.trackableObjects[objectID].endFrame
            conf = endFrame - startFrame + 1

            out_str = '{} {:06d} {:06d} {:06d} {:06d} {}\n'.format(1,
                                                                   startFrame,
                                                                   endFrame,
                                                                   trackID,
                                                                   objectID, conf)
            trackFile.write(out_str)
            trackID += 1
        trackFile.close()


    def saveData(self, filename = "test.txt"):
        trackFile = open(filename, 'w')
        for objectID in sorted(self.trackableObjects.keys()):
            trackData = self.trackableObjects[objectID].histDist
            for frameData in trackData:
                frameID =  frameData[0]
                histDist = frameData[1]

                out_str = '{} {:06d} {:06.4f} {:06d} \n'.format(1,
                                                                   frameID,
                                                                   histDist,
                                                                   objectID)
                trackFile.write(out_str)

        trackFile.close()


    def saveResults(self, filename = "test.txt"):
        trackFile = open(filename, 'w')
        for objectID in sorted(self.trackableObjects.keys()):
            percentage = self.trackableObjects[objectID].countResults()

            out_str = ' {:06d} {:03.4f} \n'.format(objectID, percentage)
            trackFile.write(out_str)

        trackFile.close()

    def generateOutput(self, filename="output.txt"):
        trackFile = open(filename, 'w')
        for objectID in sorted(self.trackableObjects.keys()):
            percentage = self.trackableObjects[objectID].countResults()
            initFrame = self.trackableObjects[objectID].startFrame
            finalFrame = self.trackableObjects[objectID].endFrame
            deltaFrames = finalFrame - initFrame
            centroids = np.asarray(self.trackableObjects[objectID].centroids)
            xCentroid = np.mean(centroids[:, 0])
            yCentroid = np.mean(centroids[:, 1])
            trackData = self.trackableObjects[objectID].histDist
            count = 0
            threshold = 5
            for frameData in trackData:
                histDist = frameData[1]
                if histDist > threshold:
                    count = count + 1

            out_str = ' {:06d} {:03.4f} {:06d} {:06d} {:06d} {:06f} {:06f} {:06d} \n' \
                .format(objectID,
                        percentage,
                        initFrame,
                        finalFrame,
                        deltaFrames,
                        xCentroid,
                        yCentroid,
                        count)
            trackFile.write(out_str)

        trackFile.close()