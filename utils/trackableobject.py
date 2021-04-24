from imageAnalysis import compareHistograms
import numpy as np

class TrackableObject:
        def __init__(self, objectID, centroid, startFrame, img, result, frame_detection,
                     characteristics, imageThumbnail):
                # store the object ID, then initialize a list of centroids
                # using the current centroid
                self.objectID = objectID
                self.centroids = [centroid]

                # initialize a boolean used to indicate if the object has
                # already been counted or not
                self.counted = False

                
                self.startFrame = startFrame
                self.endFrame   = startFrame

                self.prevImg = img
                self.histDist =[]
                self.results = []
                self.results.append((startFrame, result))

                self.characteristics = []
                self.characteristics.append((startFrame,frame_detection, characteristics))
                self.imageThumbnail = imageThumbnail
                self.prevImageThumbnail = imageThumbnail

        def update (self, frameID, centroid, img, result, frame_detection,
                    characteristics, imageThumbnail):
                self.endFrame = frameID
                self.centroids.append(centroid)
                self.results.append((frameID, result))
                self.histDist.append((frameID, compareHistograms(img, self.prevImg)))
                self.characteristics.append((frameID, frame_detection, characteristics))

                self.prevImg = img

                difImage = imageThumbnail - self.prevImageThumbnail
                energy_threshold = 250
                if (difImage.mean() < energy_threshold):
                        self.imageThumbnail = np.maximum(self.imageThumbnail, imageThumbnail)
                self.prevImageThumbnail = imageThumbnail


        def countResults(self):
                total = len(self.results)
                count = 1
                for result in self.results:
                        if result[1] == 1:
                                count = count + 1
                percentage = 100*(float(count)/float(total))
                return percentage
