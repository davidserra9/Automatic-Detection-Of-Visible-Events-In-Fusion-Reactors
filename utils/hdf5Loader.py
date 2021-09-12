import h5py
import os

class hdf5Loader():



    def __init__(self, path):
        f = h5py.File(path, 'r')
        #list(self.f.keys())
        self.dset = f['ROIP']
        self.dset = self.dset['ROIP1']
        self.path = os.path.dirname(path)
        self.filename, _ = os.path.splitext(os.path.basename(path))
        return

    def loadImage(self, nImage):
        image = self.dset['ROIP1Data'][: , : ,nImage]
        return image

    def nFrames(self):
        aux = self.dset['ROIP1Data'][0,0]
        return aux.shape[0]

    def findTime(self, timestamp):
        timestamps = self.dset['ROIP1W7XTime']
        return

    def getPath(self):
        return self.path

    def getFileName(self):
        return self.filename
