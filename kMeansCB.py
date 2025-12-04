import time
import numpy as np
import SIFT
from sklearn.cluster import KMeans


class Codebook():

    def __init__(self, size, descr_sizes, equalDescr):
        self.size = size
        self.descr_sizes = descr_sizes
        self.equalDescr = equalDescr

    def fit(self, trainImages, y):
        #y is not needed but for compatibility I put it
        stime = time.time()
        #perform the kmeans clustering to find the centers of the       
        kmeans = KMeans(n_clusters = self.size, max_iter = 10000)

        if(self.equalDescr):

            frames, descr = SIFT.getDenseSIFTByImages(trainImages, y, self.descr_sizes) 

            descriptors = descr.reshape((-1, 128)) 

        else:
            descriptors = SIFT.getDenseSIFT(trainImages)


        kmeans.fit(descriptors)
        #print('KMeans algorithm took ', time.time()-stime)

        self.kmeans = kmeans


    def getHistogram(self, image):

        #we get the decriptor of the image, and then we 
        if(self.equalDescr):
            descriptors = SIFT.getDenseSIFT([image])

        else:
            frames, descr = SIFT.getDenseSIFTByImage(image, self.descr_sizes) 

            descriptors = descr.reshape((-1, 128)) 

        descriptors = self.kmeans.predict(descriptors)

        values, counts = np.unique(descriptors, return_counts=True)

        #convert this into an array
        histogram = np.zeros((self.size))

        for i in range(len(values)):
            histogram[values[i]] = counts[i]

        return histogram


    



