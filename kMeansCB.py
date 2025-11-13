import time
import numpy as np
import SIFT
from sklearn.cluster import KMeans


class Codebook():

    def __init__(self, trainImages, size):
        self.size = size
        self.trainImages = trainImages

    def fit(self):
        stime = time.time()
        #perform the kmeans clustering to find the centers of the       
        kmeans = KMeans(n_clusters = self.size, max_iter = 10000)

        descriptors = SIFT.getDenseSIFT(self.trainImages)
        #reset the train images to gain memory
        self.trainImages = None

        kmeans.fit(descriptors)
        print('KMeans algorithm took ', time.time()-stime)

        self.kmeans = kmeans


    def getHistogram(self, image):

        #we get the decriptor of the image, and then we 
        descriptors = SIFT.getDenseSIFT([image])

        descriptors = self.kmeans.predict(descriptors)

        values, counts = np.unique(descriptors, return_counts=True)

        #convert this into an array
        histogram = np.zeros((self.size))

        for i in range(len(values)):
            histogram[values[i]] = counts[i]

        return histogram


    



