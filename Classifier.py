import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier



class Classifier():
    def __init__(self, codebook, trainX, trainY, twoPixel=False):
        self.codebook = codebook
        self.trainX = trainX
        self.trainY = trainY

        #self.tree = DecisionTreeClassifier(splitter='twoPixel' if twoPixel else 'best')
        self.tree = RandomForestClassifier(n_estimators = 400,
                                            max_samples = 50,
                                           min_samples_split=5,
                                           max_features=200,
                                           splitter='twoPixel' if twoPixel else 'best')


    def fit(self):

        stime = time.time()
        X = [self.codebook.getHistogram(self.trainX[i]) for i in range(len(self.trainX))]
        self.tree.fit(X, self.trainY)

    def predict(self,images):
        #get histogram for each image

        X = [self.codebook.getHistogram(images[i]) for i in range(len(images))]
        return self.tree.predict(X)


    def accuracy(self, X,Y):

        preds = self.predict(X)
        correct = 0
        for p,t in zip(preds, Y):
            if(p == t):
                correct+=1

        return correct/ len(preds)





