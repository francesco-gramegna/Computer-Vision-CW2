import DualPixelTree
import time 
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import kMeansCB
import RFCodebook


class FullClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 cbParams={'size':500},
                 n_trees=10,
                 max_samples=60,
                 min_samples_split=5,
                 max_features=100,
                 splitter='best'):
        self.cbParams = cbParams
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.splitter = splitter


    def fit(self, X, y):
        #create codebook
        cb = kMeansCB.Codebook(**self.cbParams) if 'size' in self.cbParams else RFCodebook.Codebook(**self.cbParams)

        cb.fit(X,y)

        self.codebook = cb

        self.forest = Classifier(cb ,
        self.n_trees ,
        self.max_samples ,
        self.min_samples_split  ,
        self.max_features , 
        self.splitter )

        self.forest = self.forest.fit(X,y)

        return self
        

    def score(self, X, y):
        score = self.forest.score(X,y)
        return score

    def predict(self,images):
        return self.forest.predict(images)




class Classifier():
    
    def __init__(self, codebook, 
             n_trees=100,
             max_samples=10,
             min_samples_split=2,
             max_features=10,
             dp=False,
             commutative = False,
             ):
        self.codebook = codebook
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.max_features = max_features



        if dp == False:
            self.tree = RandomForestClassifier(n_estimators = self.n_trees,
                                            max_samples = self.max_samples,
                                           min_samples_split= self.min_samples_split,
                                           max_features=self.max_features,
                                           )
        else:
            self.tree = DualPixelTree.DPForest(n_trees, max_samples, min_samples_split,max_features,dp, commutative)
            print("HYE", self.tree)

    def fit(self, X, y):
        stime = time.time()
        X = [self.codebook.getHistogram(X[i]) for i in range(len(X))]
        self.tree.fit(X, y)
        return self

    def predict(self,images):
        #get histogram for each image
        X = [self.codebook.getHistogram(images[i]) for i in range(len(images))]
        X = np.array(X)
        return self.tree.predict(X)


    def score(self, X, Y):
        print(self.tree)
        X = [self.codebook.getHistogram(X[i]) for i in range(len(X))]

        X = np.array(X)
        score = self.tree.score(X, Y)
        return score





