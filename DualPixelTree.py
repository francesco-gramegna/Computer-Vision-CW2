
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def DPabs(col, Y):
    Y = np.abs(Y - col[:, None])
    return Y

def DPsub(col, Y):
    Y = (col[:,None] - Y)
    return Y


class DPForest():
    def __init__(self, 
                 n_trees,
             max_samples,
             min_samples_split,
             max_features,
             dual_pixel_test_method,
             commutative
                 ):

        self.method = dual_pixel_test_method
        self.commutative = commutative
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        self.tree = RandomForestClassifier(n_estimators = self.n_trees,
                                            max_samples = self.max_samples,
                                           min_samples_split= self.min_samples_split,
                                           max_features=self.max_features,
                                           )

    
    def compute_dual_pixel(self,X):
        X = np.array(X)
        #for each data point in each sample, we create all the dual pixel possible, and feed them to the tree
        features = X.shape[1]
        newLen = features * (features - 1) if not self.commutative else (features * (features-1) // 2)
        Y = np.zeros((X.shape[0], newLen))

        counter = 0
        for i in range(X.shape[1] ):
            toTake = X.shape[1] - i - 1
            if(toTake == 0):
                break
            
            f1 = X[:, i]
            X_f = X[:, i+1:] 

            X_f = self.method(f1 ,X_f)
            
            Y[: , counter:counter+toTake] = X_f
            counter += toTake

        if(not self.commutative):
            X = X.T
            for i in range(X.shape[1] ):
                toTake = X.shape[1] - i - 1
                if(toTake == 0):
                    break
            
                f1 = X[:, i]
                X_f = X[:, i+1:] 

                X_f = dual_pixel_test_method(f1 ,X_f)
            
                Y[: , counter:counter+toTake] = X_f
            counter += toTake

        return Y

                
    def fit(self,X, Y):
        X = self.compute_dual_pixel(X)

        self.tree.fit(X,Y)


    def predict(self,X):
        X = self.compute_dual_pixel(X)
        return self.tree.predict(X)

    def score(self, X,y):
        X = self.compute_dual_pixel(X)
        score = self.tree.score(X,y)
        return score
        




